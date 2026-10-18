// Playback of saved solver states. The browser never fabricates cloth motion.
export class DrapePlayback {
  constructor(button,slider,label) {
    this.button=button;this.slider=slider;this.label=label;this.key='';this.token=0;
    button.onclick=()=>{if(!this.data)return;this.playing=!this.playing;
      if(this.playing&&+slider.value>=this.meta.frames-1)slider.value=0;
      this.clock=performance.now();this.start=+slider.value;button.textContent=this.playing?'Pause drape':'Play drape';};
    slider.oninput=()=>{this.playing=false;button.textContent='Play drape';this.show(+slider.value);};
  }
  async bind(root,url) {
    if(this.root===root&&this.key===url)return;
    this.root=root;this.key=url;this.playing=false;this.data=null;const token=++this.token;
    this.button.disabled=true;this.slider.disabled=true;this.label.textContent=url?'Loading recorded drape…':'No recorded drape for this view';
    if(!url)return;
    try {
      const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw Error('Missing drape record');
      const meta=await response.json();
      const binary=await fetch(new URL(meta.positions,url));if(!binary.ok)throw Error('Missing drape positions');
      const data=new Float32Array(await binary.arrayBuffer());if(token!==this.token)return;
      if(data.length!==meta.frames*meta.vertices*3)throw Error('Incomplete drape record');
      const meshes=meta.layout.map(g=>{let mesh;root.traverse(o=>{if(o.isMesh&&o.name===g.name)mesh=o;});
        if(!mesh||mesh.geometry.attributes.position.count!==g.count)throw Error('Drape topology mismatch');
        return {mesh,...g};});
      this.meta=meta;this.meshes=meshes;this.data=data;this.slider.max=meta.frames-1;this.slider.value=meta.frames-1;
      this.button.disabled=false;this.slider.disabled=false;this.button.textContent='Play drape';this.show(meta.frames-1);
    } catch(e) {if(token===this.token)this.label.textContent=e.message;}
  }
  show(frame) {
    if(!this.data)return;
    const offset=frame*this.meta.vertices*3;
    for(const {mesh,count,offset:vertexOffset} of this.meshes) {
      const attr=mesh.geometry.attributes.position;
      attr.array.set(this.data.subarray(offset+vertexOffset*3,offset+(vertexOffset+count)*3));
      attr.needsUpdate=true;mesh.geometry.computeVertexNormals();mesh.geometry.computeBoundingSphere();
    }
    this.slider.value=frame;this.label.textContent=`Recorded physics · ${(this.meta.simulation_frames[frame]/60).toFixed(2)} s · ¼ speed`;
  }
  tick(now) {
    if(!this.playing||!this.data)return;
    const startTime=this.meta.simulation_frames[this.start]/60;
    const target=startTime+(now-this.clock)/1000*.25;
    let frame=this.start;
    while(frame<this.meta.frames-1&&this.meta.simulation_frames[frame+1]/60<=target)frame++;
    if(frame!==+this.slider.value)this.show(frame);
    if(frame===this.meta.frames-1){this.playing=false;this.button.textContent='Play drape';}
  }
}
