"""Original two-panel tee and pull-on trouser patterns, in metres.

The 2D cut pieces define membrane rest lengths. Matching labelled boundaries
are sewn by sharing simulation vertices; the 3D arrangement is only a start
state. No body surface triangles are used as garment geometry.
"""
import numpy as np
import trimesh
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point, LineString

PROFILES = {
    'tight': {'chest_ease_m': .025, 'hip_ease_m': .06, 'ankle_circumference_m': .30,'crotch_drop_m':.014},
    'medium': {'chest_ease_m': .115, 'hip_ease_m': .14, 'ankle_circumference_m': .36,'crotch_drop_m':.028},
    'loose': {'chest_ease_m': .235, 'hip_ease_m': .24, 'ankle_circumference_m': .44,'crotch_drop_m':.045},
}


def _panel(corners, tags, spacing):
    """Sample exact paired boundaries plus a deterministic triangular grid."""
    points, labels, segments = [], [], []
    for i, (a, b, tag) in enumerate(zip(corners, np.roll(corners, -1, axis=0), tags)):
        count = max(1, int(np.ceil(np.linalg.norm(b-a)/spacing)))
        segments.append((a,b,tag))
        for j in range(count):
            points.append(a+(b-a)*j/count)
            # Every corner belongs to its outgoing and incoming seam if closed.
            labels.append((i,j,count,tag))
    boundary_count=len(points)
    polygon=Polygon(corners)
    low, high = np.min(corners,axis=0), np.max(corners,axis=0)
    for row,y in enumerate(np.arange(low[1]+spacing*.55,high[1],spacing*np.sqrt(3)/2)):
        for x in np.arange(low[0]+spacing*.5+(row%2)*spacing*.5,high[0],spacing):
            p=Point(x,y)
            if polygon.contains(p) and polygon.boundary.distance(p)>spacing*.32:
                points.append([x,y])
    points=np.asarray(points)
    raw=Delaunay(points).simplices
    cover=polygon.buffer(1e-10)
    faces=np.asarray([f for f in raw if cover.covers(Polygon(points[f]))])
    v=points[faces]
    det=(v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1])-(v[:,1,1]-v[:,0,1])*(v[:,2,0]-v[:,0,0])
    faces[det<0]=faces[det<0][:,[0,2,1]]
    faces=faces[np.abs(det)>1e-10]
    if abs(np.abs(det).sum()/2-polygon.area)>1e-6:
        raise ValueError('Pattern triangulation has a missing boundary segment')
    return points, faces, labels, segments, boundary_count


def _shirt(profile):
    w=(.993+profile['chest_ease_m'])/4
    # A short-sleeve tee with a real front scoop; shoulders carry its weight.
    hem=.885 if profile['chest_ease_m']>.2 else .90
    grade=(profile['chest_ease_m']-.025)/6
    right=np.array([[w,hem],[w,1.305],[.365+grade,1.235],[.43+grade,1.325],
                    [.260+grade,1.520],[.079,1.548]])
    # Closed boundaries: sides, underarms and shoulders. Open: hem, cuffs, neck.
    neck_front=np.array([[.060,1.512],[.032,1.493],[0,1.487],[-.032,1.493],[-.060,1.512]])
    neck_back=neck_front.copy(); neck_back[:,1]=1.540+(neck_back[:,1]-1.487)*.13
    def shape(neck):
        return np.vstack([[-w,hem],right,neck, right[::-1]*[-1,1]])[:-1]
    tags=['open-hem','side-R','underarm-R','open-cuff-R','shoulder-R','shoulder-neck-R']
    tags += ['open-neck']*6
    tags += ['shoulder-neck-L','shoulder-L','open-cuff-L','underarm-L','side-L']
    front,back=shape(neck_front),shape(neck_back)
    assert len(front)==len(tags)==len(back),(len(front),len(tags))
    return front,back,tags


def _pants(profile):
    hip=(1.007+profile['hip_ease_m'])/4
    waist=.936/4+.006
    cuff=profile['ankle_circumference_m']/4
    knee=(.359+profile['hip_ease_m']*.6)/4
    # Both cut pieces include two legs. The paired crotch/inside-leg seam
    # makes one continuous pair of trousers with three open boundaries.
    thigh=(.523+profile['hip_ease_m']*.5)/4
    crotch=profile['body_crotch_y_m']-profile['crotch_drop_m']
    right=np.array([[waist,1.035],[hip,.89],[.18+thigh,.745],[.22+knee,.47],[.24+cuff,.13],
                    [.24-cuff,.13],[.22-knee,.47],[.18-thigh,.745],
                    [.035,crotch-.018],[0,crotch]])
    corners=np.vstack([[-waist,1.035],right,right[-2::-1]*[-1,1]])[:-1]
    tags=['open-waist','outer-hip-R','outer-thigh-R','outer-leg-R','outer-ankle-R','open-cuff-R',
          'inner-ankle-R','inner-leg-R','crotch-R1','crotch-R2',
          'crotch-L2','crotch-L1','inner-leg-L','inner-ankle-L','open-cuff-L',
          'outer-ankle-L','outer-leg-L','outer-thigh-L','outer-hip-L']
    assert len(corners)==len(tags),(len(corners),len(tags))
    return corners,corners.copy(),tags


def _arrange(uv, segments, name, side, profile):
    # Rounded, collision-free-ish arrangement around torso and arms. This
    # does not set the rest metric and is never used to alter final exports.
    xy=uv.copy()
    if name=='shirt':
        xy[:,0]*=.84
        closed=[LineString([a,b]) for a,b,tag in segments if not tag.startswith('open')]
        d=np.array([min(s.distance(Point(p)) for s in closed) for p in uv])
        depth=np.where(np.abs(uv[:,0])>.25,.085,.180 if side==1 else .145)
        lower=np.clip((1.12-uv[:,1])/.12,0,1)
        depth=depth*(1-lower)+(.22 if side==1 else .21)*lower
        z=side*depth*np.sqrt(np.clip(d/.075,0,1))+.015
    else:
        y=uv[:,1]
        shift=np.interp(y,[.13,.47,.745,.89,1.035],[.015,.007,-.016,0,0])
        xy[:,0]=uv[:,0]*.70+np.sign(uv[:,0])*shift
        closed=[LineString([a,b]) for a,b,tag in segments if not tag.startswith('open')]
        d=np.array([min(s.distance(Point(p)) for s in closed) for p in uv])
        depth=np.interp(y,[.13,.47,.745,.89,1.035],[.062,.085,.115,.160,.155])
        halfwidth=np.interp(y,[.13,.47,.745,.89,1.035],
            [profile['ankle_circumference_m']/4,(.359+profile['hip_ease_m']*.6)/4,
             (.523+profile['hip_ease_m']*.5)/4,(1.007+profile['hip_ease_m'])/4,.24])
        zcenter=np.interp(y,[.13,.47,.745,.89,1.035],[-.067,-.055,0,.015,.015])
        t=np.clip(d/halfwidth,0,1)
        z=zcenter+side*depth*np.sqrt(t*(2-t))
    return np.column_stack([xy,z])


def draft(body,fit,face_budget=3500):
    scale=float(body.extents[1])/1.776
    # Upward rays in the gap between the legs measure the actual lowest
    # central body surface, rather than placing the seam inside the pelvis.
    z=np.linspace(-.02,.06,5)*scale
    origins=np.column_stack([np.zeros(len(z)),np.zeros(len(z)),z])
    points,_,_=body.ray.intersects_location(origins,np.tile([0,1,0],(len(z),1)),multiple_hits=True)
    hits=points[(points[:,1]>.6*scale)&(points[:,1]<1.05*scale)]
    if not len(hits):raise ValueError('Cannot measure the upright avatar crotch clearance')
    profile={**PROFILES[fit],'body_crotch_y_m':float(hits[:,1].min()/scale)}
    garments=[]
    for name, pattern in [('shirt',_shirt),('pants',_pants)]:
        front,back,tags=pattern(profile)
        # Face budget is approximate; never decimate sewing boundaries.
        area=Polygon(front).area+Polygon(back).area
        spacing=np.sqrt(area/(face_budget*.433))
        charts=[]; chart_faces=[]; positions=[]; faces=[]; seam_lookup={}; supports=[]
        for side,corners in [(1,front),(-1,back)]:
            uv,tri,labels,segments,n_boundary=_panel(corners,tags,spacing)
            arranged=_arrange(uv,segments,name,side,profile)
            mapping=[]
            for i,p in enumerate(uv):
                # Label by the actual corner/edge so the shorter front neckline
                # may differ while every closed sewing segment remains paired.
                seam_key=None
                if i<n_boundary:
                    seg,j,count,tag=labels[i]
                    if j==0 and (not tag.startswith('open') or not tags[seg-1].startswith('open')):
                        seam_key=('corner',seg)
                    elif not tag.startswith('open'):
                        seam_key=(tag,j,count)
                if seam_key is not None and seam_key in seam_lookup:
                    index=seam_lookup[seam_key]
                    positions[index]=(positions[index]+arranged[i])/2
                else:
                    index=len(positions);positions.append(arranged[i]);supports.append(name=='pants' and i<n_boundary and abs(p[1]-1.035)<1e-7)
                    if seam_key is not None:seam_lookup[seam_key]=index
                mapping.append(index)
            mapping=np.asarray(mapping)
            if side==-1:tri=tri[:,[0,2,1]]
            faces.extend(mapping[tri]);chart_faces.extend(tri+sum(len(c) for c in charts));charts.append(uv)
        positions=np.asarray(positions)*scale
        mesh=trimesh.Trimesh(positions,np.asarray(faces),process=False)
        charts=np.vstack(charts)*scale; chart_faces=np.asarray(chart_faces)
        # Move only intersecting placement points out before stepping; keep
        # the input avatar and flat rest charts untouched.
        for _ in range(6):
            closest,distance,ids=trimesh.proximity.closest_point(body,mesh.vertices)
            signed=trimesh.proximity.signed_distance(body,mesh.vertices)
            inside=signed>-.004
            if not inside.any():break
            direction=mesh.vertices-closest
            length=np.linalg.norm(direction,axis=1)
            direction*=np.where(signed>0,-1,1)[:,None]
            direction/=np.maximum(length,1e-9)[:,None]
            mesh.vertices[inside]=closest[inside]+direction[inside]*.005
        # Rest edge lengths come from the cut fabric, not the 3D arrangement.
        lengths={}
        for f,uvf in zip(mesh.faces,charts[chart_faces]):
            for a,b in [(0,1),(1,2),(2,0)]:
                key=tuple(sorted((int(f[a]),int(f[b]))))
                length=np.linalg.norm(uvf[b]-uvf[a])
                if key in lengths and abs(lengths[key]-length)>1e-6:
                    raise ValueError('Paired sewing edges have unequal cut lengths')
                lengths[key]=length
        garments.append({'name':name,'mesh':mesh,'panel_vertices':charts,'panel_indices':chart_faces,
            'pinned':np.asarray(supports,dtype=bool),
            'rest_edge_lengths':np.array([lengths[tuple(e)] for e in mesh.edges_unique]),
            'draft':{'method':'two flat cut panels with paired sewn side, shoulder and crotch boundaries',
                     'pattern_scope':'original short-sleeve tee and elastic-waist trousers',
                     'profile':profile,'flat_area_m2':area*scale**2,'target_spacing_m':spacing*scale,
                     'support_type':'waistband edge only' if name=='pants' else 'none; shoulder and body contact',
                     'support_vertices':int(sum(supports)),'bending_rest_angle_rad':0,
                     'fabric_scope':'assumed isotropic cotton-like preset; not calibrated',
                     'landmark_scope':'supplied upright male MHR A-pose'}})
    return garments
