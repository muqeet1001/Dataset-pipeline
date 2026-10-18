"""Run the supplied MHR avatar through real Newton cloth and export three fits."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import trimesh

from .sewing_patterns import draft, PROFILES, STANDARD_FITS, REFERENCE_FITS
from .cloth_quality import validate
from .cloth_receipts import job_lock, reusable

ROOT = Path(__file__).resolve().parents[1]
NEWTON_COMMIT = 'f49482ce07e20764110fd147a67c7334eeae638d'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    temp.replace(path)


def load_body(folder):
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata.get('obj_units') != 'centimetres' or metadata.get('vertical_axis') != 'Y':
        raise ValueError('This adapter requires explicit centimetre OBJ units and Y-up metadata')
    if metadata.get('sex') != 'male':
        raise ValueError('The initial template has only been adapted to the supplied male MHR A-pose')
    path = folder / 'avatar.obj'
    actual = digest(path)
    expected = metadata.get('sha256', {}).get('avatar.obj')
    if expected is not None and actual != expected:
        raise ValueError('Avatar OBJ checksum does not match metadata')
    body = trimesh.load(path, force='mesh', process=False)
    body.apply_scale(.01)
    if not body.is_watertight or not np.isfinite(body.vertices).all():
        raise ValueError('Avatar must be finite and watertight')
    if abs(body.extents[1] * 100 - metadata['height_cm']) > .01:
        raise ValueError('Avatar height does not match metadata')
    return body, metadata, actual


def export_scene(path, body, garments):
    scene = trimesh.Scene()
    for name, mesh, color in [('body', body, [191, 168, 147, 255])] + [
        (g['name'], g['mesh'], g.get('color',[46, 115, 172, 255] if g['name'] == 'shirt' else [49, 57, 72, 255]))
        for g in garments]:
        mesh = mesh.copy()
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=name, baseColorFactor=color, roughnessFactor=.85, metallicFactor=0, doubleSided=True))
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    path.write_bytes(scene.export(file_type='glb', include_normals=True))


def simulate(body, garments, args, progress):
    import newton
    import warp as wp
    from .body_contact import BodyContact
    started = time.perf_counter()
    with wp.ScopedDevice(args.device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.default_edge_ke = 5e-5
        if args.solver == 'style3d':
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        ranges = []
        for garment in garments:
            mesh = garment['mesh']
            start = builder.particle_count
            tri_start, edge_start = builder.tri_count, builder.edge_count
            if args.solver == 'style3d':
                newton.solvers.style3d.add_cloth_mesh(
                    builder, pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), vel=wp.vec3(0, 0, 0),
                    vertices=mesh.vertices.tolist(), indices=mesh.faces.flatten().tolist(), density=.25,
                    panel_verts=garment['panel_vertices'].tolist(),
                    panel_indices=garment['panel_indices'].flatten().tolist(),
                    tri_aniso_ke=wp.vec3(3000, 3000, 1000), edge_aniso_ke=None,
                    tri_kd=.002, edge_kd=.001, particle_radius=.002, label=garment['name'])
            else:
                builder.add_cloth_mesh(pos=wp.vec3(0,0,0), rot=wp.quat_identity(), scale=1,
                    vel=wp.vec3(0,0,0), vertices=mesh.vertices.tolist(), indices=mesh.faces.flatten().tolist(),
                    density=.25, tri_ke=3000, tri_ka=1000, tri_kd=.002, edge_ke=5e-5, edge_kd=.001,
                    particle_radius=.003)
                if builder.tri_count-tri_start!=len(mesh.faces):
                    raise ValueError('Initial garment arrangement contains collapsed triangles')
                # The cut panels, rather than the curved arrangement, define
                # the unstretched material. Cotton cloth bends towards flat.
                uv=garment['panel_vertices'][garment['panel_indices']]
                a,b=uv[:,1]-uv[:,0],uv[:,2]-uv[:,0]
                length=np.linalg.norm(a,axis=1)
                x=np.einsum('ij,ij->i',a,b)/length
                h=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/length
                matrix=np.zeros((len(uv),2,2))
                matrix[:,0,0]=1/length;matrix[:,0,1]=-x/(length*h);matrix[:,1,1]=1/h
                builder.tri_poses[tri_start:]=matrix.tolist()
                area=length*h/2
                builder.tri_areas[tri_start:]=area.tolist()
                masses=np.zeros(len(mesh.vertices))
                np.add.at(masses,mesh.faces.ravel(),np.repeat(area*.25/3,3))
                builder.particle_mass[start:]=masses.tolist()
                builder.edge_rest_angle[edge_start:]=[0.]*(builder.edge_count-edge_start)
            if len(garment.get('elastic_edges',[])):
                if args.solver!='vbd':
                    raise ValueError('The physical elastic-cuff adapter currently requires VBD')
                for edge,rest_length in zip(garment['elastic_edges'],garment['elastic_rest_lengths']):
                    # Springs are in series around each cuff. Scale each edge
                    # by its share of the ring so refinement preserves the
                    # specified whole-ring stiffness, rather than weakening it.
                    circumference=garment['draft']['elastic_cuff_rest_circumference_m']
                    stiffness=80*circumference/float(rest_length)
                    builder.add_spring(start+int(edge[0]),start+int(edge[1]),ke=stiffness,kd=.02,control=0)
                    builder.spring_rest_length[-1]=float(rest_length)
            stop = builder.particle_count
            ranges.append((start, stop))
            for index in np.flatnonzero(garment['pinned']):
                builder.particle_mass[start + int(index)] = 0.0
                builder.particle_flags[start + int(index)] &= ~int(newton.ParticleFlags.ACTIVE)
        builder.add_shape_mesh(body=-1, mesh=newton.Mesh(body.vertices.astype(np.float32), body.faces.flatten().astype(np.int32)),
                               xform=wp.transform_identity())
        if args.solver == 'vbd':
            builder.color()
        model = builder.finalize(device=args.device)
        print(f'  cloth: {model.particle_count} vertices, {model.tri_count} triangles; min mass {min(x for x in builder.particle_mass if x > 0):.3g} kg', flush=True)
        model.set_gravity((0, -9.81, 0))
        model.soft_contact_ke = 3e3
        model.soft_contact_kd = 2.0
        model.soft_contact_mu = .4
        if args.solver == 'style3d':
            solver = newton.solvers.SolverStyle3D(model, iterations=args.iterations, linear_iterations=8)
        else:
            solver = newton.solvers.SolverVBD(model, iterations=args.iterations,
                particle_enable_self_contact=True, particle_self_contact_margin=.005,
                particle_self_contact_gap=.004)
        state, target = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        control = model.control()
        body_contact = BodyContact(body, garments, model) if args.solver == 'vbd' else None
        setup_seconds = time.perf_counter() - started
        last = state.particle_q.numpy().copy()
        initial = last.copy()
        samples=[initial.copy()]
        sample_frames=[0]
        history = []
        settled = False
        solve_started = time.perf_counter()
        for frame in range(args.steps):
            for substep in range(args.substeps):
                state.clear_forces()
                pipeline.collide(state, contacts)
                solver.step(state, target, control, contacts, 1 / 60 / args.substeps)
                if body_contact is not None and ((substep+1)%args.contact_every==0 or substep==args.substeps-1):
                    body_contact.solve(target)
                state, target = target, state
            points = state.particle_q.numpy().copy()
            velocities = state.particle_qd.numpy()
            if not np.isfinite(points).all() or np.max(np.abs(points)) > 10:
                raise RuntimeError(f'Cloth diverged at frame {frame}: bounds {np.nanmin(points,axis=0)} / {np.nanmax(points,axis=0)}')
            movement = np.linalg.norm(points - last, axis=1)
            history.append({'frame': frame + 1, 'p95_movement_m': float(np.percentile(movement,95)),
                            'max_movement_m': float(movement.max()),
                            'p95_speed_m_s': float(np.percentile(np.linalg.norm(velocities,axis=1),95))})
            last = points
            if (frame+1)%2==0:
                samples.append(points.copy());sample_frames.append(frame+1)
            if (frame+1) % 15 == 0:
                print(f"  frame {frame+1}/{args.steps}: p95 movement {history[-1]['p95_movement_m']*1000:.2f} mm", flush=True)
                progress(frame+1, history[-1], points, ranges)
            if frame >= 89 and all(h['p95_movement_m'] < .0005 and h['max_movement_m'] < .003
                                   and h['p95_speed_m_s'] < .03 for h in history[-30:]):
                settled = True
                break
        for garment, (start, stop) in zip(garments, ranges):
            garment['mesh'].vertices = points[start:stop]
        if sample_frames[-1]!=frame+1:
            samples.append(points.copy());sample_frames.append(frame+1)
        return {'setup_seconds': setup_seconds, 'simulation_seconds': time.perf_counter()-solve_started,
                '_trajectory':np.asarray(samples,dtype='<f4'),'sample_frames':sample_frames,
                'steps_completed': frame+1, 'settled': settled, 'history': history,
                'max_actual_simulation_displacement_m': float(np.linalg.norm(points-initial,axis=1).max()),
                'pinned_vertices': sum(int(g['pinned'].sum()) for g in garments),
                'fabric': {'density_kg_m2': .25, 'stretch': [3000,3000,1000], 'isotropic_bending':5e-5,
                           'velocity_damping_per_substep':.97, 'body_contact_margin_m':.004,
                           'self_contact_margin_m':.005,
                           'body_contact_method':'VBD plus sampled MHR triangle-mesh Jacobi constraints',
                           'status':'assumed prototype preset, not physically calibrated'}}


def _run(args):
    folder = args.input.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    body, metadata, body_hash = load_body(folder)
    config = {key: getattr(args,key) for key in ('device','solver','steps','substeps','iterations','face_budget','contact_every')}
    code = {name:digest(Path(__file__).with_name(name)) for name in
            ('avatar_clothing.py','sewing_patterns.py','cloth_quality.py','body_contact.py')}
    key = hashlib.sha256(json.dumps({'body':body_hash,'config':config,'code':code,'profiles':PROFILES},sort_keys=True).encode()).hexdigest()
    manifest = {'subject_id':metadata['subject_id'],'sex':metadata['sex'],'body_height_cm':metadata['height_cm'],
                'body_sha256':body_hash,'units':'metres','device':args.device,'solver':args.solver,
                'newton_commit':NEWTON_COMMIT,'signature':key,'fits':{},'status':'running',
                'versions':{p:importlib.metadata.version(p) for p in ('newton','warp-lang','numpy','trimesh')},
                'body_scope':metadata['body_fit_scope'],'training_ready':False}
    manifest['fit_order']=[fit for fit in PROFILES if fit in args.fits]
    manifest['fit_labels']={fit:PROFILES[fit].get('label',fit.title()) for fit in args.fits}
    manifest['design']='original crew-neck tee and elastic-cuff trousers'
    manifest['reference_match_validated']=False
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        try:
            previous_manifest=json.loads(manifest_path.read_text())
            if previous_manifest.get('signature')==key:
                manifest['fits']=previous_manifest.get('fits',{})
        except (ValueError,OSError):
            pass
    write_json(manifest_path, manifest)
    failed = False
    for fit in args.fits:
        destination=out/fit
        destination.mkdir(exist_ok=True)
        report_path=destination/'fit-report.json'
        previous=reusable(report_path,key) if not args.force else None
        if previous:
            manifest['fits'][fit]={'status':previous['status'],'report':f'{fit}/fit-report.json',
                'scene':f'{fit}/scene.glb','rest':f'{fit}/rest.glb','quality':previous['quality'],
                'total_seconds':previous['total_seconds']}
            print(f'{fit}: verified artifacts reused',flush=True)
            write_json(manifest_path,manifest)
            continue
        started=time.perf_counter()
        print(f'{fit}: drafting outfit on supplied MHR body',flush=True)
        garments=draft(body,fit,args.face_budget)
        for garment in garments:
            np.savez_compressed(destination/(garment['name']+'-rest.npz'),vertices=garment['mesh'].vertices,
                faces=garment['mesh'].faces, panel_vertices=garment['panel_vertices'],panel_indices=garment['panel_indices'],
                pinned=garment['pinned'],elastic_edges=garment.get('elastic_edges',[]),
                elastic_rest_lengths=garment.get('elastic_rest_lengths',[]))
        export_scene(destination/'rest.glb',body,garments)
        manifest['fits'][fit]={'status':'simulating','scene':f'{fit}/rest.glb','report':f'{fit}/fit-report.json'}
        write_json(manifest_path,manifest)
        def progress(frame, last, points, ranges):
            manifest['fits'][fit]['frame']=frame
            manifest['fits'][fit]['p95_movement_mm']=last['p95_movement_m']*1000
            preview=[]
            for g,(a,b) in zip(garments,ranges):
                copy={**g,'mesh':g['mesh'].copy()};copy['mesh'].vertices=points[a:b];preview.append(copy)
            export_scene(destination/'preview.glb',body,preview)
            manifest['fits'][fit]['scene']=f'{fit}/preview.glb?frame={frame}'
            write_json(manifest_path,manifest)
        try:
            simulation=simulate(body,garments,args,progress)
        except Exception as error:
            failed=True
            failure={'status':'failed','signature':key,'fit':fit,'error':str(error),'configuration':config}
            write_json(report_path,failure)
            manifest['fits'][fit]={'status':'failed','report':f'{fit}/fit-report.json',
                'scene':f'{fit}/rest.glb','error':str(error)}
            write_json(manifest_path,manifest)
            print(f'{fit}: failed: {error}',flush=True)
            continue
        print(f'{fit}: validating actual simulated cloth',flush=True)
        trajectory=simulation.pop('_trajectory')
        trajectory.tofile(destination/'drape.f32')
        offset=0;layout=[]
        for g in garments:
            n=len(g['mesh'].vertices);layout.append({'name':g['name'],'offset':offset,'count':n});offset+=n
        write_json(destination/'drape.json',{'format':'little-endian float32 [frame, vertex, xyz]',
            'positions':'drape.f32','frames':len(trajectory),'vertices':offset,
            'simulation_frames':simulation['sample_frames'],'frame_dt_seconds':1/60,
            'layout':layout,'source':'actual Newton substeps; no synthetic animation'})
        export_scene(destination/'scene.glb',body,garments)
        for garment in garments:
            garment['mesh'].export(destination/(garment['name']+'.glb'))
            garment['mesh'].export(destination/(garment['name']+'.obj'))
        reloaded=trimesh.load(destination/'scene.glb',force='scene',process=False)
        exported_body=reloaded.geometry['body']
        body_error=float(np.max(np.abs(exported_body.vertices-body.vertices)))
        if body_error > 1e-7 or not np.array_equal(exported_body.faces,body.faces):
            raise RuntimeError('Export changed original body geometry')
        quality=validate(body,garments,simulation['settled'],simulation['history'])
        status='numerical_pass_unvalidated_fabric' if quality['numerical_pass'] else 'needs_review'
        failed |= not quality['numerical_pass']
        report={'fit':fit,'status':status,'signature':key,'configuration':config,'input_body_sha256':body_hash,
                'body_roundtrip_max_error_m':body_error,'body_faces_unchanged':True,
                'drafts':{g['name']:g['draft'] for g in garments},'simulation':simulation,
                'quality':quality,'total_seconds':time.perf_counter()-started,
                'artifacts':{p.name:digest(p) for p in destination.iterdir()
                             if p.suffix in ('.glb','.npz','.obj','.f32') or p.name=='drape.json'}}
        write_json(report_path,report)
        manifest['fits'][fit]={'status':status,'scene':f'{fit}/scene.glb','rest':f'{fit}/rest.glb',
                              'drape':f'{fit}/drape.json',
                              'report':f'{fit}/fit-report.json','total_seconds':report['total_seconds'],
                              'quality':{k:v for k,v in quality.items() if k not in ('settling_history',)}}
        write_json(manifest_path,manifest)
        print(f'{fit}: {status}; {report["total_seconds"]:.1f}s',flush=True)
    manifest['status']='needs_review' if failed else 'prototype_complete'
    write_json(manifest_path,manifest)
    print(f'Viewer data: {manifest_path}',flush=True)
    return 2 if failed else 0


def run(args):
    with job_lock(args.output):
        try:
            return _run(args)
        except BaseException as error:
            path=args.output.resolve()/'manifest.json'
            if path.exists():
                manifest=json.loads(path.read_text())
                manifest['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
                manifest['error']=str(error)
                write_json(path,manifest)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'inputs/MHR-male-10027')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/local-avatar/MHR-male-10027')
    parser.add_argument('--fits',nargs='+',choices=list(PROFILES))
    parser.add_argument('--preset',choices=['standard','reference'],default='standard')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--solver',choices=['style3d','vbd'],default='vbd')
    parser.add_argument('--steps',type=int,default=180)
    parser.add_argument('--substeps',type=int,default=4)
    parser.add_argument('--contact-every',type=int,default=1,
                        help='Apply supplemental sampled body contact every N substeps; the final substep is always corrected')
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--face-budget',type=int,default=1800)
    parser.add_argument('--force',action='store_true')
    options=parser.parse_args()
    if options.fits is None:
        options.fits=list(REFERENCE_FITS if options.preset=='reference' else STANDARD_FITS)
    if min(options.steps, options.substeps, options.iterations, options.face_budget,options.contact_every)<1:
        parser.error('Steps, substeps, iterations and face budget must be positive')
    raise SystemExit(run(options))
