"""Turn a verified or explicitly development-only clothing result into pairs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh

from ..avatar_clothing import load_body,write_json
from . import ADAPTER_VERSION,UPSTREAM_COMMIT
from .geometry import pair_targets,topology_digest


def prepare_pair(input_body,fit_folder,marker_file,output,count=5000,seed=1,development=False,anchor_count=6000):
    input_body,fit_folder,output=Path(input_body),Path(fit_folder),Path(output)
    body,metadata,body_hash=load_body(input_body)
    report=json.loads((fit_folder/'fit-report.json').read_text(encoding='utf-8'))
    manifest=json.loads((fit_folder.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    scene_path=fit_folder/'scene.glb'
    scene_sha=hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if report.get('artifacts',{}).get('scene.glb')!=scene_sha:
        raise ValueError('Clothing scene checksum differs from fit report')
    if report.get('input_body_sha256')!=body_hash or manifest.get('body_sha256')!=body_hash:
        raise ValueError('Clothing and source MHR body hashes differ')
    ready=bool(report['quality'].get('numerical_pass') and manifest.get('training_ready',False))
    if not ready and not development:
        raise ValueError('Clothing has not passed training readiness. Use --development only for an explicitly labelled engineering preview.')
    markers=json.loads(Path(marker_file).read_text(encoding='utf-8'))
    loaded=trimesh.load(scene_path,force='scene',process=False)
    original=loaded.geometry['body']
    if not np.array_equal(original.faces,body.faces) or np.max(np.abs(original.vertices-body.vertices))>1e-7:
        raise ValueError('Exported scene does not preserve the source body')
    # Use transformed scene geometry (not bare mesh buffers) for ray visibility.
    meshes=[];garment_meshes=[]
    for node in loaded.graph.nodes_geometry:
        transform,name=loaded.graph[node]
        mesh=loaded.geometry[name].copy();mesh.apply_transform(transform);meshes.append(mesh)
        if name=='body' and np.max(np.abs(mesh.vertices-body.vertices))>1e-7:
            raise ValueError('Scene body transform differs from the ground-truth coordinate system')
        if name in ('shirt','pants'):garment_meshes.append(mesh)
    parts=trimesh.util.concatenate(meshes)
    clothes=trimesh.util.concatenate(garment_meshes)
    arrays,facts=pair_targets(body,parts,clothes,markers,count,seed,anchor_count)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    receipt={'adapter':ADAPTER_VERSION,'upstream_commit_reviewed':UPSTREAM_COMMIT,
        'subject_id':metadata['subject_id'],'fit':report['fit'],'seed':seed,
        'training_ready':ready,'development_only':not ready,
        'contains_released_etch_weights':False,'source_body_sha256':body_hash,'scene_sha256':scene_sha,
        'synthetic_body_fit_scope':metadata.get('body_fit_scope'),
        'source_cloth_status':report['status'],'source_cloth_configuration':report['configuration'],
        'marker_definition_sha256':hashlib.sha256(Path(marker_file).read_bytes()).hexdigest(),
        'body_topology_sha256':topology_digest(body.faces,len(body.vertices)),
        'npz_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),**facts}
    write_json(output.with_suffix('.json'),receipt)
    # Small JSON preview is a deliberate view of ground-truth supervision.
    # It is never represented as neural-network inference.
    preview={'label':'GROUND-TRUTH TARGETS · NOT MODEL PREDICTIONS',
        'training_ready':ready,'development_only':not ready,'subject_id':metadata['subject_id'],
        'outer':arrays['outer_points'].tolist(),'inner':arrays['inner_points'].tolist(),
        'markers':arrays['markers'].tolist(),'labels':arrays['marker_labels'].tolist(),
        'confidence':arrays['marker_confidence'].tolist(),'center_m':arrays['center_m'].tolist()}
    write_json(output.with_name(output.stem+'-preview.json'),preview)
    return receipt
