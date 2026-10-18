"""Train from scratch and predict using the explicitly MHR-native adapter."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

from ..avatar_clothing import write_json
from . import ADAPTER_VERSION,UPSTREAM_COMMIT
from .geometry import load_pair,visible_surface_sample
from .model import TightnessNetwork,supervised_loss,aggregate_markers


def subject_split(records,development=False):
    # Join by either subject ID or body hash. Pose-dependent hashes alone would
    # let frames from one person leak across the split; renamed copies must
    # also stay together even if their subject IDs differ.
    parent=list(range(len(records)));seen={}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,record in enumerate(records):
        keys=[('body',record['source_body_sha256'])]
        if record.get('subject_id'):keys.append(('subject',record['subject_id']))
        for key in keys:
            if key in seen:parent[root(i)]=root(seen[key])
            seen[key]=i
    membership={}
    for i in range(len(records)):membership.setdefault(root(i),[]).append(i)
    def group_key(ids):
        return hashlib.sha256('|'.join(sorted(
            records[i]['source_body_sha256'] for i in ids)).encode()).hexdigest()
    groups=sorted(membership.values(),key=group_key)
    if len(groups)<2:
        if not development:raise ValueError('Need at least two distinct bodies/subjects for a held-out evaluation')
        return list(range(len(records))),[]
    held_out={i for ids in groups[:max(1,len(groups)//5)] for i in ids}
    validation=[i for i in range(len(records)) if i in held_out]
    training=[i for i in range(len(records)) if i not in held_out]
    return training,validation


def train(paths,marker_definition,output,steps=1000,points=5000,width=32,
          device='cpu',development=False,seed=7,learning_rate=1e-4,rotation_augmentation=False,
          resume=None,head_width=128):
    if min(steps,points)<1:raise ValueError('Steps and points must be positive')
    paths=[Path(p) for p in paths];definition=json.loads(Path(marker_definition).read_text())
    marker_sha=hashlib.sha256(Path(marker_definition).read_bytes()).hexdigest()
    records=[]
    for path in paths:
        receipt=json.loads(path.with_suffix('.json').read_text())
        if not receipt.get('training_ready') and not development:
            raise ValueError(f'{path}: development-only data cannot enter a normal training run')
        if receipt['marker_definition_sha256']!=marker_sha:raise ValueError('Training marker definitions differ')
        records.append(receipt)
    if not paths:raise ValueError('No prepared pairs were supplied')
    train_ids,val_ids=subject_split(records,development)
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    network=TightnessNetwork(markers=definition['count'],width=width,head_width=head_width).to(device)
    optimizer=torch.optim.Adam(network.parameters(),lr=learning_rate)
    history=[];started=time.perf_counter();order=np.asarray(train_ids)
    if resume is not None:
        network,previous=load_checkpoint(resume,device,development)
        if previous['marker_definition_sha256']!=marker_sha:raise ValueError('Resume marker definitions differ')
        if previous['training_pairs']!=[{'file':str(paths[i]),'npz_sha256':records[i]['npz_sha256']} for i in train_ids]:
            raise ValueError('Resume requires the same training pairs')
        optimizer=torch.optim.Adam(network.parameters(),lr=learning_rate)
        saved=torch.load(resume,map_location=device,weights_only=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        for group in optimizer.param_groups:group['lr']=learning_rate
        torch.set_rng_state(saved['torch_rng_state'].cpu());rng.bit_generator.state=saved['numpy_rng_state']
        order=np.asarray(saved['training_order']);history=previous['history']
        if previous['rotation_augmentation']!=rotation_augmentation:raise ValueError('Resume augmentation setting changed')
        if steps<=len(history):raise ValueError('Requested total steps must exceed the resumed checkpoint')

    def batch(index,augment):
        data,_=load_pair(paths[index],require_training_ready=not development)
        n=min(points,len(data['outer_points']));choice=rng.choice(len(data['outer_points']),n,replace=False)
        xyz=data['outer_points'][choice].copy();vectors=data['vectors_inward'][choice].copy()
        if augment:
            rotation=Rotation.random(random_state=rng).as_matrix().astype('f4')
            xyz=xyz@rotation.T;vectors=vectors@rotation.T
        return {'outer_points':torch.tensor(xyz[None],device=device),
            'vectors_inward':torch.tensor(vectors[None],device=device),
            'marker_labels':torch.tensor(data['marker_labels'][choice][None],device=device),
            'marker_confidence':torch.tensor(data['marker_confidence'][choice][None],device=device)}

    for step in range(len(history),steps):
        if step%len(order)==0:rng.shuffle(order)
        data=batch(int(order[step%len(order)]),rotation_augmentation)
        network.train();optimizer.zero_grad(set_to_none=True)
        total,losses=supervised_loss(network(data['outer_points']),data)
        if not torch.isfinite(total):raise RuntimeError('Nonfinite loss; training stopped')
        total.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        history.append({'step':step+1,'total':float(total.detach()),
                        **{k:float(v.detach()) for k,v in losses.items()}})
        if step==0 or (step+1)%100==0:
            print(f'ETCH adaptation step {step+1}/{steps}: loss {history[-1]["total"]:.4f}',flush=True)
    validation=[];network.eval()
    with torch.inference_mode():
        for index in val_ids[:10]:
            data=batch(index,False);loss,_=supervised_loss(network(data['outer_points']),data)
            validation.append(float(loss))
    receipt={'adapter':ADAPTER_VERSION,'architecture':'independent G60 encoder and XYZ PointTransformer-style scalar heads',
        'official_etch_checkpoint_compatible':False,'upstream_commit_reviewed':UPSTREAM_COMMIT,
        'initialized_from':'random; no pretrained ETCH/ArtEq/CAPE/4D-Dress weights',
        'model_source_sha256':hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(),
        'development_only':development or any(not r['training_ready'] for r in records),
        'training_ready':False,'steps':steps,'points_per_step':points,'seed':seed,'device':str(device),
        'learning_rate':learning_rate,'rotation_augmentation':rotation_augmentation,
        'resumed_from':str(resume) if resume is not None else None,
        'configuration':network.configuration,'marker_definition':definition,'history':history,
        'training_body_hashes':sorted({records[i]['source_body_sha256'] for i in train_ids}),
        'validation_body_hashes':sorted({records[i]['source_body_sha256'] for i in val_ids}),
        'training_subject_ids':sorted({records[i]['subject_id'] for i in train_ids if records[i].get('subject_id')}),
        'validation_subject_ids':sorted({records[i]['subject_id'] for i in val_ids if records[i].get('subject_id')}),
        'training_pairs':[{ 'file':str(paths[i]),'npz_sha256':records[i]['npz_sha256']}
                          for i in train_ids],
        'marker_definition_sha256':marker_sha,
        'validation_mean_multitask_loss':float(np.mean(validation)) if validation else None,
        'body_fitting_accuracy_evaluated':False,'seconds':time.perf_counter()-started,
        'limitations':['Compact backbone differs from the paper EPN/PointTransformer',
            'Multitask loss is not a body reconstruction accuracy measurement',
            'No unseen-subject accuracy claim from a development smoke run']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    torch.save({'adapter':ADAPTER_VERSION,'state_dict':network.state_dict(),
                'configuration':network.configuration,'metadata':receipt,
                'optimizer_state':optimizer.state_dict(),'torch_rng_state':torch.get_rng_state(),
                'numpy_rng_state':rng.bit_generator.state,'training_order':order.tolist()},temporary)
    temporary.replace(output)
    receipt['checkpoint_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    write_json(output.with_suffix('.json'),receipt)
    return receipt


def load_checkpoint(path,device='cpu',development=False):
    path=Path(path);receipt=json.loads(path.with_suffix('.json').read_text())
    if receipt['adapter']!=ADAPTER_VERSION:raise ValueError('This is not a native MHR adaptation checkpoint')
    if receipt.get('development_only') and not development:
        raise ValueError('Checkpoint is a development smoke model; explicit --development is required')
    current_source=hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest()
    if receipt.get('model_source_sha256')!=current_source:
        raise ValueError('Checkpoint network source does not match this revision')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum does not match its receipt')
    saved=torch.load(path,map_location=device,weights_only=True)
    network=TightnessNetwork(**saved['configuration']).to(device)
    network.load_state_dict(saved['state_dict']);network.eval()
    return network,receipt


def predict(scan,checkpoint,output,count=5000,seed=9,device='cpu',development=False):
    if count<2:raise ValueError('Prediction requires at least two points')
    torch.set_num_threads(2)
    network,checkpoint_receipt=load_checkpoint(checkpoint,device,development)
    mesh=trimesh.load(scan,force='scene',process=False)
    surfaces=[]
    for node in mesh.graph.nodes_geometry:
        transform,name=mesh.graph[node];part=mesh.geometry[name].copy();part.apply_transform(transform);surfaces.append(part)
    if not surfaces:raise ValueError('Input contains no surface or observed points')
    all_meshes=all(isinstance(part,trimesh.Trimesh) for part in surfaces)
    all_clouds=all(isinstance(part,trimesh.points.PointCloud) for part in surfaces)
    if not all_meshes and not all_clouds:
        raise ValueError('Use a triangle surface or an observed point cloud; mixed scene types are unsupported')
    surface=trimesh.util.concatenate(surfaces)
    if not .5<float(np.max(surface.extents))<3.5:
        raise ValueError('Expected a human scan in metres; check input units and scale')
    if all_meshes:
        observed,_,_,tested=visible_surface_sample(surface,count,seed)
        observation_scope='externally visible samples from input triangle surface'
    else:
        vertices=np.asarray(surface.vertices)
        if len(vertices)<2 or not np.isfinite(vertices).all():
            raise ValueError('Observed point cloud must contain at least two finite positions')
        observed=vertices[np.random.default_rng(seed).choice(len(vertices),min(count,len(vertices)),replace=False)]
        tested=len(vertices)
        observation_scope='supplied observed point cloud; no hidden body or visibility reconstruction added'
    center=surface.bounds.mean(0);observed=observed-center
    with torch.inference_mode():
        result=network(torch.tensor(observed[None],dtype=torch.float32,device=device))
    vector=result['vectors_inward'][0].cpu().numpy();inner=observed+vector
    labels=result['label_logits'][0].argmax(-1).cpu().numpy();confidence=result['confidence'][0].cpu().numpy()
    markers,valid,support=aggregate_markers(inner,labels,confidence,network.configuration['markers'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # A failed rerun must not leave an old fitted body looking like this run's
    # reconstruction. These are generated body-fit outputs, including evaluation
    # previews if this folder was also used for a scored run.
    for name in ('predicted-mhr.obj','mhr-inputs.npz','body-fit-report.json','body-preview.json'):
        (output/name).unlink(missing_ok=True)
    np.savez_compressed(output/'predictions.npz',outer_points=observed+center,inner_points=inner+center,
        vectors_inward=vector,marker_labels=labels,marker_confidence=confidence,
        markers=markers+center,marker_valid=valid,marker_support=support)
    trimesh.PointCloud(observed+center).export(output/'observed.ply')
    trimesh.PointCloud(inner+center).export(output/'predicted-inner.ply')
    receipt={'adapter':ADAPTER_VERSION,'input_scan':str(Path(scan).resolve()),
        'checkpoint_sha256':checkpoint_receipt['checkpoint_sha256'],
        'development_only':checkpoint_receipt.get('development_only',False),
        'points':len(observed),'candidate_points_tested':tested,'valid_markers':int(valid.sum()),
        'observation_scope':observation_scope,
        'source':'actual native-network predictions; no body mesh used in prediction',
        'body_fit_status':'not_run','training_ready':False}
    write_json(output/'prediction-report.json',receipt)
    write_json(output/'prediction-preview.json',{
        'label':'DEVELOPMENT NETWORK PREDICTION · UNVALIDATED' if receipt['development_only'] else 'NETWORK PREDICTION',
        'development_only':receipt['development_only'],'training_ready':False,
        'training_steps':checkpoint_receipt['steps'],
        'outer':observed.tolist(),'inner':inner.tolist(),'markers':markers[valid].tolist(),
        'labels':labels.tolist(),'confidence':confidence.tolist(),'center_m':center.tolist(),
        'valid_markers':int(valid.sum())})
    return markers+center,valid,checkpoint_receipt['marker_definition'],receipt
