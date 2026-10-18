"""Measure actual neural outputs separately from known-body oracle checks."""
from pathlib import Path
import numpy as np
import torch
import trimesh

from ..avatar_clothing import load_body,write_json
from .geometry import load_pair
from .model import aggregate_markers
from .runtime import load_checkpoint


def point_metrics(prediction,data):
    outer=np.asarray(data['outer_points']);target=np.asarray(data['vectors_inward'])
    vectors=np.asarray(prediction['vectors_inward']);labels=np.asarray(prediction['marker_labels'])
    confidence=np.asarray(prediction['marker_confidence'])
    if vectors.shape!=target.shape or not np.isfinite(vectors).all():
        raise ValueError('Prediction vectors have invalid shape or values')
    if labels.shape!=data['marker_labels'].shape or not np.isfinite(confidence).all():
        raise ValueError('Invalid predicted labels/confidence')
    error=np.linalg.norm(vectors-target,axis=1)
    truth_magnitude=np.linalg.norm(target,axis=1);pred_magnitude=np.linalg.norm(vectors,axis=1)
    directional=(truth_magnitude>.002)&(pred_magnitude>1e-6)
    cosine=np.sum(target[directional]*vectors[directional],axis=1)/(
        truth_magnitude[directional]*pred_magnitude[directional])
    correct=labels==data['marker_labels']
    balanced=[float(correct[data['marker_labels']==label].mean()) for label in np.unique(data['marker_labels'])]
    return {
        'points':len(outer),'vector_rmse_mm':float(np.sqrt(np.mean(error**2))*1000),
        'vector_mean_error_mm':float(error.mean()*1000),'vector_p95_error_mm':float(np.percentile(error,95)*1000),
        'magnitude_mae_mm':float(np.mean(np.abs(truth_magnitude-pred_magnitude))*1000),
        'direction_mean_degrees_cloth_over_2mm':float(np.degrees(np.arccos(np.clip(cosine,-1,1))).mean()) if len(cosine) else None,
        'direction_evaluated_points':int(directional.sum()),
        'label_accuracy':float(correct.mean()),'balanced_label_accuracy':float(np.mean(balanced)),
        'confidence_mae':float(np.mean(np.abs(confidence-data['marker_confidence']))),
        'predicted_label_count':int(len(np.unique(labels))),
    }


def evaluate(pair,checkpoint,output,*,device='cpu',development=False,
             input_body=None,mhr_model=None,fit_body=True):
    torch.set_num_threads(2)
    data,pair_receipt=load_pair(pair,require_training_ready=not development)
    network,checkpoint_receipt=load_checkpoint(checkpoint,device,development)
    marker_sha=checkpoint_receipt['marker_definition_sha256']
    if marker_sha!=pair_receipt['marker_definition_sha256']:
        raise ValueError('Evaluation pair and checkpoint use different marker definitions')
    # The only network input is the observed clothed point cloud. Targets below
    # are used after prediction to score the result, never to guide fitting.
    with torch.inference_mode():
        raw=network(torch.tensor(data['outer_points'][None],dtype=torch.float32,device=device))
    prediction={'vectors_inward':raw['vectors_inward'][0].cpu().numpy(),
                'marker_labels':raw['label_logits'][0].argmax(-1).cpu().numpy(),
                'marker_confidence':raw['confidence'][0].cpu().numpy()}
    inner=data['outer_points']+prediction['vectors_inward']
    markers,valid,support=aggregate_markers(inner,prediction['marker_labels'],prediction['marker_confidence'],
                                          network.configuration['markers'])
    center=data['center_m'];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name in ('predicted-mhr.obj','mhr-inputs.npz','body-fit-report.json','body-preview.json'):
        (output/name).unlink(missing_ok=True)
    training_hashes={item['npz_sha256'] for item in checkpoint_receipt['training_pairs']}
    same_subject=(pair_receipt['source_body_sha256'] in checkpoint_receipt['training_body_hashes'] or
                  pair_receipt.get('subject_id') in checkpoint_receipt.get('training_subject_ids',[]))
    scope=('training_points' if pair_receipt['npz_sha256'] in training_hashes else
           'training_subject_resampled_points' if same_subject else 'unseen_subject')
    report={
        'source':'ACTUAL NEURAL PREDICTION; targets used only for scoring',
        'evaluation_scope':scope,'unseen_subject_accuracy_evaluated':not same_subject,
        'training_ready':False,'development_only':checkpoint_receipt['development_only'],
        'subject_id':pair_receipt.get('subject_id'),'pair_npz_sha256':pair_receipt['npz_sha256'],
        'checkpoint_sha256':checkpoint_receipt['checkpoint_sha256'],
        'checkpoint_steps':checkpoint_receipt['steps'],'points':point_metrics(prediction,data),
        'valid_markers':int(valid.sum()),'missing_marker_ids':np.flatnonzero(~valid).tolist(),
        'body_fit_status':'not_run','body_reconstruction':None,
    }
    if valid.any():
        marker_error=np.linalg.norm(markers[valid]-data['markers'][valid],axis=1)
        report['marker_rmse_mm']=float(np.sqrt(np.mean(marker_error**2))*1000)
        report['marker_p95_error_mm']=float(np.percentile(marker_error,95)*1000)
    np.savez_compressed(output/'predictions.npz',outer_points=data['outer_points']+center,
        inner_points=inner+center,markers=markers+center,marker_valid=valid,marker_support=support,**prediction)
    trimesh.points.PointCloud(inner+center).export(output/'predicted-inner.ply')
    if fit_body:
        if input_body is None or mhr_model is None:raise ValueError('Body fitting/evaluation requires MHR asset and scoring body')
        body,_,body_sha=load_body(input_body)
        if body_sha!=pair_receipt['source_body_sha256']:
            raise ValueError('Scoring body differs from the pair ground truth')
        from ..mhr import load_model
        from .body_fit import MHRMarkerFitter
        try:
            if valid.sum()<24:raise ValueError(f'Only {int(valid.sum())} valid marker labels; minimum 24')
            fitter=MHRMarkerFitter(load_model(Path(mhr_model),threads=2),checkpoint_receipt['marker_definition'])
            mesh,fit,inputs=fitter.fit(markers+center,valid)
            mesh.export(output/'predicted-mhr.obj')
            np.savez_compressed(output/'mhr-inputs.npz',**inputs)
            write_json(output/'body-fit-report.json',fit)
            report['body_fit_status']=fit['status']
            if not np.array_equal(mesh.faces,body.faces):raise ValueError('Predicted and scoring body topology differ')
            distance=np.linalg.norm(mesh.vertices-body.vertices,axis=1)
            # A transparent geometric baseline: zero-identity MHR, uniformly
            # scaled to the observed height and centered in observed bounds.
            # It uses no target body or neural labels to choose its parameters.
            observed=data['outer_points']+center
            prior=fitter.base/100
            baseline_scale=np.ptp(observed[:,1])/np.ptp(prior[:,1])
            baseline=(prior-(prior.max(0)+prior.min(0))/2)*baseline_scale+(observed.max(0)+observed.min(0))/2
            baseline_error=np.linalg.norm(baseline-body.vertices,axis=1)
            report['zero_identity_height_scaled_baseline']={
                'parameters':'height and AABB center from observed input; no shape fitting',
                'v2v_mean_mm':float(baseline_error.mean()*1000),
                'v2v_rmse_mm':float(np.sqrt(np.mean(baseline_error**2))*1000),
            }
            report['body_reconstruction']={
                'alignment':'none; metric world-space errors, no Procrustes correction',
                'v2v_mean_mm':float(distance.mean()*1000),
                'v2v_rmse_mm':float(np.sqrt(np.mean(distance**2))*1000),
                'v2v_p95_mm':float(np.percentile(distance,95)*1000),
                'v2v_max_mm':float(distance.max()*1000),
                'height_error_mm':float(abs(fit['height_m']-np.ptp(body.vertices[:,1]))*1000),
                'predicted_height_m':fit['height_m'],
            }
            write_json(output/'body-preview.json',{
                'source':'ACTUAL NEURAL MARKERS -> OFFICIAL MHR FIT',
                'evaluation_scope':scope,'training_ready':False,'body_fit_status':fit['status'],
                'checkpoint_sha256':checkpoint_receipt['checkpoint_sha256'],
                'pair_npz_sha256':pair_receipt['npz_sha256'],
                'positions':mesh.vertices.tolist(),'faces':mesh.faces.tolist(),
                'vertex_error_mm':(distance*1000).tolist(),
                'ground_truth_positions':body.vertices.tolist(),
            })
        except ValueError as error:
            report['body_fit_status']='failed';report['body_fit_error']=str(error)
    write_json(output/'evaluation.json',report)
    write_json(output/'prediction-preview.json',{
        'label':f'NEURAL PREDICTION · {scope.upper().replace("_"," ")}',
        'training_ready':False,'development_only':checkpoint_receipt['development_only'],
        'training_steps':checkpoint_receipt['steps'],'valid_markers':int(valid.sum()),
        'outer':data['outer_points'].tolist(),'inner':inner.tolist(),'markers':markers[valid].tolist(),
        'labels':prediction['marker_labels'].tolist(),'center_m':center.tolist(),
    })
    return report
