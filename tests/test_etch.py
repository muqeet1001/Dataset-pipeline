"""Geometry leakage, equivariance, stability and body-fitting regressions."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

torch=pytest.importorskip('torch')
from pipeline.etch.geometry import visible_points,topology_digest,load_pair
from pipeline.etch.model import TightnessNetwork,supervised_loss,aggregate_markers,knn_indices
from pipeline.etch.body_fit import MHRMarkerFitter


def test_body_under_closed_clothing_is_not_visible_input():
    body=trimesh.creation.box(extents=[1,1,1])
    clothing=trimesh.creation.box(extents=[1.2,1.2,1.2])
    scene=trimesh.util.concatenate([body,clothing])
    assert not visible_points(scene,body.triangles_center,body.face_normals).any()
    assert visible_points(scene,clothing.triangles_center,clothing.face_normals).all()


def test_group_rotation_and_translation_equivariance():
    torch.set_num_threads(2);torch.manual_seed(17)
    network=TightnessNetwork(markers=8,width=16,neighbours_count=6,head_width=32).eval()
    p=torch.randn(1,24,3)*.25;t=torch.tensor([.3,-.7,1.2])
    with torch.no_grad():
        a=network(p)
        a_group=network.encoder(p,knn_indices(p,6))
        translated=network(p+t)
        for key in ('direction','vectors_inward','magnitude_scaled','confidence','label_logits'):
            torch.testing.assert_close(a[key],translated[key],atol=2e-5,rtol=2e-5)
        for rotation in network.encoder.rotations:
            b=network(p@rotation.T+t)
            torch.testing.assert_close(b['direction'],a['direction']@rotation.T,atol=3e-4,rtol=3e-4)
            b_group=network.encoder(p@rotation.T+t,knn_indices(p@rotation.T+t,6))
            torch.testing.assert_close(a_group.mean(2),b_group.mean(2),atol=2e-5,rtol=2e-5)


def test_all_four_losses_have_finite_gradients_and_zero_skin_is_masked():
    torch.set_num_threads(2);torch.manual_seed(29)
    network=TightnessNetwork(markers=8,width=16,neighbours_count=6,head_width=32)
    p=torch.randn(1,24,3)*.25
    vectors=-p*.1;vectors[:,0]=0
    batch={'vectors_inward':vectors,'marker_labels':torch.arange(24)[None]%8,
           'marker_confidence':torch.full((1,24),.7)}
    loss,parts=supervised_loss(network(p),batch)
    assert set(parts)=={'direction','magnitude','labels','confidence'}
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    batch['vectors_inward']=torch.zeros_like(vectors)
    loss,parts=supervised_loss(network(p),batch)
    assert parts['direction'].item()==0
    assert torch.isfinite(loss)


def test_marker_voting_avoids_underflow_and_excludes_missing_markers():
    points=np.array([[1,2,3],[9,9,9],[3,2,1],[50,50,50]],dtype=float)
    markers,valid,support=aggregate_markers(points,[0,0,0,1],[1e-20,1e-30,1e-20,0],count=3)
    np.testing.assert_allclose(markers[0],[2,2,2],atol=1e-8)
    np.testing.assert_array_equal(valid,[True,False,False])
    assert support[0]==3


def test_training_resume_preserves_optimizer_and_sampling_state(tmp_path):
    from pipeline.etch.runtime import train,load_checkpoint
    rng=np.random.default_rng(42);points=(rng.normal(size=(32,3))*.3).astype('f4')
    vectors=-points*.05
    pair=tmp_path/'pair.npz';definition=tmp_path/'markers.json'
    definition.write_text(json.dumps({'count':8}))
    np.savez(pair,outer_points=points,inner_points=points+vectors,vectors_inward=vectors,
             marker_labels=np.arange(32)%8,marker_confidence=np.full(32,.6,dtype='f4'))
    pair.with_suffix('.json').write_text(json.dumps({'training_ready':False,'subject_id':'unit',
        'source_body_sha256':'unit','npz_sha256':hashlib.sha256(pair.read_bytes()).hexdigest(),
        'marker_definition_sha256':hashlib.sha256(definition.read_bytes()).hexdigest()}))
    options={'points':24,'width':16,'head_width':32,'development':True,'learning_rate':1e-3}
    train([pair],definition,tmp_path/'direct.pt',steps=4,**options)
    train([pair],definition,tmp_path/'partial.pt',steps=2,**options)
    resumed=train([pair],definition,tmp_path/'resumed.pt',steps=4,resume=tmp_path/'partial.pt',**options)
    assert len(resumed['history'])==4 and not resumed['rotation_augmentation']
    direct,_=load_checkpoint(tmp_path/'direct.pt',development=True)
    restored,_=load_checkpoint(tmp_path/'resumed.pt',development=True)
    # Float32 SVD has a nonunique internal basis; compare predictions and loss,
    # rather than demanding bit-identical null-space parameters after reload.
    xyz=torch.tensor(points[:24][None])
    with torch.no_grad():
        first=direct(xyz);second=restored(xyz)
    for key in first:torch.testing.assert_close(first[key],second[key],atol=2e-5,rtol=2e-5)
    saved_direct=torch.load(tmp_path/'direct.pt',weights_only=True)
    saved_resumed=torch.load(tmp_path/'resumed.pt',weights_only=True)
    assert saved_direct['numpy_rng_state']==saved_resumed['numpy_rng_state']
    assert torch.equal(saved_direct['torch_rng_state'],saved_resumed['torch_rng_state'])
    assert all(int(state['step'])==4 for state in saved_resumed['optimizer_state']['state'].values())


def test_neural_metrics_measure_errors_in_millimetres():
    from pipeline.etch.evaluation import point_metrics
    data={'outer_points':np.array([[0,0,0],[0,1,0.]]),
          'vectors_inward':np.array([[.01,0,0],[0,0,.02]]),
          'marker_labels':np.array([0,1]),'marker_confidence':np.array([.9,.8])}
    exact={'vectors_inward':data['vectors_inward'].copy(),
           'marker_labels':data['marker_labels'].copy(),'marker_confidence':data['marker_confidence'].copy()}
    score=point_metrics(exact,data)
    assert score['vector_rmse_mm']==0 and score['label_accuracy']==1
    exact['vectors_inward']=exact['vectors_inward']+np.array([.003,.004,0])
    exact['marker_labels']=np.array([0,0])
    score=point_metrics(exact,data)
    assert score['vector_rmse_mm']==pytest.approx(5)
    assert score['label_accuracy']==.5


def test_all_fits_of_one_body_stay_in_the_same_dataset_split():
    from pipeline.etch.runtime import subject_split
    records=[{'source_body_sha256':body} for body in ('a','a','b','b','c','c')]
    train,validation=subject_split(records)
    assert train and validation
    assert not {records[i]['source_body_sha256'] for i in train}&{
        records[i]['source_body_sha256'] for i in validation}
    with pytest.raises(ValueError,match='two distinct bodies'):subject_split(records[:2])
    # A moving subject has different body-vertex hashes in different poses.
    # Also join a renamed duplicate through its identical body hash.
    records=[{'subject_id':subject,'source_body_sha256':body} for subject,body in
             [('alice','pose1'),('alice','pose2'),('renamed','pose2'),('bob','other')]]
    train,validation=subject_split(records)
    assert all(i in train for i in (0,1,2)) or all(i in validation for i in (0,1,2))
    assert not {records[i]['subject_id'] for i in train}&{records[i]['subject_id'] for i in validation}


def test_development_pairs_and_corrupt_archives_cannot_enter_training(tmp_path):
    path=tmp_path/'case.npz'
    np.savez(path,outer_points=np.zeros((3,3)),inner_points=np.ones((3,3)),vectors_inward=np.ones((3,3)))
    receipt={'training_ready':False,'npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='development-only'):load_pair(path)
    arrays,_=load_pair(path,False)
    np.testing.assert_equal(arrays['outer_points']+arrays['vectors_inward'],arrays['inner_points'])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):load_pair(path,False)


def test_observed_point_cloud_prediction_cannot_leave_a_stale_body(tmp_path,monkeypatch):
    from pipeline.etch import runtime
    points=np.array([[0,0,0],[0,1.8,0],[.1,.7,0],[-.1,.9,0],[.1,1,.1]])
    scan=tmp_path/'observed.ply';trimesh.points.PointCloud(points).export(scan)
    class Network:
        configuration={'markers':4}
        def __call__(self,xyz):
            return {'vectors_inward':torch.zeros_like(xyz),
                    'label_logits':torch.zeros(1,xyz.shape[1],4),
                    'confidence':torch.full((1,xyz.shape[1]),.5)}
    monkeypatch.setattr(runtime,'load_checkpoint',lambda *args:(Network(),{
        'checkpoint_sha256':'unit-test','development_only':True,'steps':1,
        'marker_definition':{'count':4}}))
    output=tmp_path/'result';output.mkdir()
    for name in ('predicted-mhr.obj','mhr-inputs.npz','body-fit-report.json'):
        (output/name).write_text('stale previous output')
    _,valid,_,report=runtime.predict(scan,tmp_path/'mock.pt',output,count=4,development=True)
    assert report['points']==4 and report['candidate_points_tested']==len(points)
    assert report['observation_scope'].startswith('supplied observed point cloud')
    assert valid.sum()==1
    for name in ('predicted-mhr.obj','mhr-inputs.npz','body-fit-report.json'):
        assert not (output/name).exists()
    with np.load(output/'predictions.npz',allow_pickle=False) as result:
        # No hidden body, extra surface or interpolated point was introduced.
        assert max(min(np.linalg.norm(points-p,axis=1)) for p in result['outer_points'])<1e-6
        np.testing.assert_allclose(result['outer_points'],result['inner_points'])


def test_neutral_mhr_fitter_recovers_similarity_and_identity_without_cloth():
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=70)
    rng=np.random.default_rng(3);basis=rng.normal(size=(20,len(mesh.vertices),3))*.4
    class Model:
        character_torch=SimpleNamespace(mesh=SimpleNamespace(faces=torch.tensor(mesh.faces)))
        def __call__(self,identity,pose,expression,*args):
            v=torch.tensor(mesh.vertices,dtype=torch.float32)[None]+torch.einsum(
                'bi,ivj->bvj',identity[:,:20],torch.tensor(basis,dtype=torch.float32))
            return v,torch.zeros(len(identity),1,3)
    ids=np.arange(86)
    definition={'vertex_indices':ids.tolist(),'topology_sha256':topology_digest(mesh.faces,len(mesh.vertices))}
    fitter=MHRMarkerFitter(Model(),definition)
    identity=np.linspace(-.3,.3,20)
    rotation=Rotation.from_rotvec([.2,-.4,.1]).as_matrix();scale=1.1;translation=np.array([.2,.8,-.3])
    truth=(mesh.vertices+np.einsum('i,ivj->vj',identity,basis))/100
    targets=(truth[ids]@rotation.T)*scale+translation
    recovered,report,_=fitter.fit(targets,max_rmse_m=.001)
    assert report['status']=='neutral_marker_fit_passed'
    assert report['marker_rmse_m']<2e-5
    np.testing.assert_allclose(recovered.vertices,(truth@rotation.T)*scale+translation,atol=6e-5)
    with pytest.raises(ValueError,match='24 valid'):
        fitter.fit(targets,np.arange(86)<10)
