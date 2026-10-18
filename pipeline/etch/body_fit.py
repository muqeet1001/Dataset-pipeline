"""Native MHR marker fitting for the current fixed neutral-pose dataset.

This bounded identity/similarity fitter is not ETCH's articulated SMPL fitter.
It never substitutes SMPL vertices or optimizes towards the clothing surface.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
import trimesh

from ..mhr import evaluate,topology
from .geometry import topology_digest


class MHRMarkerFitter:
    def __init__(self,model,markers):
        self.model=model;self.markers=markers;self.faces=topology(model)
        with torch.inference_mode():
            self.base=evaluate(model)[0][0].double().numpy()
            zero=torch.zeros(1,45);pose=torch.zeros(1,204);expression=torch.zeros(1,72)
            reference=model(zero,pose,expression,False)[0][0].numpy()
            basis=[]
            # Batch size one keeps this initialization bounded on an 8 GB host.
            for i in range(20):
                identity=zero.clone();identity[0,i]=1
                basis.append(model(identity,pose,expression,False)[0][0].numpy()-reference)
            self.basis=np.asarray(basis,dtype=np.float64)
        if topology_digest(self.faces,len(self.base))!=markers['topology_sha256']:
            raise ValueError('Official MHR asset and marker topology do not match')
        self.indices=np.asarray(markers['vertex_indices'],dtype=int)

    def fit(self,targets,valid=None,max_evaluations=150,max_rmse_m=.015):
        targets=np.asarray(targets,dtype=float)
        if targets.shape!=(len(self.indices),3) or not np.isfinite(targets).all():
            raise ValueError('Expected finite [marker_count,3] target positions in metres')
        valid=np.ones(len(targets),dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
        if valid.shape!=(len(targets),) or valid.sum()<24:
            raise ValueError('Need at least 24 valid, distributed MHR markers for the neutral-pose fitter')
        base=self.base[self.indices[valid]]/100
        basis=self.basis[:,self.indices[valid]]/100
        target=targets[valid]
        if np.ptp(base[:,1])<.6:
            raise ValueError('Marker observations lack sufficient full-body height coverage')
        prior=1e-4

        def values(x):
            shape=base+np.einsum('i,ijk->jk',x[:20],basis)
            rotation=Rotation.from_rotvec(x[24:27]).as_matrix()
            scale=np.exp(x[20]);rotated=shape@rotation.T;predicted=rotated*scale+x[21:24]
            residual=np.r_[(predicted-target).ravel(),prior*x[:20]]
            jac=np.zeros((len(residual),27))
            jac[:len(target)*3,:20]=((basis@rotation.T)*scale).transpose(1,2,0).reshape(-1,20)
            jac[:len(target)*3,20]=(rotated*scale).ravel()
            jac[:len(target)*3,21:24]=np.tile(np.eye(3),(len(target),1))
            for axis in range(3):
                delta=np.zeros(3);delta[axis]=1e-6
                plus=Rotation.from_rotvec(x[24:27]+delta).as_matrix()
                minus=Rotation.from_rotvec(x[24:27]-delta).as_matrix()
                jac[:len(target)*3,24+axis]=((shape@(plus-minus).T)*scale/2e-6).ravel()
            jac[-20:,:20]=prior*np.eye(20)
            return residual,jac

        initial=np.zeros(27)
        u,_,vh=np.linalg.svd((base-base.mean(0)).T@(target-target.mean(0)))
        rotation=vh.T@u.T
        if np.linalg.det(rotation)<0:
            vh[-1]*=-1;rotation=vh.T@u.T
        initial[24:27]=Rotation.from_matrix(rotation).as_rotvec()
        scale_guess=np.linalg.norm(target-target.mean(0))/np.linalg.norm(base-base.mean(0))
        initial[20]=np.log(np.clip(scale_guess,.51,1.49))
        initial[21:24]=target.mean(0)-(base@rotation.T).mean(0)*np.exp(initial[20])
        lower=np.r_[np.full(20,-3.),np.log(.5),np.full(3,-10.),np.full(3,-2*np.pi)]
        upper=np.r_[np.full(20,3.),np.log(1.5),np.full(3,10.),np.full(3,2*np.pi)]
        result=least_squares(lambda x:values(x)[0],initial,jac=lambda x:values(x)[1],
            bounds=(lower,upper),max_nfev=max_evaluations,ftol=1e-10,gtol=1e-10,xtol=1e-10)
        coefficients=torch.zeros(1,45);coefficients[0,:20]=torch.tensor(result.x[:20],dtype=torch.float32)
        with torch.inference_mode():native,skeleton=evaluate(self.model,coefficients)
        native=native[0].double().numpy();scale=float(np.exp(result.x[20]));translation=result.x[21:24]
        rotation=Rotation.from_rotvec(result.x[24:27]).as_matrix()
        estimate=(self.base+np.einsum('i,ijk->jk',result.x[:20],self.basis))
        official_error=float(np.max(np.abs(native-estimate)))/100
        if official_error>1e-5:raise ValueError('Affine marker fitter disagrees with official MHR forward')
        vertices=(native@rotation.T)/100*scale+translation
        error=np.linalg.norm(vertices[self.indices[valid]]-target,axis=1)
        rmse=float(np.sqrt(np.mean(error**2)))
        accepted=bool(result.success and rmse<=max_rmse_m)
        mesh=trimesh.Trimesh(vertices,self.faces,process=False)
        report={'status':'neutral_marker_fit_passed' if accepted else 'needs_review',
            'marker_rmse_m':rmse,'marker_max_error_m':float(error.max()),'valid_markers':int(valid.sum()),
            'height_m':float(np.ptp(native[:,1])*scale/100),'world_y_extent_m':float(mesh.extents[1]),
            'scale':scale,'translation_m':translation.tolist(),'rotation_matrix':rotation.tolist(),
            'identity_coefficients':coefficients[0].tolist(),'official_affine_error_m':official_error,
            'body_model':'official MHR','units':'metres','solver_success':bool(result.success),
            'evaluations':result.nfev,'articulated_pose_recovery':False,
            'scope':'first 20 identity coefficients and global rigid similarity at fixed MHR neutral pose',
            'limitations':['Not the paper\'s articulated SMPL Levenberg-Marquardt fitter',
                           'A successful marker fit does not validate the predicting network']}
        inputs={'identity_coeffs':coefficients.numpy(),'model_parameters':np.zeros((1,204),dtype='f4'),
            'face_expr_coeffs':np.zeros((1,72),dtype='f4'),'scale':np.array(scale),
            'translation_m':translation,'rotation_matrix':rotation,'skeleton_state_native':skeleton.numpy()}
        return mesh,report,inputs
