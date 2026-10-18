"""Build a continuous outer garment draft, avoiding inward-folded normal offsets."""
import hashlib
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes

_CACHE={}


def outer_surface(body, name, ease, scale, hem_room=0):
    key=hashlib.sha256(body.vertices.tobytes()+body.faces.tobytes()).hexdigest()
    if key not in _CACHE:
        pitch=.006*scale
        vox=body.voxelized(pitch).fill()
        padding=14
        solid=np.pad(vox.matrix,padding,constant_values=False)
        origin=vox.transform[:3,3]-padding*pitch
        outside=distance_transform_edt(~solid,sampling=pitch).astype(np.float32)
        outside=gaussian_filter(outside,sigma=.45)
        _CACHE.clear()
        _CACHE[key]=(outside,origin,pitch)
    outside,origin,pitch=_CACHE[key]
    y=origin[1]+np.arange(outside.shape[1])*pitch
    offsets=np.full(len(y),ease*scale)
    if name=='pants':
        blend=np.clip((1.025*scale-y)/(.10*scale),0,1)
        offsets=.008*scale*(1-blend)+offsets*blend
    else:
        neck=np.clip((y-1.49*scale)/(.058*scale),0,1)
        offsets=offsets*(1-neck)+.010*scale*neck
        if hem_room:
            # Pattern room for the underlying pants at the overlapping hem.
            offsets+=hem_room*scale*np.clip((1.12*scale-y)/(.10*scale),0,1)
    field=outside-offsets[None,:,None]
    vertices,faces,_,_=marching_cubes(field,level=0,spacing=(pitch,pitch,pitch),allow_degenerate=False)
    mesh=trimesh.Trimesh(vertices+origin,faces,process=True)
    if mesh.volume<0:
        mesh.invert()
    return mesh
