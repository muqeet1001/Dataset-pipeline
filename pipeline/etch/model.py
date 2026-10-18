"""Fresh MHR-native tightness network following ETCH's intermediate tasks.

The compact PyTorch group-point encoder and distance attention are independent
implementations, NOT the authors' EPN/PointTransformer architecture. The head
separation, 60 rotations, SO(3) projection and four supervised tasks follow the
method. Only the discrete icosahedral rotations have a tested equivariance
guarantee; arbitrary articulated-pose generalization requires evaluation.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F


def knn_indices(points, count):
    n=points.shape[1];k=min(count,n-1)
    if k<1:raise ValueError('Need at least two input points')
    chunks=[]
    with torch.no_grad():
        for start in range(0,n,256):
            distance=torch.cdist(points[:,start:start+256],points)
            own=torch.arange(start,min(start+256,n),device=points.device)
            distance[:,torch.arange(len(own),device=points.device),own]=float('inf')
            chunks.append(distance.topk(k,dim=-1,largest=False).indices)
    return torch.cat(chunks,dim=1)


def neighbours(values, indices):
    batch=torch.arange(len(values),device=values.device)[:,None,None]
    return values[batch,indices]


class GroupPointEncoder(nn.Module):
    def __init__(self,width=32,radius=.4,chunk=128):
        super().__init__();self.radius=radius;self.chunk=chunk
        self.register_buffer('rotations',torch.tensor(Rotation.create_group('I').as_matrix(),dtype=torch.float32))
        self.spatial=nn.Sequential(nn.Linear(8,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.gate=nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.Linear(width,width),nn.Sigmoid())
        self.mix=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,width))

    def coordinates(self,points,indices,start,stop):
        local=neighbours(points,indices[:,start:stop])-points[:,start:stop,None]
        oriented=torch.einsum('bnki,gij->bnkgj',local/self.radius,self.rotations)
        distance=local.norm(dim=-1,keepdim=True)/self.radius
        return oriented,distance

    def forward(self,points,indices):
        n=points.shape[1];centered=points-points.mean(1,keepdim=True);chunks=[]
        for start in range(0,n,self.chunk):
            stop=min(start+self.chunk,n);local,distance=self.coordinates(points,indices,start,stop)
            global_xyz=torch.einsum('bni,gij->bngj',centered[:,start:stop]/self.radius,self.rotations)
            global_xyz=global_xyz[:,:,None].expand(-1,-1,local.shape[2],-1,-1)
            global_distance=centered[:,start:stop].norm(dim=-1)[:, :,None,None,None]/self.radius
            scalars=distance[:,:,:,None].expand(-1,-1,-1,60,-1)
            global_distance=global_distance.expand(-1,-1,local.shape[2],60,-1)
            chunks.append(self.spatial(torch.cat([local,global_xyz,scalars,global_distance],-1)).mean(2))
        first=torch.cat(chunks,1);chunks=[]
        for start in range(0,n,self.chunk):
            stop=min(start+self.chunk,n);local,distance=self.coordinates(points,indices,start,stop)
            gate=self.gate(torch.cat([local,distance[:,:,:,None].expand(-1,-1,-1,60,-1)],-1))
            message=(gate*neighbours(first,indices[:,start:stop])).mean(2)
            chunks.append(self.mix(torch.cat([first[:,start:stop],message],-1)))
        return torch.cat(chunks,1)


class ScalarContext(nn.Module):
    """Vector-channel point attention using invariant distances for position."""
    def __init__(self,width):
        super().__init__();self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width)
        self.position=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width))
        self.attention=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.out=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))

    def forward(self,points,features,indices):
        distance=(neighbours(points,indices)-points[:,:,None]).norm(dim=-1,keepdim=True)
        position=self.position(distance)
        logits=self.attention(self.q(features)[:,:,None]-neighbours(self.k(features),indices)+position)
        value=neighbours(self.v(features),indices)+position
        return features+self.out((logits.softmax(dim=2)*value).sum(2))


class TightnessNetwork(nn.Module):
    def __init__(self,markers=86,width=32,neighbours_count=12,radius=.4):
        super().__init__()
        if width%4:raise ValueError('Feature width must be divisible by four')
        self.configuration={'markers':markers,'width':width,'neighbours_count':neighbours_count,'radius':radius}
        self.encoder=GroupPointEncoder(width,radius)
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=0,batch_first=True,activation='gelu')
        self.direction_attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.direction_weights=nn.Linear(width,1)
        self.magnitude_context=ScalarContext(width);self.marker_context=ScalarContext(width)
        self.magnitude=nn.Linear(width,1);self.labels=nn.Linear(width,markers);self.confidence=nn.Linear(width,markers)

    def forward(self,points):
        if points.ndim!=3 or points.shape[-1]!=3 or not torch.isfinite(points).all():
            raise ValueError('Network input must be finite [batch, points, xyz] in metres')
        indices=knn_indices(points,self.configuration['neighbours_count'])
        group=self.encoder(points,indices);b,n,g,c=group.shape;directions=[]
        tokens=group.reshape(b*n,g,c)
        for start in range(0,b*n,128):
            weights=self.direction_weights(self.direction_attention(tokens[start:start+128])).squeeze(-1)
            matrix=torch.einsum('ng,gij->nij',weights,self.encoder.rotations)
            u,_,vh=torch.linalg.svd(matrix)
            diagonal=torch.ones((len(matrix),3),device=points.device,dtype=points.dtype)
            diagonal[:,-1]=torch.where(torch.linalg.det(u@vh)<0,-1.,1.)
            rotation=(u*diagonal[:,None])@vh
            directions.append(rotation[:,:,2])
        direction=torch.cat(directions).reshape(b,n,3)
        invariant=group.mean(2)
        magnitude=F.softplus(self.magnitude(self.magnitude_context(points,invariant,indices)))
        context=self.marker_context(points,invariant,indices);logits=self.labels(context)
        confidence=(logits.softmax(-1)*self.confidence(context).sigmoid()).sum(-1)
        return {'direction':direction,'magnitude_scaled':magnitude.squeeze(-1),
            'label_logits':logits,'confidence':confidence,
            'vectors_inward':direction*(magnitude/10)}


def supervised_loss(prediction,batch):
    target=batch['vectors_inward'];magnitude=target.norm(dim=-1);valid=magnitude>1e-6
    cosine=1-F.cosine_similarity(prediction['direction'],target,dim=-1)
    direction=cosine[valid].mean() if valid.any() else prediction['magnitude_scaled'].sum()*0
    losses={'direction':direction,
        'magnitude':F.mse_loss(prediction['magnitude_scaled'],magnitude*10),
        'labels':F.cross_entropy(prediction['label_logits'].transpose(1,2),batch['marker_labels']),
        'confidence':F.mse_loss(prediction['confidence'],batch['marker_confidence'])}
    return sum(losses.values()),losses


def aggregate_markers(inner_points,labels,confidence,count=86,top_k=3,alpha=20):
    """ETCH top-three confidence^20 voting, stable for very small confidence."""
    inner_points=np.asarray(inner_points);labels=np.asarray(labels);confidence=np.asarray(confidence).reshape(-1)
    if not np.isfinite(inner_points).all() or not np.isfinite(confidence).all():
        raise ValueError('Nonfinite marker input')
    if len(labels)!=len(inner_points) or len(confidence)!=len(labels):raise ValueError('Marker input lengths differ')
    result=np.zeros((count,3));valid=np.zeros(count,dtype=bool);support=np.zeros(count,dtype=int)
    for label in range(count):
        ids=np.flatnonzero((labels==label)&(confidence>0))
        if not len(ids):continue
        ids=ids[np.argsort(-confidence[ids],kind='stable')[:top_k]]
        log_weight=alpha*np.log(confidence[ids]);weight=np.exp(log_weight-log_weight.max())
        result[label]=np.sum(inner_points[ids]*weight[:,None],axis=0)/weight.sum()
        valid[label]=True;support[label]=len(ids)
    return result,valid,support
