"""Observable-only graph/summary embeddings for stellar-stream inference.

Graph topology uses noisy measured coordinates with recorded scales/periods.
Native particle IDs, true memberships, generator labels and cosmology truth
are excluded by construction. Node padding and missingness remain explicit.
Torch is an optional runtime dependency; this module is imported only for SBI.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from core.stream_common import finite, integer, positive
from core.stream.inference import estimator_only


class ObservableEncoder:
    def __init__(self, record):
        self.record=record
        self.columns=tuple(record['columns'])
        if len(set(self.columns))!=len(self.columns) or not self.columns:
            raise ValueError('Distinct observed feature columns required')
        self.maximum_nodes=integer(record['maximum_nodes'],'maximum_nodes',2)
        self.scale=finite(record['coordinate_scale'],(len(self.columns),))
        self.center=finite(record['coordinate_center'],self.scale.shape)
        if np.any(self.scale<=0):
            raise ValueError('Positive frozen feature scales required')
        self.features=len(self.columns)+len(self.columns)**2+len(self.columns)+1
        self.periods={int(k):positive(v,'coordinate period') for k,v in record.get('periodic_columns',{}).items()}
        self.graph_columns=tuple(record['graph_coordinate_columns'])
        if not self.graph_columns or any(j<0 or j>=len(self.columns) for j in self.graph_columns):
            raise ValueError('Graph metric must use observed coordinate columns')

    @property
    def output_size(self):
        return self.maximum_nodes*self.features+(self.maximum_nodes*len(self.columns))**2

    def encode(self, observation):
        estimator_only(observation)
        if tuple(observation['columns'])!=self.columns:
            raise ValueError('Observed feature definitions changed')
        value=np.asarray(observation['values'],float)
        # Only explicit boolean masks may hide an observed coordinate.
        missing=np.asarray(observation['missing'])
        if value.ndim!=2 or value.shape[1]!=len(self.columns) or missing.shape!=value.shape or missing.dtype.kind!='b':
            raise ValueError('Observable dimensions/missingness disagree')
        n,d=value.shape
        if not n:
            raise ValueError('Empty observable selections cannot be used as a trained-posterior condition')
        if n>self.maximum_nodes:
            raise MemoryError('Graph node cap exceeded; select observations explicitly before inference')
        if np.any(~missing & ~np.isfinite(value)):
            raise ValueError('Unmasked nonfinite observable')
        covariance=finite(observation['covariance'],(n*d,n*d))
        if not np.allclose(covariance,covariance.T) or np.linalg.eigvalsh(covariance).min(initial=0)<-1e-10:
            raise ValueError('Observation covariance is not positive semidefinite')
        normalized=(np.where(missing,self.center,value)-self.center)/self.scale
        for j,period in self.periods.items():
            normalized[:,j]=((np.where(missing[:,j],self.center[j],value[:,j])-self.center[j]+period/2)%period-period/2)/self.scale[j]
        local=np.array([covariance[i*d:(i+1)*d,i*d:(i+1)*d]/np.outer(self.scale,self.scale) for i in range(n)])
        packed=np.zeros((self.maximum_nodes,self.features),np.float32)
        packed[:n]=np.column_stack((normalized,local.reshape(n,d*d),(~missing).astype(float),np.ones(n)))
        full=np.zeros((self.maximum_nodes*d,self.maximum_nodes*d),np.float32)
        scale=np.tile(self.scale,n)
        full[:n*d,:n*d]=covariance/np.outer(scale,scale)
        return np.concatenate((packed.ravel(),full.ravel()))


class SummaryEncoder(nn.Module):
    """Permutation-invariant measurable moment baseline with explicit masks."""
    def __init__(self,nodes,dimensions):
        super().__init__()
        # SBI queries next(embedding.parameters()) for its device. This
        # summary has no trainable weights; an empty frozen parameter exposes
        # its device without changing the statistics or optimizer degrees of
        # freedom. It follows .to(device) with the rest of the module.
        self.register_parameter('_device_anchor',nn.Parameter(torch.empty(0),requires_grad=False))
        self.nodes=nodes
        self.dimensions=dimensions
        self.features=dimensions+dimensions**2+dimensions+1
    def forward(self,packed):
        d=self.dimensions
        node_size=self.nodes*self.features
        q=packed[:,:node_size].reshape(-1,self.nodes,self.features)
        full=packed[:,node_size:].reshape(-1,self.nodes,d,self.nodes,d)
        value=q[:,:,:d]
        present=q[:,:,d+d*d:d+d*d+d]*q[:,:,-1:]
        count=present.sum(dim=1).clamp_min(1)
        mean=(value*present).sum(dim=1)/count
        centered=value-mean[:,None]
        variance=(centered.square()*present).sum(dim=1)/count
        skew=(centered.pow(3)*present).sum(dim=1)/count/variance.clamp_min(1e-12).pow(1.5)
        covariance=q[:,:,d:d+d*d].reshape(-1,self.nodes,d,d)
        error=(covariance.diagonal(dim1=-2,dim2=-1)*present).sum(dim=1)/count
        missing_fraction=1-present.sum(dim=1)/q[:,:,-1:].sum(dim=1).clamp_min(1)
        occupancy=q[:,:,-1].sum(dim=1,keepdim=True)/self.nodes
        pair_mask=present[:,:,:,None,None]*present[:,None,None,:,:]
        joint_error=(full*pair_mask).sum(dim=(1,3))/(count[:,:,None]*count[:,None,:]).clamp_min(1)
        return torch.cat((mean,variance,skew,error,missing_fraction,occupancy,joint_error.flatten(1)),dim=1)


class StreamGraphEncoder(nn.Module):
    """Genuine edge-conditioned message passing with deterministic kNN.

    No external graph library is required. The graph is rebuilt from observed
    coordinates in each batch; isolated nodes retain their measured features.
    The computational budget is O(batch*maximum_nodes^2).
    """
    def __init__(self,definition,*,hidden_features,layers,output_features,neighbors):
        super().__init__()
        self.definition=definition
        self.nodes=integer(definition['maximum_nodes'],'maximum_nodes',2)
        self.dimensions=len(definition['columns'])
        d=self.dimensions
        self.features=d+d*d+d+1
        self.coordinate_columns=tuple(definition['graph_coordinate_columns'])
        self.neighbors=integer(neighbors,'neighbors')
        self.layers=integer(layers,'layers')
        if self.neighbors>=self.nodes:
            raise ValueError('Neighbors must be smaller than node cap')
        h=integer(hidden_features,'hidden_features',4)
        out=integer(output_features,'output_features',2)
        self.input=nn.Sequential(nn.Linear(self.features,h),nn.SiLU(),nn.Linear(h,h))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*h+len(self.coordinate_columns)+1+d*d,h),nn.SiLU(),nn.Linear(h,h)) for _ in range(self.layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,h),nn.LayerNorm(h)) for _ in range(self.layers)])
        self.output=nn.Sequential(nn.Linear(2*h+1,h),nn.SiLU(),nn.Linear(h,out))
        self.periods={int(j):float(p)/definition['coordinate_scale'][int(j)] for j,p in definition.get('periodic_columns',{}).items()}

    def forward(self,packed):
        d=self.dimensions
        node_size=self.nodes*self.features
        q=packed[:,:node_size].reshape(-1,self.nodes,self.features)
        batch=len(q)
        joint=packed[:,node_size:].reshape(-1,self.nodes,d,self.nodes,d).permute(0,1,3,2,4)
        occupied=q[:,:,-1]>0
        present=q[:,:,d+d*d:d+d*d+d]>0
        eligible=occupied & present[:,:,self.coordinate_columns].all(dim=2)
        coordinate=q[:,:,self.coordinate_columns]
        delta=coordinate[:,None]-coordinate[:,:,None]
        for index,column in enumerate(self.coordinate_columns):
            if column in self.periods:
                period=self.periods[column]
                delta[...,index]=(delta[...,index]+period/2)%period-period/2
        distance=delta.square().sum(dim=-1)
        allowed=eligible[:,:,None]&eligible[:,None,:]&~torch.eye(self.nodes,dtype=torch.bool,device=q.device)[None]
        distance=distance.masked_fill(~allowed,float('inf'))
        chosen=distance.topk(self.neighbors,dim=2,largest=False,sorted=True).indices
        edge_mask=torch.gather(allowed,2,chosen)
        rows=torch.arange(batch,device=q.device)[:,None,None]
        source=torch.arange(self.nodes,device=q.device)[None,:,None]
        edge_delta=delta[rows,source,chosen]
        edge_covariance=joint[rows,source,chosen].flatten(-2)
        edge_distance=distance[rows,source,chosen].masked_fill(~edge_mask,0).sqrt()[...,None]
        state=self.input(q)*occupied[:,:,None]
        for message,update in zip(self.messages,self.updates):
            neighbor=state[rows,chosen]
            current=state[:,:,None].expand(-1,-1,self.neighbors,-1)
            edge=message(torch.cat((current,neighbor,edge_delta,edge_distance,edge_covariance),dim=-1))*edge_mask[:,:,:,None]
            aggregate=edge.sum(dim=2)/edge_mask.sum(dim=2).clamp_min(1)[:,:,None]
            state=(state+update(torch.cat((state,aggregate),dim=-1)))*occupied[:,:,None]
        count=occupied.sum(dim=1,keepdim=True).clamp_min(1)
        mean=state.sum(dim=1)/count
        maximum=state.masked_fill(~occupied[:,:,None],float('-inf')).amax(dim=1)
        maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
        return self.output(torch.cat((mean,maximum,count/self.nodes),dim=1))
