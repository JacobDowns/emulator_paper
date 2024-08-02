import os
import sys
sys.path.append('./')
import numpy as np
import firedrake as fd
#from speceis_dg.data_mapper import DataMapper
from speceis_dg.feature_constructor import FeatureConstructor
from speceis_dg.hybrid import UncoupledModel
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
from torch_scatter import scatter_add
from speceis_dg.egcn.model import EGCN 


name = 'beaverhead'
res = 1000
input_file = f'training_runs/output/{name}_{res}/output_{name}_{res}.h5'
with fd.CheckpointFile(input_file, 'r') as afile:
    mesh = afile.load_mesh()
    B = afile.load_function(mesh, 'B')

    feature_constructor = FeatureConstructor(mesh, version='egcn')
    data_mapper = feature_constructor.data_mapper 


    j = 10
    H = afile.load_function(mesh, 'H0', idx=j)
    beta2 = afile.load_function(mesh, 'beta2', idx=j)
    Ubar = afile.load_function(mesh, 'Ubar0', idx=j)
    Udef = afile.load_function(mesh, 'Udef0', idx=j)
    adot = afile.load_function(mesh, 'adot', idx=j)

    model = UncoupledModel(mesh)
    model.adot.assign(adot)
    model.beta2.assign(beta2)
    model.W.sub(0).dat.data[:] = Ubar.dat.data[:]
    model.W_i.sub(0).dat.data[:] = Ubar.dat.data[:]
    model.Ubar0.dat.data[:] = Ubar.dat.data[:]
    model.H0.assign(H)

    model.eval_transport_loss(1.)
    quit()


    x_node, x_edge = feature_constructor.construct_features(B, H, beta2)
    
    coords = feature_constructor.data_mapper.coords 
    edges = feature_constructor.data_mapper.edges

    edges = np.concatenate([
        edges, 
        edges[:,::-1]
    ])

    coords = torch.tensor(coords, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.int64)


    g = Data(
        pos = coords,
        edge_index = edge_index,
        edge_attr = torch.tensor(x_edge, dtype=torch.float32),
        node_attr = torch.tensor(x_node, dtype=torch.float32)
    )

    layer = EGCN(3, 3, 64)

    g1 = layer(g)

