import os
import sys
sys.path.append('./')
import numpy as np
import firedrake as fd
from speceis_dg.feature_constructor import FeatureConstructor
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
from velocity_loss import LossIntegral
from multiprocessing import Pool

class SimulationLoader:

    def __init__(self, name):
        
        self.name = name
        base_dir = f'training_runs/output/{name}/'
        self.h5_input = f'{base_dir}output_{name}.h5'
        self.sim_steps = 200
        self.skip = 1
        self.N = int(200 / self.skip)

        with fd.CheckpointFile(self.h5_input, 'r') as afile:
            self.mesh = afile.load_mesh()
            self.feature_constructor = FeatureConstructor(self.mesh, version = 'new')
            self.B = afile.load_function(self.mesh, 'B')
            self.loss_integral = LossIntegral(self.mesh)

            self.coords = self.feature_constructor.data_mapper.coords
            self.edges = self.feature_constructor.data_mapper.edges

            # Input features
            self.Xs = []
            # Output features
            self.Ys = []
        

    def get_vars(self, j):
        d = {}
        d['B'] = self.B

        with fd.CheckpointFile(self.h5_input, 'r') as afile:
            d['H'] = afile.load_function(self.mesh, 'H0', idx=j)
            d['beta2'] = afile.load_function(self.mesh, 'beta2', idx=j)
            d['Ubar'] = afile.load_function(self.mesh, 'Ubar0', idx=j)
            d['Udef'] = afile.load_function(self.mesh, 'Udef0', idx=j)

            return d
        
    
    def load_features_from_h5(self):



        # Use every other time step (5 years)
        for j in range(0, self.N, self.skip):
            print(self.name, j)
            vars = self.get_vars(j)

            X = self.feature_constructor.construct_features(vars['B'], vars['H'], vars['beta2'])

            # Outputs
            Ubar = vars['Ubar'].dat.data.reshape((-1,3))

            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Ubar, dtype=torch.float32)

            self.Xs.append(X)
            self.Ys.append(Y)


    def load_features_from_arrays(self):

        # Use every other time step (5 years)
        for j in range(0, self.N, self.skip):

            base_dir = f'training/graph_data/{self.name}'

            # Load outputs from arrays
            X = np.load(f'{base_dir}/X_{j}.npy')
            Y = np.load(f'{base_dir}/Y_{j}.npy')
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)

            self.Xs.append(X)
            self.Ys.append(Y)

    
    def save_feature_arrays(self):
        base_dir = f'training/graph_data/{self.name}'

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for j in range(0, self.N, self.skip):
            x = self.Xs[j]
            y = self.Ys[j]
            np.save(f'{base_dir}/X_{j}.npy', x)
            np.save(f'{base_dir}/Y_{j}.npy', y)


class SimulatorDataset(Dataset):
     
    def __init__(self):

        self.sim_loaders = sim_loaders = []
        # Number of timesteps per simulation
        self.num_steps = 100
        # Regions
        self.regions = [
            'beaverhead_500',
            'cabinet_500',
            'mission_500',
            'pintlers_500'
        ]
        # Number of regions
        self.num_simulations = len(self.regions)

        for i in range(self.num_simulations):
            region = self.regions[i]
            print(f'Loading Datastet: {region}')
            sim_loader = SimulationLoader(region)
            sim_loader.load_features_from_arrays()
            sim_loaders.append(sim_loader)

    def __len__(self):
        return self.num_simulations*self.num_steps
    
    def __getitem__(self, idx):
        sim_idx = int(idx / self.num_steps)
        time_idx = idx % self.num_steps
        #print(sim_idx, time_idx)

        sim_loader = self.sim_loaders[sim_idx]
        x = sim_loader.Xs[time_idx]
        y = sim_loader.Ys[time_idx]
        coords = torch.tensor(sim_loader.coords, dtype=torch.float32)
        edge_index = torch.tensor(sim_loader.edges, dtype=torch.int64)

        g = Data(
            pos = coords,
            edge_index = edge_index,
            x = x
        )

        return g, y, sim_loader