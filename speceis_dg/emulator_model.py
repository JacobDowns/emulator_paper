import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc
import time
from speceis_dg.hybrid import UncoupledModel
from speceis_dg.grad_solver import GradSolver
from speceis_dg.data_mapper import DataMapper
from speceis_dg.new.simulator import Simulator
from speceis_dg.feature_constructor import FeatureConstructor
import torch 
from torch_geometric.data import Data

class EmulatorModel:

    def __init__(self, mesh):

        self.uncoupled_model = UncoupledModel(mesh)
        self.feature_constructor = FeatureConstructor(mesh)
        self.data_mapper = self.feature_constructor.data_mapper 

        self.coords = torch.tensor(self.data_mapper.coords, dtype=torch.float32)
        self.edge_index = torch.tensor(self.data_mapper.edges, dtype=torch.int64)

        # GNN model   
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        simulator = Simulator(message_passing_num=10, edge_input_size=13, device=device)
        simulator.load_checkpoint()
        simulator.eval()
        self.simulator = simulator

    
    def get_velocity(self, B, H, beta2):
        
         with torch.no_grad():

            X = self.feature_constructor.construct_features(B, H, beta2)        
            X = torch.tensor(X, dtype=torch.float32)        

            g = Data(
                pos = self.coords,
                edge_index = self.edge_index,
                x = X
            )

            g = g.cuda()

            # Estimate velocity
            out = self.simulator(g)
            # Return MTW coefficients
            out = out.cpu().flatten().numpy()

            return out

    
    def step(self, dt, solver = 'emulator'):
        
        if solver == 'emulator':
            B = self.uncoupled_model.B 
            H = self.uncoupled_model.H0
            beta2 = self.uncoupled_model.beta2 

            Ubar = self.get_velocity(B, H, beta2)

            self.uncoupled_model.W.sub(0).dat.data[:] = Ubar
            self.uncoupled_model.W_i.sub(0).dat.data[:] = Ubar
            self.uncoupled_model.Ubar0.dat.data[:] = Ubar
            self.uncoupled_model.solve_transport(dt)

        else:
            self.uncoupled_model.step(dt)
