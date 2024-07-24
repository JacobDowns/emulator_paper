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
from speceis_dg.gnn_model.simulator import Simulator
import torch 
from torch_geometric.data import Data

class EmulatorModel:

    def __init__(self, mesh):

        self.uncoupled_model = UncoupledModel(mesh)
        self.grad_solver = GradSolver(mesh)
        self.data_mapper = DataMapper(mesh)

        # Edge offsets and lengths
        self.coords = self.data_mapper.coords
        self.edges = self.data_mapper.edges

        # Geometric variables
        x_g = np.column_stack([
            self.data_mapper.edge_lens,
            self.data_mapper.dx0,
            self.data_mapper.dy0,
            self.data_mapper.dx1,
            self.data_mapper.dy1
        ])

        self.x_g = x_g

        # GNN model   
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        simulator = Simulator(message_passing_num=12, edge_input_size=13, device=device)
        simulator.load_checkpoint()
        self.simulator = simulator
        simulator.eval()

    
    def get_velocity(self, B, H, beta2):
        
         with torch.no_grad():

            # Create input tensor
            x_i = self.__get_inputs__(B, H, beta2)

            x = np.column_stack([
                self.x_g,
                x_i
            ])

            x = torch.tensor(x, dtype=torch.float32)

            coords = torch.tensor(self.coords, dtype=torch.float32)
            edge_index = torch.tensor(self.edges, dtype=torch.int64)

            g = Data(
                pos = coords,
                edge_index = edge_index,
                x = x
            )
            g = g.cuda()

            # Estimate velocity
            out = self.simulator(g)
            # Return MTW coefficients
            out = out.cpu().flatten().numpy()

            return out

        
    def __get_inputs__(self, B, H, beta2):
        """
        Prepare inputs for GNN model. 
        """

        H_avg = self.data_mapper.get_avg(H)
        beta2_avg = self.data_mapper.get_avg(beta2)
        B_grad = self.grad_solver.solve_grad(B)
        S_grad = self.grad_solver.solve_grad(B+H)
        B_grad = B_grad.dat.data.reshape((-1,3))
        S_grad = S_grad.dat.data.reshape((-1,3))
        
        # Input variables
        x_i = np.column_stack([
            H_avg,
            B_grad,
            S_grad,
            beta2_avg
        ])

        return x_i
    
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
