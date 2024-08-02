import os
import sys
import firedrake as fd
import numpy as np
from speceis_dg.data_mapper import DataMapper
from speceis_dg.grad_solver import GradSolver
import matplotlib.pyplot as plt


class FeatureConstructor:

    def __init__(self, mesh, version = 'new'):

        self.data_mapper = DataMapper(mesh)
        self.coords = self.data_mapper.coords
        self.edges = self.data_mapper.edges
        self.grad_solver = GradSolver(mesh)
        self.version = version 


        
    def construct_features(self, B, H, beta2):
        
        if self.version == 'new':
            S = B+H
            H_avg = self.data_mapper.get_avg(H)
            beta2_avg = self.data_mapper.get_avg(beta2)
            B_grad = self.grad_solver.solve_grad(B)
            S_grad = self.grad_solver.solve_grad(S)
            B_grad = B_grad.dat.data.reshape((-1,3))
            S_grad = S_grad.dat.data.reshape((-1,3))

            # Geometric variables
            X_g = np.column_stack([
                self.data_mapper.edge_lens,
                self.data_mapper.dx0,
                self.data_mapper.dy0,
                self.data_mapper.dx1,
                self.data_mapper.dy1
            ])

            # Input variables
            X_i = np.column_stack([
                H_avg / 100.,
                B_grad / 100.,
                S_grad / 100.,
                beta2_avg
            ])

            X_edge = np.column_stack([
                X_g,
                X_i
            ]) 

            return X_edge
        
        elif self.version == 'egcn':
            S = B+H
            H_avg = self.data_mapper.get_avg(H)
            B_jump = self.data_mapper.get_jump(B)
            S_jump = self.data_mapper.get_jump(S)
            
            # Input variables
            x_edge = np.column_stack([
                H_avg  / 100.,
                B_jump / 100., 
                S_jump / 100.
            ])

            x_node = np.column_stack([
                self.data_mapper.coords[:,0],
                self.data_mapper.coords[:,1],
                beta2.dat.data[:]
            ])

            x_edge = np.concatenate([x_edge, x_edge*[1.,-1.,-1.]])

            return x_node, x_edge
