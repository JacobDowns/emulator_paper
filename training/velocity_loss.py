import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
import numpy as np
import torch
from petsc4py import PETSc

class LossIntegral:
    def __init__(self, mesh):
        self.V_mt = df.FunctionSpace(mesh, "MTW", 3)
        self.V_dg = df.FunctionSpace(mesh, 'DG', 0)
        
        self.Ubar_obs = Ubar_obs =  df.Function(self.V_mt)
        self.Ubar = Ubar = df.Function(self.V_mt)
        self.w_bar = df.TestFunction(self.V_mt)

        c0 = df.Constant(1.)

        rbar = Ubar - Ubar_obs
        self.I = c0*df.dot(rbar, rbar)*df.dx

        self.J_bar = df.derivative(self.I, self.Ubar, self.w_bar)
        self.j_bar = df.Function(self.V_mt)


class VelocityLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ubar, Ubar_obs, loss_integral):
        ctx.loss_integral = loss_integral 

        ctx.Ubar = Ubar    
        ctx.Ubar_obs = Ubar_obs

        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Ubar.dat.data[:] = Ubar

        return torch.tensor(df.assemble(loss_integral.I)) 

    @staticmethod
    def backward(ctx, grad_output):
        loss_integral = ctx.loss_integral  

        Ubar_obs = ctx.Ubar_obs
        Ubar = ctx.Ubar

        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Ubar.dat.data[:] = Ubar

        df.assemble(loss_integral.J_bar, tensor=loss_integral.j_bar) 
        j_bar = torch.tensor(loss_integral.j_bar.dat.data[:]) 

        return j_bar*grad_output, None, None, None