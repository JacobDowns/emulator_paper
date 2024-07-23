import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np

class GradSolver:
    def __init__(
            self,
            mesh,
    ):
        self.mesh = mesh
        nhat = fd.FacetNormal(mesh)

        V_rt = fd.FunctionSpace(mesh, 'MTW', 3)
        V_dg = fd.FunctionSpace(mesh, 'DG', 0)

        f = fd.Function(V_dg)
        dF = fd.TrialFunction(V_rt)
        f_grad = fd.Function(V_rt)
        Chi = fd.TestFunction(V_rt)
        self.f = f
        self.f_grad = f_grad

        R_grad = (fd.dot(Chi, dF)*fd.dx 
              + fd.div(Chi)*f*fd.dx 
              - fd.dot(Chi,nhat)*f*fd.ds)
        
        grad_problem = fd.LinearVariationalProblem(
            fd.lhs(R_grad),
            fd.rhs(R_grad),
            f_grad
        )
        
        projection_parameters = {'ksp_type':'cg','mat_type':'matfree'}

        self.grad_solver = fd.LinearVariationalSolver(
            grad_problem,
            solver_parameters=projection_parameters
        )

    def solve_grad(self, f):
        self.f.assign(f)
        self.grad_solver.solve()
        return self.f_grad

 
