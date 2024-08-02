import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc
import time

def full_quad(order):
    points,weights = np.polynomial.legendre.leggauss(order)
    points = (points+1)/2.
    weights /= 2.
    return points,weights

class VerticalBasis(object):
    def __init__(self,u,H,S_grad,B_grad,p=4,ssa=False):
        self.u = u
        if ssa:
            self.coef = [lambda s: 1.0]
            self.dcoef = [lambda s: 0.0]
        else:
            self.coef = [lambda s:1.0, lambda s:1./p*((p+1)*s**p - 1)]
            self.dcoef = [lambda s:0, lambda s:(p+1)*s**(p-1)]
        
        self.H = H
        self.S_grad = S_grad
        self.B_grad = B_grad

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dz(self,s):
        return self.ds(s)*self.dsdz(s)

    def dx_(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

    def dx(self,s,x):
        return self.dx_(s,x) + self.ds(s)*self.dsdx(s,x)

    def dsdx(self,s,x):
        return 1./self.H*(self.S_grad[x] - s*(self.S_grad[x] - self.B_grad[x]))

    def dsdz(self,x):
        return -1./self.H

class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights

    def integral_term(self,f,s,w):
        return w*f(s)

    def intz(self,f):
        return sum([self.integral_term(f,s,w) 
                    for s,w in zip(self.points,self.weights)])  
    

def get_stress_form(model):

    B = model.B 
    H = model.H
    H_i = model.H_i 
    beta2 = model.beta2
    B_grad = model.B_grad 
    S_grad = model.S_grad 
    Ubar = model.Ubar 
    Udef = model.Udef 
    Ubar_i = model.Ubar_i 
    Udef_i = model.Udef_i
    Phibar = model.Phibar 
    Phidef = model.Phidef 
    nhat = model.nhat
    gamma = model.gamma
    omega = model.omega
    alpha = model.alpha
    delta = model.delta
    eps_reg = model.eps_reg
    n = model.n

    ubar,vbar = Ubar
    udef,vdef = Udef

    ubar_i,vbar_i = Ubar_i
    udef_i,vdef_i = Udef_i

    phibar_x,phibar_y = Phibar
    phidef_x,phidef_y = Phidef
    
    u = VerticalBasis([ubar, udef], H_i, S_grad, B_grad)
    v = VerticalBasis([vbar,vdef], H_i, S_grad, B_grad)
    u_i = VerticalBasis([ubar_i,udef_i], H_i, S_grad, B_grad)
    v_i = VerticalBasis([vbar_i,vdef_i],H_i, S_grad, B_grad)
    phi_x = VerticalBasis([phibar_x,phidef_x], H_i, S_grad, B_grad)
    phi_y = VerticalBasis([phibar_y,phidef_y], H_i, S_grad, B_grad)

    U_b = fd.as_vector([u(1),v(1)])
    Phi_b = fd.as_vector([phi_x(1),phi_y(1)])

    vi_x = VerticalIntegrator(*full_quad(2))
    vi_z = VerticalIntegrator(*full_quad(3)) 

    def eps_i_II(s):
        return (delta**2*(u_i.dx(s,0))**2 
                    + delta**2*(v_i.dx(s,1))**2 
                    + delta**2*(u_i.dx(s,0))*(v_i.dx(s,1)) 
                    + delta**2*0.25*((u_i.dx(s,1)) + (v_i.dx(s,0)))**2 
                    +0.25*(u_i.dz(s))**2 + 0.25*(v_i.dz(s))**2 
                    + eps_reg)

    def eta(s):
        return 0.5*eps_i_II(s)**((1-n)/(2*n))

    def phi_grad_membrane(s):
        return np.array([[delta*phi_x.dx(s,0), delta*phi_x.dx(s,1)],
                            [delta*phi_y.dx(s,0), delta*phi_y.dx(s,1)]])

    def phi_grad_shear(s):
        return np.array([[phi_x.dz(s)],
                            [phi_y.dz(s)]])

    def eps_membrane(s):
        return np.array([[2*delta*u.dx(s,0) + delta*v.dx(s,1), 
                            0.5*delta*u.dx(s,1) + 0.5*delta*v.dx(s,0)],
                            [0.5*delta*u.dx(s,1) + 0.5*delta*v.dx(s,0),
                            delta*u.dx(s,0) + 2*delta*v.dx(s,1)]])

    def eps_shear(s):
        return np.array([[0.5*u.dz(s)],
                        [0.5*v.dz(s)]])

    def membrane_form(s):
        return (2*eta(s)*(eps_membrane(s)
                * phi_grad_membrane(s)).sum()*H_i*fd.dx(degree=9))

    def shear_form(s):
        return (2*eta(s)*(eps_shear(s)
                * phi_grad_shear(s)).sum()*H_i*fd.dx(degree=9))

    def membrane_boundary_form_nopen(s):
        un = u(s)*nhat[0] + v(s)*nhat[1]
        return alpha*(phi_x(s)*un*nhat[0] + phi_y(s)*un*nhat[1])*fd.ds
   
    membrane_stress = -(vi_x.intz(membrane_form) 
                        + vi_z.intz(shear_form) 
                        + vi_x.intz(membrane_boundary_form_nopen))
    
    basal_stress = -gamma*beta2*fd.dot(U_b,Phi_b)*fd.dx
   
    driving_stress = (-omega*fd.div(Phibar*H)*B*fd.dx 
                    - omega*fd.div(Phibar*H_i)*H*fd.dx 
                    + omega*fd.jump(Phibar*H,nhat)*fd.avg(B)*fd.dS 
                    + omega*fd.jump(Phibar*H_i,nhat)*fd.avg(H)*fd.dS 
                    + omega*fd.dot(Phibar*H,nhat)*B*fd.ds 
                    + omega*fd.dot(Phibar*H_i,nhat)*H*fd.ds
                    )
    R_stress = membrane_stress + basal_stress - driving_stress
    
    return R_stress


def get_transport_form(model):
        
        H0 = model.H0
        H = model.H 
        H_i = model.H_i
        nhat = model.nhat
        Ubar = model.Ubar
        Ubar_i = model.Ubar_i
        xsi = model.xsi 
        dt = model.dt
        adot = model.adot
        flux_type= model.flux_type
        zeta = model.zeta

        H_avg = 0.5*(H_i('+') + H_i('-'))
        H_jump = H('+')*nhat('+') + H('-')*nhat('-')
        xsi_jump = xsi('+')*nhat('+') + xsi('-')*nhat('-')

        unorm_i = fd.dot(Ubar_i, Ubar_i)**0.5

        if flux_type=='centered':
            uH = fd.avg(Ubar)*H_avg

        elif flux_type=='lax-friedrichs':
            uH = fd.avg(Ubar)*H_avg + fd.Constant(0.5)*fd.avg(unorm_i)*H_jump

        elif flux_type=='upwind':
            uH = fd.avg(Ubar)*H_avg + 0.5*abs(fd.dot(fd.avg(Ubar_i),nhat('+')))*H_jump

        else:
            print('Invalid flux')
        
        R_transport = ((H - H0)/dt - adot)*xsi*fd.dx + zeta*fd.dot(uH,xsi_jump)*fd.dS

        return R_transport
      

class BaseModel:
    def __init__(
            self, 
            mesh,
            solver_type='direct',
            vel_scale=1.,
            thk_scale=1.,
            len_scale=1e3, 
            beta_scale=1e4,
            time_scale=1, 
            g=9.81, 
            rho_i=917., 
            rho_w=1000.0,
            n=3.0, 
            A=1e-16, 
            eps_reg=1e-6,
            thklim=2., 
            theta=1.0, 
            alpha=0,
            flux_type='lax-friedrichs',
        ):
            
        self.mesh = mesh
        nhat = fd.FacetNormal(mesh)
        self.nhat = nhat 

        E_cg = fd.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.E_cg = E_cg 

        E_dg = fd.FiniteElement('DG', mesh.ufl_cell(), 0)
        self.E_dg = E_dg 

        E_mtw = fd.FiniteElement('MTW', mesh.ufl_cell(), 3)
        self.E_mtw = E_mtw

        E_rt = fd.FiniteElement('RT', mesh.ufl_cell(), 1)
        self.E_rt = E_rt

        Q_cg = fd.FunctionSpace(mesh, E_cg)
        self.Q_cg = Q_cg 

        Q_mtw = fd.FunctionSpace(mesh, E_mtw)
        self.Q_mtw = Q_mtw 

        Q_dg = fd.FunctionSpace(mesh, E_dg)
        self.Q_dg = Q_dg 

        Q_rt = fd.FunctionSpace(mesh, E_rt)
        self.Q_rt = Q_rt 
      
        self.one = fd.Function(Q_dg)
        self.one.assign(1.0)
        self.area = fd.assemble(self.one * fd.dx)

        self.t = fd.Constant(0.)
        dt = fd.Constant(1.0)
        self.dt = dt 

        self.theta = fd.Constant(theta)
        self.g = fd.Constant(g)
        self.rho_i = fd.Constant(rho_i)
        self.rho_w = fd.Constant(rho_w)
        self.n = fd.Constant(n)
        self.eps_reg = fd.Constant(eps_reg)
        self.thklim = fd.Constant(thklim)

        vel_scale = fd.Constant(vel_scale)
        self.vel_scale = vel_scale
        self.thk_scale = fd.Constant(thk_scale)
        len_scale = fd.Constant(len_scale)
        beta_scale = fd.Constant(beta_scale)
        time_scale = fd.Constant(time_scale) 

        eta_star = fd.Constant(A**(-1./n)
            * (vel_scale / thk_scale)**((1-n) / n))
        self.eta_star = eta_star

        delta = fd.Constant(thk_scale / len_scale)
        self.delta = delta

        self.gamma = fd.Constant(beta_scale*thk_scale/eta_star)

        self.omega = fd.Constant(rho_i*g*thk_scale**3
                                         / (eta_star*len_scale*vel_scale))

        self.zeta = fd.Constant(time_scale*vel_scale/len_scale)

        self.S_grad = fd.Function(Q_rt)
        self.B_grad = fd.Function(Q_rt)
        self.Chi = fd.TestFunction(Q_rt)
        self.dS = fd.TrialFunction(Q_rt)

        self.Ubar0 = fd.Function(Q_mtw, name='Ubar0')
        self.Udef0 = fd.Function(Q_mtw, name='Udef0')
        self.H0 = H = fd.Function(Q_dg, name='H0')
        self.B = fd.Function(Q_dg, name='B')
        self.adot = fd.Function(Q_dg, name='adot') 
        self.beta2 = fd.Function(Q_cg, name='beta2')
        self.alpha = fd.Constant(alpha)
        self.H_temp = fd.Function(self.Q_dg)

        self.flux_type = flux_type
        self.solver_type = solver_type

        if solver_type=='direct':
            self.solver_params = {"ksp_type": "preonly",
                                  "pmat_type":"aij",
                                  "pc_type": "lu",  
                                  "pc_factor_mat_solver_type": "mumps"} 
        else:
            self.solver_params = {'pc_type': 'bjacobi',
                                  "ksp_rtol":1e-5}
            
        self.projection_params = {'ksp_type':'cg','mat_type':'matfree'}


    def init_vars(self):
        theta = self.theta
        self.Hmid = theta*self.H + (1-theta)*self.H0
        self.Hmid_i = theta*self.H_i + (1-theta)*self.H0

        self.S = self.B + self.H
        S0 = self.B + self.H0
        self.Smid = theta*self.S + (1-theta)*S0

        R_B = (fd.dot(self.Chi, self.dS)*fd.dx 
              + fd.div(self.Chi)*(self.B)*fd.dx 
              - fd.dot(self.Chi, self.nhat)*(self.B)*fd.ds)

        R_S = (fd.dot(self.Chi, self.dS)*fd.dx 
              + fd.div(self.Chi)*(self.Smid)*fd.dx 
              - fd.dot(self.Chi, self.nhat)*(self.Smid)*fd.ds)
        
        S_grad_problem = fd.LinearVariationalProblem(fd.lhs(R_S), fd.rhs(R_S), self.S_grad)
        self.S_grad_solver = fd.LinearVariationalSolver(
            S_grad_problem,
            solver_parameters=self.projection_params
        )

        B_grad_problem = fd.LinearVariationalProblem(fd.lhs(R_B), fd.rhs(R_B), self.B_grad)
        self.B_grad_solver = fd.LinearVariationalSolver(
            B_grad_problem,
            solver_parameters=self.projection_params
        )

         

        
class CoupledModel(BaseModel):

    def __init__(self, mesh, **kwargs):

    
        super().__init__(mesh, **kwargs)

        E = fd.MixedElement(self.E_mtw, self.E_mtw, self.E_dg)
        self.E = E 

        V = fd.FunctionSpace(self.mesh, E)
        self.V = V 

        W = self.W = fd.Function(self.V)
        W_i = self.W_i = fd.Function(self.V)
        Psi = fd.TestFunction(self.V)
        dW = fd.TrialFunction(self.V)

        self.Ubar, self.Udef, self.H = fd.split(W)
        self.Ubar_i, self.Udef_i, self.H_i = fd.split(W_i)
        self.Phibar, self.Phidef, self.xsi = fd.split(Psi)

        self.init_vars()

      
        self.R_stress = get_stress_form(self)
        self.R_transport = get_transport_form(self)

        R = self.R_stress + self.R_transport
        R_lin = self.R_lin = fd.replace(R, {W:dW})

        coupled_problem = fd.LinearVariationalProblem(fd.lhs(R_lin), fd.rhs(R_lin),W)

        self.coupled_solver = fd.LinearVariationalSolver(
            coupled_problem,
            solver_parameters=self.solver_params)

    
    def step(
            self,
            dt,
            picard_tol=1e-6,
            max_iter=50,
            momentum=0.0,
            error_on_nonconvergence=False,
            convergence_norm='linf',
        ):

        self.W.sub(0).assign(self.Ubar0)
        self.W.sub(1).assign(self.Udef0)
        self.W.sub(2).assign(self.H0)

        self.W_i.assign(self.W)
        self.dt.assign(dt)

        eps = 1.0
        i = 0
        
        while eps>picard_tol and i<max_iter:
            t_ = time.time()

            self.S_grad_solver.solve()
            self.B_grad_solver.solve()
            self.coupled_solver.solve()
            self.H_temp.interpolate(fd.max_value(self.W.sub(2),self.thklim))
            self.W.sub(2).assign(self.H_temp)
            
            if convergence_norm=='linf':
                with self.W_i.dat.vec_ro as w_i:
                    with self.W.dat.vec_ro as w:
                        eps = abs(w_i - w).max()[1]
            else:
                eps = (np.sqrt(
                       fd.assemble((self.W_i.sub(2) - self.W.sub(2))**2*fd.dx))
                       / self.area)

            PETSc.Sys.Print(i,eps,time.time()-t_)


            self.W_i.assign((1-momentum)*self.W + momentum*self.W_i)
            i+=1

        if i==max_iter and eps>picard_tol:
            converged=False
        else:
            converged=True


        if error_on_nonconvergence and not converged:
            return converged

        self.Ubar0.assign(self.W.sub(0))
        self.Udef0.assign(self.W.sub(1))
        self.H0.assign(self.W.sub(2))

        return converged


class UncoupledModel(BaseModel):
    
    def __init__(self, mesh, **kwargs):

        super().__init__(mesh, **kwargs)

        E = fd.MixedElement(self.E_mtw, self.E_mtw)
        self.E = E 

        V = fd.FunctionSpace(self.mesh, E)
        self.V = V 

        W = self.W = fd.Function(self.V)
        W_i = self.W_i = fd.Function(self.V)
        Psi = fd.TestFunction(self.V)
        dW = fd.TrialFunction(self.V)

        self.Ubar, self.Udef = fd.split(W)
        self.Ubar_i, self.Udef_i = fd.split(W_i)
        self.Phibar, self.Phidef = fd.split(Psi)

        self.H = fd.Function(self.Q_dg)
        self.H.interpolate(fd.max_value(self.H,  self.thklim))
        self.H_i = self.H 

        dH = fd.TrialFunction(self.Q_dg)
        self.xsi = fd.TestFunction(self.Q_dg)

        self.init_vars()

        self.R_stress = get_stress_form(self)
        self.R_transport = get_transport_form(self)
    
        R_stress_lin = self.R_stress_lin = fd.replace(self.R_stress, {W:dW})
        velocity_problem = fd.LinearVariationalProblem(fd.lhs(R_stress_lin), fd.rhs(R_stress_lin),W)

        R_transport_lin = self.R_transport_lin = fd.replace(self.R_transport, {self.H:dH, self.H_i:dH})
        transport_problem = fd.LinearVariationalProblem(fd.lhs(R_transport_lin), fd.rhs(R_transport_lin), self.H)

        self.velocity_solver = fd.LinearVariationalSolver(
            velocity_problem,
            solver_parameters=self.solver_params
        )

        self.transport_solver = fd.LinearVariationalSolver(
            transport_problem,
            solver_parameters=self.solver_params
        )

        ### Loss functions

        self.H_obs = fd.Function(self.Q_dg)
        self.Ubar_obs = fd.Function(self.Q_mtw)
        self.Lambda = fd.Function(self.Q_dg)
        self.Delta = fd.Function(self.Q_dg)

        self.transport_loss = (self.H - self.H_obs)**2*fd.dx
        self.velociy_loss = fd.dot(self.Ubar_obs - self.Ubar, self.Ubar_obs - self.Ubar)*fd.dx 
        self.dUbar = fd.TestFunction(self.Q_mtw)

    def solve_velocity(
            self,
            picard_tol=1e-3,
            max_iter=50,
            momentum=0.0,
            error_on_nonconvergence=False,
            convergence_norm='linf'
        ):

        self.W.sub(0).assign(self.Ubar0)
        self.W.sub(1).assign(self.Udef0)
        self.W_i.assign(self.W)

        eps = 1.0
        i = 0
    
        while eps>picard_tol and i<max_iter:
            t_ = time.time()

            self.S_grad_solver.solve()
            self.B_grad_solver.solve()
            self.velocity_solver.solve()
    
            if convergence_norm=='linf':
                with self.W_i.dat.vec_ro as w_i:
                    with self.W.dat.vec_ro as w:
                        eps = abs(w_i - w).max()[1]
            else:
                eps = (np.sqrt(
                       fd.assemble((self.W_i.sub(1) - self.W.sub(1))**2*fd.dx))
                       / self.area)

            PETSc.Sys.Print(i,eps,time.time()-t_)

            self.W_i.assign((1-momentum)*self.W + momentum*self.W_i)
            i+=1

        if i==max_iter and eps>picard_tol:
            converged=False
        else:
            converged=True

        if error_on_nonconvergence and not converged:
            return converged

        self.Ubar0.assign(self.W.sub(0))
        self.Udef0.assign(self.W.sub(1))



        return converged
    
    def solve_transport(self, dt):
        self.dt.assign(dt)
        self.transport_solver.solve()
        self.H.interpolate(fd.max_value(self.H,  self.thklim))
        self.H0.assign(self.H)

    def step(self, dt, **kwargs):
        converged = self.solve_velocity(**kwargs)
        self.solve_transport(dt)
        return converged
    

    def eval_transport_loss(self, dt):
        
        # Solve for h given Ubar
        self.dt.assign(dt)
        self.transport_solver.solve()

        # Solve the adjoint equation
        A = fd.derivative(self.R_transport, self.H)
        A = fd.assemble(A)
        fd.solve(A, self.Lambda, fd.assemble(fd.derivative(-self.transport_loss, self.H)))

        g = fd.derivative(self.R_transport, self.Ubar)
        quit()
        # Compute derivative of loss w.r.t. Ubar
        R_u = fd.assemble(fd.derivative(self.R_transport, self.Ubar, self.dUbar))
        
        quit()
        #self.product1(R_u, self.Lambda, self.du)

        # Eval loss 
        R = fd.assemble(self.transport_loss)    

        return R, self.du