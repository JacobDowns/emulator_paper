import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('./')
import firedrake as df
from firedrake.petsc import PETSc
from speceis_dg.hybrid_new import CoupledModel, UncoupledModel
import numpy as np
from scipy.special import expit


class Run:
    def __init__(self, name, res=500):

        results_dir = f'training_runs/uncoupled_test/{name}_{res}'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
      
        with df.CheckpointFile(f'training_data/inputs/input_{name}_{res}.h5', 'r') as afile:
            mesh = afile.load_mesh()
            B = afile.load_function(mesh, 'z')
            B_smooth = afile.load_function(mesh, 'z_smooth')
            dist = afile.load_function(mesh, 'dist')
            beta2 = afile.load_function(mesh, 'beta2')

        vel_scale = 1.
        thk_scale = 1.
        len_scale = 1e3
        beta_scale = 1e4
        
        config = {
            'solver_type': 'direct',
            'vel_scale': vel_scale,
            'thk_scale': thk_scale,
            'len_scale': len_scale,
            'beta_scale': beta_scale,
            'theta': 1.0,
            'thklim': 2.,
            'alpha': 1000.0
        }
          
        beta2.interpolate(beta2 + df.Constant(0.1))
        model = self.model = UncoupledModel(mesh,**config)
        model.beta2.interpolate(beta2)
        model.B.interpolate(B)
        model.H0.interpolate(df.Constant(1. / thk_scale))        
        
        adot_vals = -2.*(expit((B_smooth.dat.data - B.dat.data) / 100.) - 0.5)
        adot = df.Function(model.Q_cg)
        adot.dat.data[:] =  -25.*(1. - expit(dist.dat.data / 125.)) + 7.5*adot_vals
        model.adot.assign(df.project(adot, model.Q_dg))
        dist = df.project(dist, model.Q_dg)

        

        S_file = df.File(f'{results_dir}/S.pvd')
        H_file = df.File(f'{results_dir}/H.pvd')
        Ubar_file = df.File(f'{results_dir}/Ubar.pvd')
        Udef_file = df.File(f'{results_dir}/Udef.pvd')

        S_out = df.Function(model.Q_dg,name='S')

        t = 0.
        t_end = 50
        dt = 0.5

        j = 0
        while t < t_end:

            beta_scale = 1. + (1./4.)*np.cos(t*2.*np.pi / 100.)
            adot0 = 2.*np.sin(t*2.*np.pi / 1000. ) 
            model.beta2.interpolate(beta2*df.Constant(beta_scale))
            eps = 0.5*np.sqrt(model.H0.dat.data)*np.random.randn(len(model.adot.dat.data))
            eps[dist.dat.data < 40.] = 0.
            model.adot.interpolate(adot + df.Constant(adot0))
            model.adot.dat.data[:] += eps
            
            converged = model.step(
                dt,
                picard_tol=1e-3,
                momentum=0.5,
                max_iter=25,
                convergence_norm='l2')

            if not converged:
                dt*=0.5
                continue
            
            t += dt

            PETSc.Sys.Print('step', t,dt,df.assemble(model.H0*df.dx))
            
            n_out = 1
            if j % n_out == 0:
                out_idx = int(j/n_out)

                S_out.assign(df.project(model.S, model.Q_dg))

                S_file.write(S_out, time=t)
                H_file.write(model.H0,time=t)
                Ubar_file.write(model.Ubar0, time=t)
                Udef_file.write(model.Udef0, time=t)
            
                
                j += 1
        

for res in [1000]:
    run = Run('beaverhead', res)
#if __name__=='__main__':
#    i = int(sys.argv[1])
#    print(i)
#    bc = Run(i)
