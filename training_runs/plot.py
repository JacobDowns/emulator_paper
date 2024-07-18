import os
import sys
#os.environ['OMP_NUM_THREADS'] = '1'
#sys.path.append('../../')
import firedrake as fd
#import pickle
#from firedrake.petsc import PETSc
#from speceis_dg.hybrid import CoupledModel
import numpy as np


with fd.CheckpointFile(f'training_data/inputs/input_mission_750.h5', 'r') as afile:
    mesh = afile.load_mesh()
    z = afile.load_function(mesh, 'z')
    z_smooth = afile.load_function(mesh, 'z_smooth')
    beta2 = afile.load_function(mesh, 'beta2')
    dist = afile.load_function(mesh, 'dist')
    print(dist.dat.data)

    #dist.dat.data[:] = -10.*np.exp(-dist.dat.data[:] / 50.)

    out = fd.File('training_runs/dist.pvd')
    out.write(dist)

    """
    B = afile.load_function(mesh, 'B')
    adot = afile.load_function(mesh, 'adot')
    beta2 = afile.load_function(mesh, 'beta2')
    c = mesh.coordinates.vector().array()
    """