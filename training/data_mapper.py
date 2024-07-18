import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from matplotlib.tri import Triangulation
import os
os.environ['OMP_NUM_THREADS'] = '1'

class DataMapper:

    def __init__(self, mesh):
        self.mesh = mesh
        V_cg = self.V_cg = fd.FunctionSpace(mesh, 'CG', 1)
        V_dg = self.V_dg = fd.FunctionSpace(mesh, 'DG', 0)
        V_cr = self.V_cr = fd.FunctionSpace(mesh, 'CR', 1)
        V_mt = self.V_mt = fd.FunctionSpace(mesh, 'MTW', 3)

        # Node coordinates
        self.xs = mesh.coordinates.dat.data[:,0]
        self.ys = mesh.coordinates.dat.data[:,1]
        self.coords = np.c_[self.xs, self.ys]

        # Faces
        self.faces = faces = V_cg.cell_node_list

        # Get edges
        e0 = faces[:,[0,1]]
        e1 = faces[:,[1,2]]
        e2 = faces[:,[2,0]]
        e0.sort(axis=1)
        e1.sort(axis=1)
        e2.sort(axis=1)
        edges = np.concatenate((e0, e1, e2)) 
        edges = np.unique(edges, axis=0)
        self.edges = edges

        # Edge midpoint coordinates
        x = fd.SpatialCoordinate(mesh)
        x_cr = fd.interpolate(x[0], V_cr)
        y_cr = fd.interpolate(x[1], V_cr)
        x_cr = x_cr.dat.data
        y_cr = y_cr.dat.data
        self.x_cr = x_cr
        self.y_cr = y_cr

        # Cell midpoint coordinates
        x_dg = fd.interpolate(x[0], V_dg)
        y_dg = fd.interpolate(x[1], V_dg)
        self.x_dg = x_dg = x_dg.dat.data[:]
        self.y_dg = y_dg = y_dg.dat.data[:]

        def get_edge_map():
            ex0 = 0.5*self.xs[edges].sum(axis=1)
            ey0 = 0.5*self.ys[edges].sum(axis=1)
            self.ei0 = np.lexsort((ex0, ey0)) 
            self.ei1 = np.lexsort((x_cr, y_cr))

        get_edge_map()
        edges1 = np.zeros_like(edges)
        edges1[self.ei1] = self.edges[self.ei0]
        self.edges = edges1

        # Edge lengths
        v_cr = fd.TestFunction(V_cr)
        self.edge_lens = fd.assemble(v_cr('+')*fd.dS + v_cr*fd.ds).dat.data
        self.v_cr = v_cr
        
        # Edge to cell maps
        v_dg = fd.Function(V_dg)
        v_dg.dat.data[:] = np.arange(len(v_dg.dat.data))
        c0 = fd.assemble((v_dg*v_cr)('+')*fd.dS + v_dg*v_cr*fd.ds).dat.data
        c1 = fd.assemble((v_dg*v_cr)('-')*fd.dS + v_dg*v_cr*fd.ds).dat.data
        c0 /= self.edge_lens
        c1 /= self.edge_lens
        self.c0 = np.array(np.rint(c0), dtype=int)
        self.c1 = np.array(np.rint(c1), dtype=int)

        # Vectors from edge midpoints to cell edges
        self.dx0 = self.x_dg[self.c0] - self.x_cr
        self.dy0 = self.y_dg[self.c0] - self.y_cr 
        self.dx1 = self.x_dg[self.c1] - self.x_cr
        self.dy1 = self.y_dg[self.c1] - self.y_cr  

    def get_avg(self, f):
        vals = fd.assemble(fd.avg(f*self.v_cr)*fd.dS + f*self.v_cr*fd.ds).dat.data
        vals /= self.edge_lens
        return vals
    
    def get_jump(self, f):
        x = f*self.v_cr
        vals = fd.assemble((x('+') - x('-'))*fd.dS + x*fd.ds).dat.data
        vals /= self.edge_lens
        return vals
