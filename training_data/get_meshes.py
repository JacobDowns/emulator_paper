import numpy as np 
import pandas as pd
import geopandas as gp
import rasterio
import glob
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pandamesh as pm
import meshio
import firedrake as fd 
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from rasterio.features import geometry_mask
from shapely.geometry import mapping
import xarray as xr
from scipy.ndimage import distance_transform_edt
import scipy
from scipy.ndimage import gaussian_filter 


# Create meshes and mesh masks
def get_meshes(resolutions=[500.,750.,1000.]):
    mountains = gp.read_file('mountains.geojson')
    mountains = mountains.to_crs('EPSG:3857')

    for index, row in mountains.iterrows():
        name = row['name']
        dem = rioxarray.open_rasterio(f'dems/{name}.tif')

        # Convert the polygon to image coordinates
        poly = row.geometry
        poly = poly.buffer(20000.)
        poly_geom = [mapping(poly)]
        mask = geometry_mask(poly_geom, transform=dem.rio.transform(), invert=True, out_shape=dem.data[0].shape)
        
        mesh_mask = xr.zeros_like(dem)
        mesh_mask.data[0] = mask
        mesh_mask.rio.to_raster(f'dems/{name}_mesh_mask.tif')

        for res in resolutions:
            gdf = gp.GeoDataFrame(geometry=[poly])
            gdf['cellsize'] = res
            mesher = pm.TriangleMesher(gdf)
            vertices, triangles = mesher.generate()

            vertices[:,0] -= dem.x.min().item()
            vertices[:,1] -= dem.y.min().item()
            vertices /= 1e3
                
            cells = [("triangle", np.array(triangles))]
            mesh = meshio.Mesh(vertices, cells)
            meshio.write(f"meshes/{name}_{int(res)}.msh", mesh, file_format='gmsh22')



def gen_field(nx, ny, correlation_scale):

    # Create the smoothing kernel
    x = np.arange(-correlation_scale, correlation_scale)
    y = np.arange(-correlation_scale, correlation_scale)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    filter_kernel = np.exp(-dist**2/(2*correlation_scale))

    # Generate random noise and smooth it
    noise = np.random.randn(nx, ny) 
    z = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        
    # Normalize so its in 0-1 range
    z -= z.min()
    z /= z.max()

    return z

def get_data(resolutions=[500.,750.,1000.]):
    mountains = gp.read_file('mountains.geojson')
    mountains = mountains.to_crs('EPSG:3857')

    for index, row in mountains.iterrows():
        name = row['name']
        print(name)
        dem = rioxarray.open_rasterio(f'dems/{name}.tif')
        mesh_mask = rioxarray.open_rasterio(f'dems/{name}_mesh_mask.tif')
        dist = distance_transform_edt(mesh_mask.data[0])
        beta2 = gen_field(dist.shape[0], dist.shape[1], 2000) 
        dem = dem.fillna(0.)
        z_smooth = gaussian_filter(dem.data[0], sigma = 70., mode='reflect')

        x = dem.x.data 
        y = dem.y.data 
        x -= x.min()
        y -= y.min()
        x /= 1e3
        y /= 1e3

        xx, yy = np.meshgrid(x, y)
        z_interp = NearestNDInterpolator(list(zip(xx.flatten(), yy.flatten())), dem.data[0][:,:].flatten())
        z_smooth_interp = NearestNDInterpolator(list(zip(xx.flatten(), yy.flatten())), z_smooth.flatten())
        dist_interp = NearestNDInterpolator(list(zip(xx.flatten(), yy.flatten())), dist[:,:].flatten())
        beta2_interp = NearestNDInterpolator(list(zip(xx.flatten(), yy.flatten())), beta2[:,:].flatten())
 
        for res in resolutions:
            mesh = fd.Mesh(f'meshes/{name}_{int(res)}.msh')
            mesh_coords = mesh.coordinates.dat.data[:]

            V = fd.FunctionSpace(mesh, 'CG', 1)
            z_f = fd.Function(V, name='z')
            z_smooth_f = fd.Function(V, name='z_smooth')
            dist_f = fd.Function(V, name='dist')
            beta2_f = fd.Function(V, name='beta2')

            z_mesh = z_interp(mesh_coords[:,0], mesh_coords[:,1])
            dist_mesh = dist_interp(mesh_coords[:,0], mesh_coords[:,1])
            beta2_mesh = beta2_interp(mesh_coords[:,0], mesh_coords[:,1])
            z_smooth_mesh = z_smooth_interp(mesh_coords[:,0], mesh_coords[:,1])

            z_f.dat.data[:] = z_mesh
            dist_f.dat.data[:] = dist_mesh
            beta2_f.dat.data[:] = beta2_mesh
            z_smooth_f.dat.data[:] = z_smooth_mesh

            """
            plt.subplot(4,1,2)
            plt.scatter(mesh_coords[:,0], mesh_coords[:,1], c=adot_mesh)
            plt.colorbar()

            plt.subplot(4,1,3)
            plt.scatter(mesh_coords[:,0], mesh_coords[:,1], c=edge_mesh)
            plt.colorbar()

            plt.subplot(4,1,4)
            plt.scatter(mesh_coords[:,0], mesh_coords[:,1], c=beta2_mesh)
            plt.colorbar()

            plt.show()
            """

            with fd.CheckpointFile(f'inputs/input_{name}_{int(res)}.h5', 'w') as afile:
                afile.save_mesh(mesh)
                afile.save_function(z_f)
                afile.save_function(dist_f)
                afile.save_function(beta2_f)
                afile.save_function(z_smooth_f)

#get_meshes()
get_data()