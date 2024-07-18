import numpy as np 
import pandas as pd
import geopandas as gp
from dem_stitcher import stitch_dem
import rasterio
import glob
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# Download dems for different mountain ranges
def get_dems():
    mountains = gp.read_file('mountains.geojson')

    for index, row in mountains.iterrows():

        name = row['name']
        bounds = list(row.geometry.bounds)
        bounds[0] -= 0.25
        bounds[1] -= 0.25
        bounds[2] += 0.25
        bounds[3] += 0.25

        X, g = stitch_dem(bounds,
                        dem_name='glo_90', 
                        dst_ellipsoidal_height=False,
                        dst_area_or_point='Point')

        with rasterio.open(f'dems/{name}.tif', 'w', **g) as ds:
            ds.write(X, 1)
            ds.update_tags(AREA_OR_POINT='Point')

# Reproject dems
def reproject_dems(res = 100.):
    file_names = glob.glob('dems/*.tif')
    for file_name in file_names:
        X = rioxarray.open_rasterio(file_name)

        #plt.imshow(X[0])
        #plt.show()

        X = X.rio.reproject("EPSG:3857", resolution=res, resampling=Resampling.bilinear)
        X.rio.to_raster(file_name)

reproject_dems()