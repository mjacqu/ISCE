import sys
import os
import numpy as np
sys.path.append('./')
import interferogram
import richdem
from osgeo import gdal, gdalconst
import matplotlib.pyplot as plt
#for viz
import rasterio
import earthpy
import earthpy.spatial as es
import earthpy.plot as ep

'''
Outline:
1. Create shadow and layover mask following Rees 2000
2. Apply compression-factor analysis following Cigna 2014 et al.,

Define geometric parameters:

β: Terrain slope relative to horizontal
α: Terrain aspect clockwise from NORTH
γ: Satellite heading angle clockwise from NORTH
θ: Angle between radar beam and vertical (look angle or incidence angle)
_: Angle between radar beam and terrain normal (LOCAL incidence angle)
_: Look direction clockwise from NORTH (normal to γ)


'''

# 1. Input: Geometric parameters
#look direction --> either from ifg or manual entry
#load radar data
ifg = interferogram.Interferogram(
    path ='/Volumes/Science/SpitzerStein/testdata/20151006_20151018'
    #path='/Users/mistral/Documents/ETHZ/Science/SpitzerStein/testdata/20151006_20151018'
)

look_direction = np.median(ifg.los[1])*-1
heading = look_direction + 90

#Question: does the slight variation of look_direction matter?

# 2. Input: DEM
#Spitzer Stein DEM (note: needs to be in meters for slope calculation to work)
path = '/Users/mistral/Documents/ETHZ/Science/SpitzerStein/kandersteg10m_crop.tif'
#path = '/Volumes/Science/LudovicArolla/SurfaceElevation_ArollaCrop.tif'
dem = richdem.LoadGDAL(path)
cell_size = dem.geotransform[1]

#3. reproject ifg.los into DEM grid (lat/long to local coordinate system)
t_srs = 'EPSG:2056'
dem_extent=[dem.geotransform[0],
    dem.geotransform[3]+dem.geotransform[5]*dem.shape[0],
    dem.geotransform[0]+dem.geotransform[1]*dem.shape[1],
    dem.geotransform[3]]


options = gdal.WarpOptions(
    xRes=dem.geotransform[1],
    yRes=dem.geotransform[1],
    outputBounds=dem_extent,
    errorThreshold=0.01,
    resampleAlg=gdalconst.GRA_Bilinear,
    dstSRS=t_srs,
    format='VRT'
)

los = gdal.Warp('',
    gdal.Open(os.path.join(ifg.path,'los.rdr.geo')),
    options=options
)

los_reprojected = los.ReadAsArray() #this are now the look angle and incidence angles reprojected to CH1903 LV95 (same as DEM)

plt.imshow(los_reprojected[0],vmin=40.5, vmax=42)
plt.show()
#4. Let the fun begin!
# Rotate the whole matrix to generate the new indices for each pixel
#https://en.wikipedia.org/wiki/Rotation_matrix
#x' = x*cos(theta)-y*sin(theta)
#y' = x*sin(theta)+y*cos(theta)
#where x,y = original coordinates, theta=rotation angle from horizontal x, x',y'=rotated coordinates

def ind2xy(r, c, array):
    'convert (row, colum) index into cartesian (x,y) coordinates: flip y-axis so that origin is bottom left corner'
    x = c
    y = array.shape[0]-(r+1)
    return x, y

def xy2ind(x, y, array):
    '''
    reverse operation from ind2xy: shift origin back to top right
    '''
    c = x
    r = array.shape[0]-y
    return r, c

rot = -np.radians(360-heading) # rotation of grid from due north
x, y = ind2xy(*np.indices(dem.shape), dem) #rows = y, columns = x
x_rot = x*np.cos(rot)-y*np.sin(rot) #distance from sensor plane
y_rot = x*np.sin(rot)+y*np.cos(rot) #distance from top edge of image

#empty array of same size
#flat indices of non-zero elements of line of sight mask
#distance sort flat indices
#filter distance sort flat indicies by visibility masks
y_rot_start = np.arange(y_rot.min(), y_rot.max(), 1)
visibility = np.full((dem.shape), np.nan)


def calc_projected_height(theta, dist, h):
    p_height = np.tan(np.radians(90-np.median(theta)))*dist + h
    return p_height

for i in y_rot_start:
    los_mask = (y_rot>=i) & (y_rot < i+1)
    true_los_ind = np.flatnonzero(los_mask) # r, c indicies of True elements in los_mask
    distances = x_rot[los_mask]
    d_sort = np.argsort(distances)
    d = distances[d_sort]*cell_size
    h_los = dem[los_mask][d_sort]
    p_height = calc_projected_height(los_reprojected[0], d, h_los)
    vis = p_height>=np.maximum.accumulate(p_height)
    visibility.flat[true_los_ind[d_sort]] = vis


#Hillshade for plotting results onto
# Open the DEM with Rasterio
with rasterio.open(path) as src:
    dtm = src.read(1)
    # Set masked values to np.nan
    dtm[dtm < 0] = np.nan


hillshade = es.hillshade(dtm)

oneline = np.full((dem.shape), 0)
onemask = (y_rot >=201) & (y_rot<202) #| (y_rot >=208) & (y_rot<209) | (y_rot >=212) & (y_rot<213)
oneind = np.flatnonzero(onemask)
oneline.flat[oneind] = np.ones(oneind.shape)

plt.ion()
plt.figure()
plt.imshow(hillshade, cmap='Greys_r')
plt.imshow(visibility, alpha = 0.6)
plt.imshow(oneline, alpha=0.2)

dist = x_rot[onemask]
d_sort = np.argsort(dist)
d = dist[d_sort]*cell_size
h_los = dem[onemask][d_sort]
p_height = calc_projected_height(los_reprojected[0], d, h_los)
vis = p_height>=np.maximum.accumulate(p_height)

plt.figure()
plt.plot(d, h_los, 'b')
plt.scatter(d, h_los, color='k')
plt.scatter(d[~vis], h_los[~vis], color='r')
plt.plot(d[1:-1], smooth, 'g')
plt.axis('equal')

#smooth line?
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

smooth=moving_average(h_los,3)
