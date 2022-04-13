import sys
sys.path.append('../MudCreek')
import los_projection as lp
import numpy as np
sys.path.append('./')
import interferogram
import shadow_functions
import insarhelpers
from osgeo import gdal
import scipy.signal
import richdem
#from mpmath import *
import rasterio
import earthpy
import earthpy.spatial as es
import earthpy.plot as ep
import matplotlib.pyplot as plt
from matplotlib import colors


'''
Outline:
1. Create shadow mask using projected height approach from Pairman and McNeil
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

#reproject los parameters to the extent of the dem (not used right now because of artefacts in the los data)
reprojected = insarhelpers.reproject_to_target_raster(
    source_path = '/Volumes/Science/SpitzerStein/testdata/20151006_20151018/merged/los.rdr.geo',
    target = dem,
    target_srs = 'EPSG:2056')
los_reprojected = reprojected.ReadAsArray() #this are now the look angle and incidence angles reprojected to CH1903 LV95 (same as DEM)

# run the functions
prime_array, x_prime, y_prime = shadow_functions.rotate_array(dem, angle=heading)
h_p = shadow_functions.calc_projected_height(np.median(ifg.los[0]), prime_array, cell_size)
rc = shadow_functions.calc_layover_distance(np.median(ifg.los[0]), prime_array, cell_size)
vis = h_p >= np.fmax.accumulate(h_p, axis=1)
vis_reg = shadow_functions.rotate_array_inverse(vis, x_prime, y_prime)
#layover 1 (search for maximum from left to right)
layover1 = rc >= np.fmax.accumulate(rc, axis=1)
lay1_reg = shadow_functions.rotate_array_inverse(layover1, x_prime, y_prime)
#layover 2 (search for minimum from right to left)
layover2 = rc >= np.fmin.accumulate(np.flip(rc), axis=1)
lay2_reg = shadow_functions.rotate_array_inverse(layover2, x_prime, y_prime)

#plot results
# plt.figure()
# plt.imshow(dem)
#
# plt.figure()
# plt.imshow(prime_array)
#
# plt.figure()
# plt.imshow(vis)
#plt.figure()
plt.imshow(hillshade, cmap='Greys_r')
plt.imshow(lay2_reg, alpha = 0.7)
plt.show()

plt.plot(rc[100,:])
plt.imshow(rc[rc==10])
# old stuff below

# calculate slope and aspect of DEM
slope_deg = richdem.TerrainAttribute(dem, attrib='slope_degrees')
slope_dx = richdem.TerrainAttribute(dem, attrib='slope_riserun')
aspect = richdem.TerrainAttribute(dem, attrib='aspect')

look_angle = np.radians(np.median(ifg.los[0]))


#Thresholds:
shadow_threshold = 1/np.tan(look_angle) #all areas where slope <= threshold (Rees 2000)
layover_threshold = np.tan(look_angle) #all areas where slope > threshold (Rees 2000)

#Rees 2000 EQ. 1: highlighting factor q "Number of scatterers contributing to a unit length in the image"
q = (np.sqrt(1+slope_dx**2))/(1-slope_dx*(shadow_threshold))

plt.imshow(q, vmin=-10, vmax=10, cmap='bwr')
plt.colorbar()
plt.show()

#Normal lighting = 1/2 < q < 2
normal_illumination = np.ma.masked_outside(q, 0.5, 2)

#Reverse imaging (layover (q<0)):
layover = np.ma.masked_where(q>0, q) #show everything where q<0

#shadow (q < sin(look_angle))
shadow = np.ma.masked_where(q > np.sin(look_angle), q) #show everything where q<sin(theta)


#Pairman + McNeil: shadow occurs where aspect is away from sensor and slope is more than pi/2-incidence_angel
pairman_threshold = np.pi/2-look_angle


#Hillshade for plotting results onto
# Open the DEM with Rasterio
with rasterio.open(path) as src:
    dtm = src.read(1)
    # Set masked values to np.nan
    dtm[dtm < 0] = np.nan

hillshade = es.hillshade(dtm)

# Plot the data

cmap_blue = colors.ListedColormap(['blue'])
cmap_black = colors.ListedColormap(['black'])
cmap_red = colors.ListedColormap(['red'])

f, ax = plt.subplots()
ax.imshow(hillshade, cmap='Greys_r')
#ax.imshow(q, vmin=-10, vmax=10, alpha=0.5)
ax.imshow(normal_illumination, cmap=cmap_blue, alpha=0.5)
ax.imshow(shadow, cmap=cmap_black, alpha=0.5)
ax.imshow(layover,cmap=cmap_red, alpha=0.5)
ax.set_title('normal=blue | shadow=black | layover=red')
f.show()


#From Magali (SAR Geocoding data and systems: Chapter 4)
#Foreshortening: surface slope smaller than incidence angle
#Layover: surface slope larger or equal to incidence angle

#Aspects at which shadowing/layover is possile
#Assumption Nr. 1: Shadowing happens in area +/- 90 degrees from look direction (face away from radar)
#Assumption Nr. 2: Layover + foreshortening happens in area +/- 90 degress opposite from look direction (facing the radar)
look_direction = np.median(ifg.los[1])*-1

def get_sl_interval(look_direction):
    s_interval = [look_direction+90, look_direction-90]
    l_interval = [look_direction-90, look_direction+90]
    return s_interval, l_interval


s_interval, l_interval = get_sl_interval(look_direction)

#Plot areas with aspect categorized by whether or not they face the radar:
facing_away = np.ma.masked_inside(aspect, s_interval[0], s_interval[1])
facing_toward = np.ma.masked_outside(aspect, l_interval[0], l_interval[1])

layover = np.ma.masked_less(slope_deg, ifg.los[0].mean())

shadow =np.ma.masked_less(np.tan(np.radians(slope_deg)), pairman_threshold) #pairman_threshold yields larger shadowed area. More realistic?

# DEM areas that are facing radar
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(hillshade, cmap='Greys_r')
ax1.imshow(facing_toward, cmap=cmap_blue)
ax2.imshow(hillshade, cmap='Greys_r')
ax2.imshow(np.ma.masked_array(dem, mask=facing_toward.mask))
ax2.imshow(np.ma.masked_array(layover, mask=facing_toward.mask), cmap=cmap_red)
ax2.imshow(np.ma.masked_array(shadow, mask=facing_away.mask), cmap=cmap_black)
f.show()
