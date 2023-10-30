import sys
import los_projection as lp
import numpy as np
sys.path.append('./')
import xdem
import interferogram
import shadow_functions
import insarhelpers
from osgeo import gdal
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib import colors
import rasterio



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
    #path ='/Volumes/Science/SpitzerStein/testdata/20151006_20151018'
    path='/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/testdata/20180716_20180728'
    #path = '/Volumes/Science/ChamoliSAR/results/A56/20200802_20200814'
    #path='/scratch-third/mylenej/radar/SpitzerStein/results/asc_88/20151006_20151018'
)

look_direction = np.median(ifg.los[1])*-1
heading = look_direction + 90

#Question: does the slight variation of look_direction matter?

# 2. Input: DEM
#Spitzer Stein DEM (note: needs to be in meters for slope calculation to work)
path = '/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/kandersteg10m.tif'
#path = '/scratch-second/mylenej/Projects/Kandersteg/kandersteg10m.tif'
#path = '/Volumes/Science/ChamoliSAR/HiMAT-DEM/Chamoli_Sept2015_8m_crop_gapfill.tif'
dem = xdem.DEM(path)
cell_size = dem.res[0]

'''
#reproject los parameters to the extent of the dem (not used right now because of artefacts in the los data)
reprojected = insarhelpers.reproject_to_target_raster(
    source_path = '/Volumes/Science/SpitzerStein/testdata/20151006_20151018/merged/los.rdr.geo',
    target = dem,
    target_srs = 'EPSG:2056')
los_reprojected = reprojected.ReadAsArray() #this are now the look angle and incidence angles reprojected to CH1903 LV95 (same as DEM)
'''

# run the functions
prime_array, x_prime, y_prime = shadow_functions.rotate_array(dem.data, angle=heading)
h_p = shadow_functions.calc_projected_height(np.median(ifg.los[0]), prime_array, cell_size)
rc = shadow_functions.calc_layover_distance(np.median(ifg.los[0]), prime_array, cell_size)
vis = h_p >= np.fmax.accumulate(h_p, axis=1)
vis_reg = shadow_functions.rotate_array_inverse(vis, x_prime, y_prime)
vis_reg[vis_reg>=1] = np.nan
#layover 1 (search for maximum from left to right)
layover1 = rc >= np.fmax.accumulate(rc, axis=1)
lay1_reg = shadow_functions.rotate_array_inverse(layover1, x_prime, y_prime)
lay1_reg[lay1_reg>=1] = np.nan
#layover 2 (search for minimum from right to left)
layover2 = np.fliplr(rc) <= np.fmin.accumulate(np.fliplr(rc), axis=1)
lay2_reg = shadow_functions.rotate_array_inverse(np.fliplr(layover2), x_prime, y_prime)
lay2_reg[lay2_reg>=1] = np.nan


#Foreshortening following Cigna et al. compression factor analysis:
fs = shadow_functions.calc_foreshortening(path, heading, ifg.los[0], orbit='ascending')


hillshade = xdem.terrain.hillshade(dem, azimuth=315.0, altitude=45.0)


# measureable velocity
slope_deg = xdem.terrain.slope(dem, resolution=dem.res)
aspect = xdem.terrain.aspect(dem)

azi_rot = lp.rotate_azimuth(np.median(ifg.los[1]), direction = 'cc') #azimuth rotated to cw from E (x-axis)
target_to_platform = lp.pol2cart(azi_rot, np.median(ifg.los[0])) 
p2t = lp.reverse_vector(target_to_platform)

aspect_rotated = lp.rotate_azimuth(scipy.signal.medfilt(aspect.data, 11))
vert_slope = lp.slope_from_vertical(scipy.signal.medfilt(slope_deg.data, 11))
slope_vectors = lp.pol2cart(aspect_rotated, vert_slope)

p2t_3d = np.dstack([np.asarray(i) for i in p2t])
slope_vec_3d = np.dstack([np.asarray(i) for i in slope_vectors])
prop_def = np.sum(p2t_3d*slope_vec_3d, axis = 2) #just the dot-product of the two vectors as scaling factor as proposed by Handwerger #note from 10-2023: this is a dot-product but Al does not use a dot product...



#plot results

cmap_shadow = colors.ListedColormap(['k'])
cmap_lay1 = colors.ListedColormap(['red'])
cmap_lay2 = colors.ListedColormap(['orange'])
cmap_foreshortening = colors.ListedColormap(['Gold'])


f, ax = plt.subplots()
ax.imshow(hillshade.data, cmap='Greys_r', zorder=0)
ax.imshow(vis_reg, alpha=0.95, cmap=cmap_shadow, zorder=10)
ax.imshow(lay1_reg, alpha=0.95, cmap=cmap_lay1, zorder=11)
ax.imshow(lay2_reg, alpha=0.95, cmap=cmap_lay2, zorder=12)
ax.imshow(fs, alpha=0.95, cmap=cmap_foreshortening, zorder=4)
v_rel = ax.imshow(prop_def, alpha=0.8, zorder=1, cmap='seismic')
f.colorbar(v_rel)
#plt.imshow(layover, alpha=0.95)
f.show()

#aspect vs visible deformation for slopes >30°
plt.scatter(aspect.data[slope_deg.data>20], prop_def[slope_deg.data>20], s=0.0001)
plt.show()


#save prop_def as geotiff

meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': -9999,
    'width': dem.shape[1],
    'height': dem.shape[0],
    'count': dem.count,
    'crs': 'EPSG:2056',  # You should use the appropriate CRS for your data
    'transform': dem.transform  # Define the transformation
}

with rasterio.open('output.tif', 'w', **meta) as dst:
    dst.write(prop_def, 1) 


#synthetic example for topography:
synt_topo = np.zeros((100, 100))

# Define the size of the pyramid base
base_size = 100

# Fill the array to create the pyramid
for i in range(base_size):
    synt_topo[i:100-i, i:100-i] += 1

# calculate prop_def for synthetic topography:
#todo: creat synthetic look angle looking due east and 45° down
azi_rot = [0]#azimuth rotated to cw from E (x-axis) --> directly east looking
target_to_platform = lp.pol2cart(azi_rot, 45) 
p2t = lp.reverse_vector(target_to_platform)

synt_slope = xdem.terrain.slope(synt_topo, resolution=1)
synt_aspect = xdem.terrain.aspect(synt_topo)

synt_aspect_rotated = lp.rotate_azimuth(scipy.signal.medfilt(synt_aspect, 11))
synt_vert_slope = lp.slope_from_vertical(scipy.signal.medfilt(synt_slope, 11))
synt_slope_vectors = lp.pol2cart(synt_aspect_rotated, synt_vert_slope)

p2t_3d = np.dstack([np.asarray(i) for i in p2t])
slope_vec_3d = np.dstack([np.asarray(i) for i in synt_slope_vectors])
synt_prop_def = np.sum(p2t_3d*slope_vec_3d, axis = 2)

#Simple synthetic example:
#radar vector:
radar = lp.rotate_azimuth(-270, direction='cc')[0], 45
radar = lp.pol2cart(*radar)
radar = lp.reverse_vector(radar)
radar = np.asarray(radar)
np.sqrt(np.sum(radar**2))

slope = lp.pol2cart(theta=0, phi=135)
slope = np.asarray(slope)
np.sqrt(np.sum(slope**2))

# np.asarray([1, 1, -1], dtype=float)
# slope /= np.sqrt(np.sum(slope**2))
projection = np.sum(radar * slope)
round(projection, 12)
assert projection == slope[2]


#synthetic example for layover 1 and 2:
h_t = np.array([1,1,1,1,1,1,1,1,2,4,8,16,32,64,70,75,78,75,72,69,66,63,60,57,54,51,48,47,46,45,44,44,44,44,44,44,44,44])

beta = np.radians(90 - np.median(ifg.los[0])) #inclination of plane relative to horizontal
l = h_t * np.tan(np.radians(np.median(ifg.los[0])))
dist = np.indices(h_t.shape)[0]*10
d = dist - l
rc = d * np.sin(beta)

layover1 = rc >= np.fmax.accumulate(rc)
layover2 = np.flip(rc) <= np.fmin.accumulate(np.flip(rc))

plt.figure()
plt.plot(dist, rc, label='rc', zorder=0)
plt.plot(dist, h_t, label='terrain', zorder=1)
plt.plot(dist[~layover1], h_t[~layover1], label='layover1', zorder=2)
#plt.plot(dist, np.flip(rc), label='flip rc')
plt.plot(dist[np.flip(~layover2)], h_t[np.flip(layover2)], label='layover2', zorder=3)
#plt.plot(dist[~layover2], h_t[~layover2])
plt.legend()

plt.figure()
plt.plot(np.flip(rc))
plt.plot(np.fmin.accumulate(np.flip(rc)))

'''
#OLD STUFF

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

'''