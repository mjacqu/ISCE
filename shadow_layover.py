import sys
sys.path.append('../MudCreek')
import los_projection as lp
import numpy as np
sys.path.append('./')
import interferogram
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
#Spitzer Stein DEM (note: needs to be in meters for slope calculation to work)
path = '/Users/mistral/Documents/ETHZ/Science/SpitzerStein/kandersteg10m.tif'

# Load DEM
dem = richdem.LoadGDAL(path)

#load radar data
ifg = interferogram.Interferogram(
    path ='/Volumes/Science/SpitzerStein/testdata/20151006_20151018'
    #path='/Users/mistral/Documents/ETHZ/Science/SpitzerStein/testdata/20151006_20151018'
)

#los vectors for plotting
#los_dx, los_dy, fd_dx, fd_dy = ifg.get_los_vec(500) #for plotting los arrows

# calculate slope and aspect of DEM
slope_deg = richdem.TerrainAttribute(dem, attrib='slope_degrees')
slope_dx = richdem.TerrainAttribute(dem, attrib='slope_riserun')
aspect = richdem.TerrainAttribute(dem, attrib='aspect')

look_angle = np.radians(np.median(ifg.los[0]))

#Todo: resample look angle and heading to DEM grid.
#t_srs = 'EPSG:2056'
#options = gdal.WarpOptions(options=['et' ,'t_srs', 'f'],
#    errorThreshold=0.01,
#    dstSRS=t_srs,
#    format='VRT'
#)

#los = gdal.Warp('',
#    gdal.Open(os.path.join(ifg.path,'los.rdr.geo')),
#    options=options
#)
#dst = gdal.GetDriverByName('VRT').Create('', dem.shape[0], dem.shape[1], 1, gdalconst.GDT_Float32)
#dst.SetGeoTransform(dem.geotransform)
#dst.SetProjection(dem.projection)
#los_resamp = gdal.ReprojectImage(los, dst, los.GetProjection(), dem.projection, gdalconst.GRA_Bilinear)


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

#Build algorithm for type-2 shadow classification ("passive shadow")
heading = look_direction + 90
'''
#painstaking approach of generating the line

#all in map coordinates


#single test index
r = 1500
c = 500

p1=np.asarray((0,0))
#p2=np.asarray((lp.pol2cart(lp.rotate_azimuth(heading), 90)[0], lp.pol2cart(lp.rotate_azimuth(heading), 90)[1]))
p2=np.asarray((-0.16, 0.98))
p3=np.asarray(ind2xy(r,c,dem))
linalg.norm(np.cross(p2-p1, p1-p3))/linalg.norm(p2-p1)

#by hand:
#|ax_1 +b<_1|/sqrt(a**2 + b**2)
((0.98*500)+(0.16*500)+0)/(np.sqrt((0.98**2)+(0.16**2)))
'''


#AAAAAACTUALLY, just rotate the whole matrix to generate the new indices for each pixel
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

rot = -np.radians(360-heading) #theta in radians
x, y = ind2xy(*np.indices(dem.shape), dem) #rows = y, columns = x
x_rot = x*np.cos(rot)-y*np.sin(rot) #cells from sensor plane
y_rot = x*np.sin(rot)+y*np.cos(rot) #cells from top edge of image

f, ax = plt.subplots(2,3)
ax[0,0].imshow(np.indices(dem.shape)[0])
ax[0,0].set_title('dem shape rows')
ax[1,0].imshow(np.indices(dem.shape)[1])
ax[1,0].set_title('dem shape columns')
ax[0,1].imshow(y)
ax[0,1].set_title('y')
ax[1,1].imshow(x)
ax[1,1].set_title('x')
ax[0,2].imshow(y_rot)
ax[0,2].set_title('y_rot')
ax[1,2].imshow(x_rot)
ax[1,2].set_title('x_rot')
f.show()

#lines of same line of sight
los_mask = (y_rot>=1000) & (y_rot < 1001)
plt.imshow(los_mask)
plt.show()


#lines of equal range
range_mask = (x_rot>1000) & (x_rot < 1001)
plt.imshow(range_mask)
plt.show()

################ new simple attempt? ###############
los_mask = (y_rot>=1000) & (y_rot < 1001)
distances = x_rot[los_mask]
d_sort = np.argsort(distances)
d = distances[d_sort]
h_los = dem[los_mask][d_sort]
###################################################

#true distance from radar projection plane:
cell_size = 10 #m
scaled_cell = np.cos(rot) * cell_size #scaled distance between cell centers scaled to account for rotation.
dist_from_projplane = x_rot * scaled_cell #distance in meters for each point from radar projection plane

plt.imshow(dist_from_projplane)
plt.show()

#Get terrain height along one line of sight at true distance from radar
h_los = dem[(y_rot>1000) & (y_rot < 1001)]
p_dist = dist_from_projplane[(y_rot>1000) & (y_rot < 1001)]
#plot terrain hight versus distance: Note (zoom in) data is not sorted by distance!
plt.scatter(p_dist, h_los)
plt.plot(p_dist, h_los, color='r')
plt.xlabel('Distance from projection plane (m)')
plt.ylabel('Elevation (m)')
plt.show()

#check
def check_los_direction(p_dist, h_los):
    if p_dist[0] < p_dist[-1]:
        print(f"{p_dist[0]} is smaller than {p_dist[-1]} \n do not flip")
        h_los_flp = h_los
    else:
        print(f"{p_dist[0]} is larger than {p_dist[-1]} \n flipping!")
        h_los_flp = np.flip(h_los)
        p_dist_flp = np.flip(p_dist)
    return h_los_flp, p_dist_flp

h_los_flp, p_dist_flp = check_los_direction(p_dist, h_los)

# The plot still looks the same, but the arrays are flipped so that the distance
# from the proj plane increases from left to right.
plt.plot(p_dist_flp, h_los_flp, 'r')
plt.scatter(p_dist_flp, h_los_flp)
plt.show()

#sort distance array such that distance increases from left to right at every step
sorted_dist = np.sort(p_dist_flp)
sort_key = np.argsort(p_dist_flp) #position of each distance in sorted distance array
sorted_elevation = [element for _, element in sorted(zip(sort_key, h_los_flp))] #apply sort key to elevation

plt.plot(sorted_dist, sorted_elevation, 'r')
plt.scatter(sorted_dist, sorted_elevation)
plt.show()

#################continuation from simple solution ####################

plt.plot(d, h_los, 'r')
plt.scatter(d, h_los)
plt.show()


#get projected height from sorted_dist
cell_size=10
dist=d*cell_size

def calc_projected_height(theta, dist, h):
    p_height = np.tan(np.radians(90-np.median(theta)))*dist + h
    return p_height

p_height = calc_projected_height(ifg.los[0], sorted_dist, sorted_elevation)

#plot projected height and elevation along one line of sight
f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
ax1.plot(sorted_dist, p_height)
ax2.plot(sorted_dist, sorted_elevation)
ax2.axis('equal')
f.show()

#difference projected height from left to right
diff = np.diff(p_height)

#if diff is negative --> terrain dipping away from radar
diff_sign = np.sign(diff)
neg_diff = np.ma.masked_where(diff_sign!=-1, p_height[:-1]) #all points that have a negative diff
sign_change = ((np.roll(diff_sign, 1) - diff_sign) != 0).astype(int) #
change_idx = np.where(sign_change!=0)[0][::2] #idx where there is a change from positive diff to negative diff (and not the change back)
h_change_idx = p_height[change_idx]


f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
ax1.scatter(sorted_dist, p_height)
ax1.scatter(sorted_dist[:-1], neg_diff)
ax1.scatter(sorted_dist[:-1], np.ma.masked_where(sign_change==0, sign_change))
ax2.plot(sorted_dist[:-1], diff)
f.show()


#find all elements change_idx and idx where h = h_change_idx and return
def find_ele(p_height, start_idx, val):
    #for ele in p_height[start_idx:]:
    for end_idx in np.arange(start_idx+1,len(p_height)):
        if p_height[end_idx] >= val:
            return np.arange(start_idx, end_idx)
        #if ele >= val:
        #    ele_idx = np.where(p_height==ele)
        #    print(ele_idx)
        #return ele_idx

'''
#test with simple example
test_height = np.asarray([1,3,5,6,4,3,1,3,5,7,8,9,6,3,4,7,8,9,11,15])
test_diff=np.diff(test_height)
test_diff_sign = np.sign(test_diff)
test_sign_change = ((np.roll(test_diff_sign, 1) - test_diff_sign) != 0).astype(int)
ch_idx = np.where(test_sign_change!=0)[0][::2]
h_ch_idx = test_height[ch_idx]

shadow_array = []
for i,s in zip(ch_idx, h_ch_idx):
    temp = find_ele(test_height, i, s)
    shadow_array = np.concatenate((shadow_array, temp), axis=0).astype(int)
    mask_array = np.ones(len(test_height), dtype=bool)
    mask_array[shadow_array] = False

plt.plot(test_height)
plt.plot(np.ma.masked_where(mask_array, test_height), 'r')
plt.show()
'''


def make_shadow_mask(ch_idx, h_ch_idx, p_h, terrain):
    shadow_array = []
    for s,v in zip(ch_idx, h_ch_idx):
        temp = find_ele(p_h, s, v)
        shadow_array = np.concatenate((shadow_array, temp), axis=0).astype(int)
    mask_array = np.ones(len(p_h), dtype=bool)
    mask_array[shadow_array] = False
    shadow_mask = np.ma.masked_where(mask_array, p_h)
    terrain_shadow = np.ma.masked_where(mask_array, terrain)
    return shadow_mask, terrain_shadow

rdr_shadow_mask, terrain_shadow = make_shadow_mask(change_idx, h_change_idx, p_height, sorted_elevation)

f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
ax1.plot(sorted_dist, p_height, 'k')
ax1.plot(sorted_dist, rdr_shadow_mask, 'r', label='shadow')
ax1.set_ylabel('Projected height')
ax1.legend()
ax2.plot(sorted_dist, sorted_elevation, 'k')
ax2.plot(sorted_dist, terrain_shadow, 'r', label='shadow')
ax2.set_ylabel('Terrain elevation')
ax2.set_xlabel('Distance from projection plane')
ax2.axis('equal')
f.show()
