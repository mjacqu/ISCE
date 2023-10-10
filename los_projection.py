import numpy as np
from osgeo import gdal
import scipy.signal

################
#initially: take mean of azimuth and mean of incidence angle for entire image.
# eventually crop azimuth and incidence angle to aoi used by giant
def crop2aoi():
    pass
###############

def read_los(fname):
    '''
    read isce output los.rdr and return arrays of incidence angle (at target, from vertical)
    and azimuth (counter clockwise from north). Returns incidence angle and azimuth
    arrays (degrees)

    parameters:
    fname (string): path to los.rdr file

    returns:
    inc (np.array)
    azi (np.array)
    raster (gdal object)
    '''
    raster = gdal.Open(fname)
    band1 = raster.GetRasterBand(1)
    band2 = raster.GetRasterBand(2)
    inc = band1.ReadAsArray()
    azi = band2.ReadAsArray()
    return inc, azi, raster


def rotate_azimuth(azi, direction = 'clockwise'):
    '''
    Rotate azimuth to clockwise from East (x-axis)

    parameters:
    azi ():         azimuth counter-clowise from North (ISCE convention) in degrees
                    or clockwise from North (topographic convention)
    direction (str):clockwise (default) or counterclockwise

    returns:
    theta:      azimuth counter clockwise from East (x-axis)

    '''
    azi = np.atleast_1d(azi)
    if direction == 'clockwise':
        if (azi < 0).all():
            raise ValueError ('Clockwise azimuth must be positive from North')
        theta = (360-(azi-90))%360
    else:
        theta = azi * -1
        theta = (360-(azi-90))%360
    return theta


def pol2cart(theta, phi, r=1):
    '''
    theta (deg) = angle counter-clockwise from x in x-y plane
    phi (deg) = positive down from z in x-z plane
    r (deg) = radius (default = 1)

    returns:
    (x,y,z)     LOS in vector notation.
    '''
    x = r * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    y = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z = r * np.cos(np.radians(phi))
    return x, y, z


def reverse_vector(v):
    '''
    Multiply line of sight vector by -1 to reverse direction.

    Parameters:
    v (tuple)   Vector (x,y,z) pointing from target to platform_to_target

    returns:
    (x,y,z)     reversed LOS vector
    '''
    v_rev = tuple(i * -1 for i in v)
    return v_rev


def loadnfilter(fn, filter = None):
    '''
    Load slope (degress from horizontal) and aspect (degrees from N) tiff files
    and apply median filter.

    return:
    raster (np.array)
    dataset (gdal object)
    '''
    dataset = gdal.Open(fn)
    raster = dataset.GetRasterBand(1).ReadAsArray()
    raster[raster == -9999] = np.nan
    if filter is not None:
        filtered = scipy.signal.medfilt(raster, filter) # 11 cells ~110 m
        return filtered
    else:
        return raster, dataset


def slope_from_vertical(slope):
    '''
    Convert slope values from degrees from horizontal to degrees from vertical.

    parameters:
    slope (np.array)    Slope value(s) in degrees from horizontal

    return:
    vert_slope          Slope value(s) in degrees from vertical
    '''
    slope = np.atleast_1d(slope)
    vert_slope = slope + 90
    return vert_slope



def compute_delta(a, b):
    '''
    Compute angle between fall_line and line-of-sight vectors.

    Parameters:
    a:      downslope vector(s)
    b:      los vector from platform to target

    returns:
    delta           angle (radians) between a and b
    '''
    a_3d = np.dstack([np.asarray(i) for i in a])
    b_3d = np.dstack([np.asarray(i) for i in b])
    dot_product = np.sum(a_3d*b_3d, axis = 2)
    delta = np.arccos(dot_product/(np.linalg.norm(a_3d, axis = 2) * np.linalg.norm(b_3d, axis = 2)))
    return delta

def los2def(los_def, delta):
    '''
    Project line of sight deformations onto fall line to retrieve surface displacements.

    Parameters:
    los_def:    measured line-of-sigh deformations
    delta:      angle (rad) between los vector and fall line
    '''
    surface_def = los_def/np.cos(delta)
    return surface_def


r=1
S = pol2cart(np.radians(230), np.radians(125))
P_asc = pol2cart(np.radians(190), np.radians(41.8))
P_desc = pol2cart(np.radians(-10), np.radians(38.7))

delta_asc = np.arccos(np.dot(P_asc,S)/(np.linalg.norm(P_asc)*np.linalg.norm(S)))
delta_desc = np.pi - np.arccos(np.dot(P_desc,S)/(np.linalg.norm(P_desc)*np.linalg.norm(S)))

desc_los = np.cos(delta_desc) * r
asc_los = np.cos(delta_asc) * r

#test with very simple vectors:
# "ascending"
a = pol2cart(np.radians(180), np.radians(161.6), np.sqrt(3**2 + 1**2))
b = pol2cart(np.radians(180), np.radians(33), np.sqrt(3**2 + 2**2))
epsilon = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
b_prime = pol2cart(np.radians(0), np.radians(147), np.sqrt(3**2 + 2**2))
delta = np.arccos(np.dot(a,b_prime)/(np.linalg.norm(a)*np.linalg.norm(b_prime)))
D = 1
LOS = np.cos(delta) * D

# "descending"
a = pol2cart(np.radians(180), np.radians(116.6), np.sqrt(4**2 + 2**2))
b = pol2cart(np.radians(0), np.radians(14), np.sqrt(4**2 + 1**2))
epsilon = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
b_prime = pol2cart(np.radians(180), np.radians(166), np.sqrt(4**2 + 2**2))
delta = np.arccos(np.dot(a,b_prime)/(np.linalg.norm(a)*np.linalg.norm(b_prime)))
D = 1
LOS = np.cos(delta) * D
