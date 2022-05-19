import sys
import os
import numpy as np
sys.path.append('./')
import interferogram
import richdem
import matplotlib.pyplot as plt
#for viz
import rasterio
import earthpy
import earthpy.spatial as es
import earthpy.plot as ep
import typing
import scipy.interpolate
import richdem


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


# Rotate the whole matrix to generate the new indices for each pixel
#https://en.wikipedia.org/wiki/Rotation_matrix
#x' = x*cos(theta)-y*sin(theta)
#y' = x*sin(theta)+y*cos(theta)
#where x,y = original coordinates, theta=rotation angle from horizontal x, x',y'=rotated coordinates

#Functions

def ind2xy(r, c, array):
    '''
    convert (row, colum) index into cartesian (x,y) coordinates: flip y-axis so that origin is bottom left corner
    '''
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

def rotate_xy(x: np.ndarray, y: np.ndarray, angle: float) -> typing.Tuple[np.ndarray, np.ndarray]:
    x_prime = x*np.cos(angle)-y*np.sin(angle) #distance from sensor plane
    y_prime = x*np.sin(angle)+y*np.cos(angle) #distance from top edge of image
    return x_prime, y_prime

def rotate_array(array: np.ndarray, angle: float) -> np.ndarray:
    """
    Reinterpolate array onto rotated grid.

    Parameters
    ----------
    array
        Rotation relative to center of bottom left cell.
    angle
        degrees counter-clockwise from North
    """
    angle = -np.radians(angle)
    x, y = ind2xy(*np.indices(array.shape), array) #rows = y, columns = x
    x_prime, y_prime = rotate_xy(x, y, angle)
    range_x_prime = np.arange(x_prime.min(), x_prime.max() + 1, 1)
    range_y_prime = np.flip(np.arange(y_prime.min(), y_prime.max() + 1, 1))
    xy_prime_grid = np.meshgrid(range_x_prime, range_y_prime)
    x_back, y_back = rotate_xy(xy_prime_grid[0], xy_prime_grid[1], -angle)
    interpolator = scipy.interpolate.RegularGridInterpolator(
        points=(-y[:, 0], x[0]),
        values=array,
        bounds_error=False
    )
    interpolated = interpolator(np.column_stack((-y_back.reshape(-1), x_back.reshape(-1))))
    return interpolated.reshape(x_back.shape), x_prime, y_prime

def rotate_array_inverse(rot_array, x_prime, y_prime):
    '''
    Rotate array back to north-up orientation.
    '''
    range_x_prime = np.arange(x_prime.min(), x_prime.max() + 1, 1)
    range_y_prime = np.flip(np.arange(y_prime.min(), y_prime.max() + 1, 1))
    interpolator = scipy.interpolate.RegularGridInterpolator(
        points=(-range_y_prime, range_x_prime),
        values=rot_array,
        bounds_error=False
    )
    interpolated = interpolator(np.column_stack((-y_prime.reshape(-1), x_prime.reshape(-1))))
    return interpolated.reshape(x_prime.shape)

def calc_projected_height(theta, rot_dem, cell_size):
    '''
    Calculate the projected height for visibility analysis
    '''
    dist = np.indices(rot_dem.shape)[1]*cell_size
    p_height = np.tan(np.radians(90-np.median(theta)))*dist + rot_dem
    return p_height

def calc_layover_distance(theta, rot_dem, cell_size):
    '''
    Calculate distance from inclined plane at image edge to terrain point.

    Parameters:
    ---------------
    theta: float
        Incidence angle positive down from vertical in degrees
    rot_dem: np.ndarray
        Terrain elevation rotated into radar view direction

    '''
    beta = np.radians(90 - theta) #inclination of plane relative to horizontal
    l = rot_dem * np.tan(np.radians(theta))
    dist = np.indices(rot_dem.shape)[1]*cell_size
    d = dist - l
    return d * np.sin(beta)

def calc_foreshortening(path, heading, incidence, orbit='ascending'):
    '''
    Find areas affected by foreshortening based.

    Parameters:
    --------------
    dem: str
        Path to digital elevation model in a local coordinate system (m)
    heading: float
        Satellite heading direction clockwise from north
    incidence: ndarray
        Raster with incidence angle positive down from vertical.
    orbit: str
        'ascending' or 'descending'
    '''
    # calculate slope and aspect of DEM
    dem = richdem.LoadGDAL(path)
    slope_deg = richdem.TerrainAttribute(dem, attrib='slope_degrees')
    aspect = richdem.TerrainAttribute(dem, attrib='aspect')
    if orbit=='ascending':
        A = np.radians(aspect + heading + 180)
    if orbit=='descending':
        A = np.radians(aspect - heading)
    R = np.sin(np.median(incidence)-np.radians(slope_deg)*np.sin(A))
    return np.ma.masked_outside(R, 0, 0.4)
