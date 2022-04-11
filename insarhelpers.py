import numpy as np
from osgeo import gdal
import sys
from datetime import datetime
sys.path.append('/home/myja3483/isce_tools/GIANT')
import json
import os
from osgeo import gdal_array
from shapely.geometry.polygon import Polygon
from PIL import Image, ImageEnhance
import rasterio
import rasterio.merge

#load files from merged directory
def load_rdr(path, fn):
    path_to_fn = os.path.join(path,fn)
    file = gdal_array.LoadFile(path_to_fn)
    ds=gdal.Open(os.path.join(path,fn))
    return file, ds

# convert unwrapped phase to deformation:
def get_deformation(unw_ig, wavelength):
    '''
    Turn unwrapped phase into deformation.

    Arguments:
    unw_ig (ndarray):   Array of unwrapped Phase
    wavelength:         Instrument wavelength.

    Returns:
    displacement(ndarray)
    '''
    disp = (unw_ig[1,:,:]*wavelength/(4*np.pi)) #unwrapped phase * wavelength (cm)/4pi
    return disp

# mask displacements based on coherence:
def mask_displacement(disp, coherence, coh_th):
    '''
    Mask displacements based on set coherence threshold

    Arguments:
    disp (ndarray):         Array of displacements
    coherence (ndarray):    Array with coherence values
    coh_th (float):         Threshold for coherence masking

    Returns:
    Masked displacements (ndarray)
    '''
    masked_disp = np.ma.masked_where(coherence < coh_th, disp)
    return masked_disp

#remove signal from reference area:
def ref_control(disp, ref_pt, size):
    '''
    Remove mean displacement around reference point from displacement

    Arguments:
    disp (ndarray):   Array of unwrapped Phase
    ref_pt (array):   [row,col] coordinates of reference points
    size (int):       Number of pixels around reference point to average

    Returns:
    corrected displacement (ndarray)
    '''
    ref_disp = disp[ref_pt[0]-size:ref_pt[0]+size, ref_pt[1]-size:ref_pt[1]+size].mean()
    corr_disp = disp - ref_disp
    return corr_disp

#retrieve dates and extract time between each image
def date_separation(dirs, days = None):
    '''
    Determines how many days separate each interferogram pair based on directory
    labels (YYYYMMDD_YYYYMMDD).

    Arguments:
    dirs (list):   List of all directories (YYYYMMDD_YYYYMMDD)
    days (int):    Number of days to divide the number of days by.

    Returns:
    number_of_days (list)
    day_groups (list)
    '''
    dates = [d.split('_') for d in dirs]
    datetimes = [[datetime.strptime(d, '%Y%m%d') for d in group] for group in dates]
    timedeltas = [d[1] - d[0] for d in datetimes]
    number_of_days = [dt.days for dt in timedeltas]
    if days is not None:
        day_groups = [nd/days for nd in number_of_days]
        return number_of_days, day_groups
    return number_of_days

# normalize displacement to some amount of time:
def normalize_disp(disp, time):
    '''
    Normalize displacement by some time (e.g., number of days between
    interferograms, some number of days etc.)

    Arguments:
    disp (ndarray):   Array of unwrapped Phase
    time (float):     Amount of time

    Returns:
    normalized_disp (ndarray)
    '''
    norm_disp = disp/time
    return norm_disp

# get geographic extent of radar data:
def GetExtent(gt,cols,rows):
    '''
    Get geographic extent of radar data

    Arguments:
    gt ():        raster GeoTransform
    cols (int):   Number of cols in radar image
    rows (int)    Number of rows in radar image

    Returns:
    extent (list)
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext


# Function to normalize the grid values
def normalize_rgb(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def get_los(az, l):
    '''
    Get dx and dy for plotting flight direction and los for plotting vectors.

    Parameters:
    los (array):    array with radar line of sight
    l (float):      length of plotting vector


    Returns:
    dx, dy (float): x and y offsets for plotting
    '''
    mean_az = az.mean()
    az_from_east = 90-(-mean_az - 180)
    dx = np.cos(np.radians(az_from_east))*l
    dy = np.sin(np.radians(az_from_east))*l
    return dx, dy


def make_MS_img(mosaic_list, band_order, res = None, bounds = None, brightness_f = 1, contrast_f = 1):
    '''
    Mosaic image tiles and return plottable image. Set to true color (band_order = [0,1,2]
    or false color image (band_order = [3,1,0]). Adjust brightness and contrast.
    '''
    mosaic, out_transform = rasterio.merge.merge(mosaic_list, res = res, bounds = bounds)
    band_1 = normalize_rgb(mosaic[band_order[0]])
    band_2 = normalize_rgb(mosaic[band_order[1]])
    band_3 = normalize_rgb(mosaic[band_order[2]])
    #red = insarhelpers.normalize_rgb(mosaic[0])
    #green = insarhelpers.normalize_rgb(mosaic[1])
    #blue = insarhelpers.normalize_rgb(mosaic[2])
    #nir = insarhelpers.normalize_rgb(mosaic[3])
    #planet_rgb = (np.dstack((nir, green, red))* 255.0) .astype(np.uint8)
    img_stack = (np.dstack((band_1, band_2, band_3))* 255.0).astype(np.uint8)
    img = Image.fromarray(img_stack) # turn in to PIL image
    brightness = ImageEnhance.Brightness(img)
    b_factor = brightness_f
    planet_bright = brightness.enhance(b_factor)
    contrast = ImageEnhance.Contrast(planet_bright)
    c_factor = contrast_f
    planet_enh = contrast.enhance(c_factor)
    mosaic_extent = [out_transform[2], out_transform[2]+(mosaic.shape[2]*out_transform[0]),
                    out_transform[5]+(mosaic.shape[1]*out_transform[4]), out_transform[5]]
    return planet_enh, mosaic_extent
