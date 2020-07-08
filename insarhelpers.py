import numpy as np
import gdal
import sys
from datetime import datetime
sys.path.append('/home/myja3483/isce_tools/GIANT')
import json
from shapely.geometry.polygon import Polygon

# convert unwrapped phase to deformation:
def get_deformation(unw_ig, wavelength):
    disp = (unw_ig[1,:,:]*wavelength/(4*np.pi)) #unwrapped phase * wavelength (cm)/4pi
    return disp

# mask displacements based on coherence:
def mask_displacement(disp, coherence, coh_th):
    masked_disp = np.ma.masked_where(coherence < coh_th, disp)
    return masked_disp

#remove signal from reference area:
def ref_control(disp, ref_pt, size):
    ref_disp = disp[ref_pt[0]-size:ref_pt[0]+size, ref_pt[1]-size:ref_pt[1]+size].mean()
    corr_disp = disp - ref_disp
    return corr_disp

#retrieve dates and extract time between each image
def date_separation(dirs, days = None):
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
    norm_disp = disp/time
    return norm_disp

# get geographic extent of radar data:
def GetExtent(gt,cols,rows):
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
