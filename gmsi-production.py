#import sys
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
from datetime import datetime
import re
import rasterio
import dask.array as da
#import xarray as xr
#from rasterio.warp import calculate_default_transform, reproject, Resampling
#import pyproj

################# functions ##############################

def create_datetime_list(dirs):
    '''
    Parse temporal baselines from file names

    Arguments:
    dirs (list):    List of all files in path

    Returns:
    list    Temporal baselines (datetime objects) of all interferograms, parsed from filename in format YYYYMMDD_YYYYMMDD
    '''
    # Define the regex pattern to extract dates
    pattern = re.compile(r'S1.(\d{8})_(\d{8}).cc.tif')
    # Create a list of datetime objects from the directory names
    datetime_list = []
    for d in dirs:
        match = re.match(pattern, d)
        if match:
            start_date = datetime.strptime(match.group(1), '%Y%m%d')
            end_date = datetime.strptime(match.group(2), '%Y%m%d')
            datetime_list.append((start_date, end_date))
    # Calculate the datetime delta for each item in the list
    datetime_deltas = [end - start for start, end in datetime_list]
    return datetime_deltas


def delta_files(dirs, datetime_deltas, delta):
    '''
    Find all files that correspond to one temporal baseline

    Arguments:
    dirs (list):            List of all files in path
    datetime_deltas (list): Corresponding list of datetime deltas
    delta (int):            Temporal baseline in days

    Returns:
    list    List of paths to files with corresponding baselines
    '''
    boolean = [d.days == delta for d in datetime_deltas]
    dt_dirs = [item for item, boolean in zip(dirs, boolean) if boolean]
    return dt_dirs

# Function to read each raster as a dask array
def read_raster_as_dask_array(file_path):
    '''
    Read raster as dask array

    Arguments:
    file_path (str):    Path to file

    Return:
    dask array  
    '''
    with rasterio.open(file_path) as src:
        return da.from_array(src.read(1), chunks=(1024, 1024))
    

# Function to save out results to .tif

def save_raster(output_file, result_array, metadata_source, datatype=np.float32):
    '''
    Save new array to .tif with corresponding spatial metadata

    Arguments:
    output_file (str):      Filename for saving raster
    metadata_source (str):  Path to .tif with same spatial extent
    datatype(dtye):         Datatype; default=np.float32
    result_array (ndarray): Data array to be written to file as .tif

    '''
    output_file = output_file
    with rasterio.open(metadata_source) as src:
        meta = src.meta.copy()
        meta.update(dtype=datatype, count=1, nodata=-9999, compress='LZW')

        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(result_array.compute(), 1)


########################## processing ###############################
#set path
track = 'A117'
path = f'/Volumes/Science/CCAMM/gmsi-production/coherence/coh_{track}'
vis_path = f'/Volumes/Science/CCAMM/gmsi-production/visibility/vis_{track}'
output_path = '/Volumes/Science/CCAMM/gmsi-v2'
################### compute coherence medians #######################

# analyze temporal baselines
dirs = os.listdir(path)

# create list of all coherence files and extract temporal baselines
datetime_list = create_datetime_list(dirs)
b_temp = np.sort([d.days for d in np.unique(datetime_list)])

################# generate median coherence files ##################
for b in b_temp:
    raster_files = [os.path.join(path,f) for f in delta_files(dirs, datetime_list, b)]

    # Read all rasters into a list of dask arrays
    raster_arrays = [read_raster_as_dask_array(f) for f in raster_files]

    stacked_array = da.stack(raster_arrays, axis=0)
    median_array = da.mean(stacked_array, axis=0)

    output_file = os.path.join(path, f'median_{b}_days.tif')
    save_raster(output_file, median_array, raster_files[0])


########### Now find the time when the coherence drops below 0.3 ########

# Construct paths in right order (according to b_temp)
median_raster_files = [os.path.join(path, f'median_{b}_days.tif') for b in b_temp]

# Read all median rasters as dask arrays
median_arrays = [read_raster_as_dask_array(f) for f in median_raster_files]

# Threshold value
threshold = 0.3

'''
Initial suggestion using the loop 
'''

# Stack the arrays along a new dimension (time)
stacked_median_array = da.stack(median_arrays, axis=0)


# Initialize an array to store the number of days at which the value drops below the threshold
drop_below_threshold = da.zeros_like(median_arrays[0], dtype=np.float32)

days_until_drop = np.insert(b_temp,0,1)[:-1]
# Iterate over time intervals and update the drop_below_threshold array
for i, days in enumerate(days_until_drop):
    # Find where the value is below the threshold and hasn't already been marked
    below_threshold = (stacked_median_array[i, :, :] < threshold) & (drop_below_threshold == 0)
    drop_below_threshold = da.where(below_threshold, days, drop_below_threshold)

# Convert cells that never drop below the threshold to a high value or NaN if preferred
drop_below_threshold = da.where(drop_below_threshold == 0, 96, drop_below_threshold)

# Mask nan-values (areas outside country borders or path)
coverage_mask = median_arrays[0] == 0.000
coherence_decay = da.where(coverage_mask, np.nan, drop_below_threshold)

#log-normalize the coherence_decay
coherence_decay_norm = da.log2(coherence_decay)/np.log2(b_temp[-1])

# mask areas in drop_below_threshold where 6-day coherence < 0.3
# low_coh_mask = median_arrays[0] < 0.3 
# apply mask to whole stack:
#coherence_decay = da.where(low_coh_mask, np.nan, drop_below_threshold)

output_file = os.path.join(output_path, f'coherence_decay_lognorm_1_{track}.tif')
save_raster(output_file, coherence_decay_norm, median_raster_files[0])

############# Combine visibility and coherence decay ################


# load visibility
visibility = read_raster_as_dask_array(glob.glob(os.path.join(vis_path, '*.norm_scale_factor_masked.tif'))[0])
visibility_zeros = da.where(visibility == 0, np.nan, visibility)
visibility_neg_inf = da.where(visibility < -1000, np.nan, visibility_zeros)
visibility_array = da.where(visibility_zeros>10e3, np.nan, visibility_neg_inf)



# mask visibility based on coherence
# mask areas in drop_below_threshold where 6-day coherence < 0.3
#low_coh_mask = median_arrays[0] < 0.3

# apply mask to whole stack:
#visibility_masked = da.where(low_coh_mask, np.nan, visibility_array)

# load coherence decay
#coherence_decay = read_raster_as_dask_array(os.path.join(path, 'coherence_decay_lognorm_A088.tif'))

# mask of all areas in the coherence_decay that are nan in the visibility file:
# not needed?????
#coherence_decay_masked = da.where(da.isnan(visibility_array), np.nan, coherence_decay)

# multiply and normalize
gmsi = coherence_decay_norm * visibility_array
# Normalize the product array to the range [0, 1]
min_value = da.nanmin(gmsi).compute()
max_value = da.nanmax(gmsi).compute()
gmsi_norm = (gmsi - min_value) / (max_value - min_value)



# export
# Output file path for the result
output_file = os.path.join(output_path, 'gmsi_v2_lognorm_1_A177.tif')
metadata_source = glob.glob(os.path.join(vis_path, '*.norm_scale_factor_masked.tif'))[0]
save_raster(output_file, gmsi_norm, metadata_source)



################### create composite gmsi map ##########################

gmsi_path = '/Volumes/Science/CCAMM/gmsi-v2'

gmsi_files = glob.glob(os.path.join(gmsi_path, 'gmsi_v2_lognorm_1_*.tif'))


rasters = [read_raster_as_dask_array(p) for p in gmsi_files]

# Compute the maximum value across all rasters
max_raster = da.nanmax(da.stack(rasters, axis=0), axis=0)

out_fn = os.path.join(output_path, 'gmsi_v2_composite_alltracks.tif')
save_raster(out_fn, max_raster, gmsi_files[0])


# Determine which raster had the maximum value
def get_index_with_max_value(rasters, max_raster):
    # Create a dask array to store indices
    indices = da.zeros(max_raster.shape, dtype=int)
    
    for i, raster in enumerate(rasters):
        mask = (raster == max_raster)
        indices = da.where(mask, i, indices)
        
    return indices

index_raster = get_index_with_max_value(rasters, max_raster)
index_raster_masked = da.where(da.isnan(max_raster), np.nan, index_raster)

out_fn = os.path.join(output_path, 'gmsi_v2_orbit_index_alltracks.tif')
save_raster(out_fn, gmsi_files[0], index_raster_masked)


del max_raster
del index_raster
gc.collect()


############################## apply water mask to composite products ###############################
# Currently done on pre-computed rasters (i.e. load in raster from file), but 
# could eventually be included in the computation

file_path = '/Volumes/Science/CCAMM/gmsi-v2'
mask_path = '/Volumes/Science/CCAMM/gmsi-production/gmsi_swissTLM3D-2025-watermask.tif'

m = read_raster_as_dask_array(mask_path)

def apply_water_mask(file_path, filename, mask):
    print(f' loading {filename} .................')
    d = read_raster_as_dask_array(os.path.join(file_path, filename))
    d_masked = da.where(mask==1, np.nan, d)
    out_fn = os.path.join(file_path, f'{filename[:-4]}_masked.tif')
    print(f'saving {filename[:-4]}_masked.tif .................')
    save_raster(out_fn, d_masked, os.path.join(file_path, filename))
    print(f'Applied water mask to {filename}')
    del d


apply_water_mask(file_path, 'gmsi_v2_composite_alltracks.tif', m)
apply_water_mask(file_path, 'gmsi_v2_orbit_index_alltracks.tif', m)

##################### combine gmsi with visibility and apply water mask ###########################
# Currently done on pre-computed rasters (i.e. load in raster from file), but 
# could eventually be included in the computation

file_path = '/Volumes/Science/CCAMM/gmsi-v2'
mask_path = '/Volumes/Science/CCAMM/gmsi-production/gmsi_swissTLM3D-2025-watermask.tif'

m = read_raster_as_dask_array(mask_path)

def apply_water_mask(file_path, filename, mask):
    print(f' loading {filename} .................')
    d = read_raster_as_dask_array(os.path.join(file_path, filename))
    d_masked = da.where(mask==1, np.nan, d)
    out_fn = os.path.join(file_path, f'{filename[:-4]}_masked.tif')
    print(f'saving {filename[:-4]}_masked.tif .................')
    save_raster(out_fn, os.path.join(file_path, filename), d_masked)
    print(f'Applied water mask to {filename}')
    del d


#apply_water_mask(file_path, 'gmsi_v2_lognorm_1_A015.tif', m)
#apply_water_mask(file_path, 'gmsi_v2_lognorm_1_A088.tif', m)
#apply_water_mask(file_path, 'gmsi_v2_lognorm_1_A177.tif', m)
#apply_water_mask(file_path, 'gmsi_v2_lognorm_1_D066.tif', m)
#apply_water_mask(file_path, 'gmsi_v2_lognorm_1_D139.tif', m)
#apply_water_mask(file_path, 'gmsi_v2_lognorm_1_D168.tif', m)



#apply water mask to visibility 


