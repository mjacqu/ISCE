#import sys
import os
import numpy as np
import glob
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
from datetime import datetime
import re
import rasterio
import dask.array as da
import xarray as xr
#from rasterio.warp import calculate_default_transform, reproject, Resampling
#import pyproj

#set path
#path = '/Volumes/Science/SpitzerStein/asc_88'
#path = '/Volumes/Science/SpitzerStein/desc_139'
path = '/Volumes/Science/CCAMM/gmsi-production/coherence/coh_A088'


# analyze temporal baselines
dirs = os.listdir(path)

def create_datetime_list(dirs):
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
    boolean = [d.days == delta for d in datetime_deltas]
    dt_dirs = [item for item, boolean in zip(dirs, boolean) if boolean]
    return dt_dirs

datetime_list = create_datetime_list(dirs)
b_temp = np.sort([d.days for d in np.unique(datetime_list)])

raster_files = [os.path.join(path,f) for f in delta_files(dirs, datetime_list, b_temp[4])]

############### calculating median with rasterio and dask  for memory efficiency ######################
# Function to read each raster as a dask array
def read_raster_as_dask_array(file_path):
    with rasterio.open(file_path) as src:
        return da.from_array(src.read(1), chunks=(1024, 1024))

# Read all rasters into a list of dask arrays
raster_arrays = [read_raster_as_dask_array(f) for f in raster_files]

stacked_array = da.stack(raster_arrays, axis=0)
median_array = da.mean(stacked_array, axis=0)

# To save the median array as a new raster file
output_file = os.path.join(path, f'median_{b_temp[4]}_days.tif')
with rasterio.open(raster_files[0]) as src:
    meta = src.meta.copy()
    meta.update(dtype=np.float32, count=1, compress='LZW')
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(median_array.compute(), 1)


########### Now find the time when the coherence drops below 0.4 ########



# Construct paths in right order (according to b_temp)
median_raster_files = [os.path.join(path, f'median_{b}_days.tif') for b in b_temp]

# Read all median rasters as dask arrays
median_arrays = [read_raster_as_dask_array(f) for f in median_raster_files]

# Threshold value
threshold = 0.5

'''
Initial suggestion using the loop 
'''

# Stack the arrays along a new dimension (time)
stacked_median_array = da.stack(median_arrays, axis=0)


# Initialize an array to store the number of days at which the value drops below the threshold
drop_below_threshold = da.zeros_like(median_arrays[0], dtype=np.float32)

# Iterate over time intervals and update the drop_below_threshold array
for i, days in enumerate(b_temp):
    # Find where the value is below the threshold and hasn't already been marked
    below_threshold = (stacked_median_array[i, :, :] < threshold) & (drop_below_threshold == 0)
    drop_below_threshold = da.where(below_threshold, days, drop_below_threshold)

# Convert cells that never drop below the threshold to a high value or NaN if preferred
drop_below_threshold = da.where(drop_below_threshold == 0, 100, drop_below_threshold)

# mask areas in drop_below_threshold where 6-day coherence < 0.3
low_coh_mask = median_arrays[0] < 0.3

# apply mask to whole stack:
below_threshold_masked = da.where(low_coh_mask, np.nan, drop_below_threshold)

# Output file path for the result
output_file = os.path.join(path, 'output_drop_below_threshold_v4_0.5.tif')

# Write the result to a new raster file
with rasterio.open(median_raster_files[0]) as src:
    meta = src.meta.copy()
    meta.update(dtype=np.float32, count=1, nodata=-9999, compress='LZW')
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(below_threshold_masked.compute(), 1)


############# Combine visibility and coherence decay ################
vis_path = '/Volumes/Science/CCAMM/gmsi-production/visibility/A-088'

# load visibility
visibility = read_raster_as_dask_array(os.path.join(vis_path, 'A088.norm_scale_factor_masked.tif'))
visibility_array = da.where(visibility == 0, np.nan, visibility)


# mask visibility based on coherence
# mask areas in drop_below_threshold where 6-day coherence < 0.3
low_coh_mask = median_arrays[0] < 0.3

# apply mask to whole stack:
visibility_masked = da.where(low_coh_mask, np.nan, visibility_array)

# load coherence decay
coherence_decay = read_raster_as_dask_array(os.path.join(path, 'output_drop_below_threshold_v4_0.5.tif'))

# mask of all areas in the coherence_decay that are nan in the visibility file:
coherence_decay_masked = da.where(da.isnan(visibility_array), np.nan, coherence_decay)

# multiply and normalize
gmsi = coherence_decay * visibility_array
# Normalize the product array to the range [0, 1]
min_value = da.nanmin(gmsi).compute()
max_value = da.nanmax(gmsi).compute()
gmsi_norm = (gmsi - min_value) / (max_value - min_value)



# export
# Output file path for the result
output_file = os.path.join(path, 'gmsi_norm_v1.tif')

# Write the result to a new raster file
with rasterio.open(os.path.join(vis_path, 'A088.norm_scale_factor_masked.tif')) as src:
    meta = src.meta.copy()
    meta.update(dtype=np.float32, count=1, nodata=-9999, compress='LZW')
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(gmsi_norm.compute(), 1)


