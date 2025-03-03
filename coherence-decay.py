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


'''
Argmax test (but with dask) and initially masking anything below 0.3... didn't work out but may be worth investigating, though other approach (in gmsi-production)
seems to work fine. 

'''
threshold = 0.5

# mask all areas in all arrays that have a value below 0.3 in the 6-day median
low_coh_mask = median_arrays[0] < 0.3

# apply mask to whole stack:
masked_median_arrays = [da.where(low_coh_mask, np.nan, array) for array in median_arrays]

# stack arrays along the time dimension
stacked_median_array = da.stack(masked_median_arrays, axis=0)

# Create a boolean array where values below the threshold are True
below_threshold = stacked_median_array < threshold

# Use argmax to find the first instance where the value is below the threshold
first_below_threshold = below_threshold.argmax(axis=0)

# Map the indices to the corresponding time intervals
days_to_drop = np.array(b_temp)[first_below_threshold]

# Handle cells that never drop below the threshold
never_dropped = below_threshold.sum(axis=0) == 0
days_to_drop = da.where(never_dropped, 100, days_to_drop)

# Output file path for the result
output_file = os.path.join(path, 'output_drop_below_threshold_v3_argmax0.5.tif')

# Write the result to a new raster file
with rasterio.open(median_raster_files[0]) as src:
    meta = src.meta.copy()
    meta.update(dtype=np.float32, count=1, nodata=-9999, compress='LZW')
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(days_to_drop.compute(), 1)



################### code from original test runs ##################################
'''

#calculate median coherence image for different temporal baselines
def med_coh_ts(dirs,tbs):
    coh_medians = []
    #create list of datetime deltas:
    datetime_deltas = create_datetime_list(dirs)
    #create subsets for each temporal baseline
    for tb in tbs:
        dt_dirs = delta_files(dirs, datetime_deltas, tb)
        print(f'{tb} day repeat: {len(dt_dirs)}')
        # for each subset, calculate the median coherence image
        coh_list = []
        for d in dt_dirs:
            #ifg = interferogram.Interferogram(os.path.join(os.path.join(path,d)))
            coh_file = os.path.join(os.path.join(path,d),'merged/phsig.cor.geo')
            dataset = rasterio.open(coh_file)
            coh = dataset.read(1)
            metadata = dataset.meta
            coh_list.append(coh)
        coh_ts = np.stack(coh_list, axis=2)
        coh_med = np.median(coh_ts, axis=2)
        coh_medians.append(coh_med)
    return coh_medians, dataset, metadata


#temporal baselines:
#tbs = [6,12,18,24,30,36,42,48,72] # not enough pairs to do all these on the descending track atm.
tbs = [12,24,36,48,60,72]
coh_medians, dataset, metadata = med_coh_ts(dirs, tbs)

#reproject coh_medians:
target_epsg = "EPSG:2056"
target_crs = pyproj.CRS.from_string(target_epsg)
transform, width, height = calculate_default_transform(dataset.crs, target_crs, dataset.width, dataset.height, *dataset.bounds)

# Update the metadata with the new transformation parameters
metadata.update({
    'crs': target_crs,
    'transform': transform,
    'width': width,
    'height': height
})
# Reproject the data from the source to the target coordinate system in memory

coh_med_proj = []
for f in coh_medians:
    reprojected_data = np.zeros((height, width), dtype=dataset.read(1).dtype)
    reproject(
        source=f,
        destination=reprojected_data,
        src_transform=dataset.transform,
        src_crs=dataset.crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear)
    coh_med_proj.append(reprojected_data)

extent = rasterio.transform.array_bounds(height, width, transform)

fig, axs = plt.subplots(2, 3, figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)
for i in range(2):
    for j in range(3):
        index = i * 3 + j
        axs[i, j].imshow(coh_med_proj[index][:-40,:],
        extent=extent)
        axs[i, j].set_title(f'{tbs[index]} days')
        
fig.suptitle('Median coherence') 
fig.tight_layout()
fig.show()
#fig.savefig('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/coherence_overview.png')

fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.imshow(coh_med_proj[0][:-40,:])
ax.set_title('6-day coherence')
fig.show()
#fig.savefig('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/6day-coherence.png')

#Mask areas with coherence <0.35 in 6-day tb and apply to all arrays
mask_35 = coh_med_proj[0] >= 0.35
#Test also coherence of 0.3
mask_30 = coh_med_proj[0] >= 0.30
#Check the difference
mask_diff = mask_35 ^ mask_30

#Difference between mask with coherence 0.3 vs. 0.35
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(mask_30[:-40,:], cmap='binary', extent=extent)
axs[0].set_title('Mask 0.3')
axs[1].imshow(mask_35[:-40,:], cmap='binary', extent=extent)
axs[1].set_title('Mask 0.35')
axs[2].imshow(mask_diff[:-40,:], cmap='binary', extent=extent)
axs[2].set_title('Mask Difference')

fig.tight_layout()#
fig.show()
#fig.savefig('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/coherence_mask.png')


#Apply mask
coh_masked = [arr.copy() for arr in coh_med_proj]
for i in range(len(coh_med_proj)):
    coh_masked[i][~mask_35] = np.nan

#stack for vectorization
coh_stack = np.dstack(coh_masked)

#For remaining pixels, find first index where coherence is below 0.5
# Create a mask for values lower than 0.5
mask = coh_stack < 0.55
indices = np.argmax(coh_stack < 0.5, axis=2)
#indices[mask.sum(axis=2) == 0] = 8 #for eight classes on current ascending subset
indices[mask.sum(axis=2) == 0] = 5 #for six classes on current descending subset
indices = indices.astype('float')
indices[~mask_35] = np.nan

# Plot the resulting indices
from matplotlib import cm
cmap=cm.get_cmap('viridis')
cmap.set_over('mediumorchid')

plt.imshow(indices, 
           cmap=cmap,
           vmin=0,
           vmax=8, 
           interpolation='nearest')
plt.colorbar()
plt.show()

#Now go from indices to values of the temporal baseline
# Replace values in the 2D array using indices from the 1D array
tb = indices.copy()
mask = ~np.isnan(tb)
tb[mask] = np.asarray(tbs)[indices[mask].astype(int)]

fig, ax = plt.subplots()
tb_plt= ax.imshow(tb[:-40,:], 
           cmap='magma',
           interpolation='nearest',
           extent=extent)
ax.set_title('Coherence decay')
cbar = plt.colorbar(tb_plt)
cbar.set_label('Days')
fig.tight_layout()
fig.show()
#fig.savefig('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/temporal_baselines.png')



#Export to Geotiff

meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': -9999,
    'width': tb.shape[1],
    'height': tb.shape[0],
    'count': 1,
    'crs': 'EPSG:2056',  # You should use the appropriate CRS for your data
    'transform': transform  # Define the transformation
}

with rasterio.open('temporal_baselines_desc139.tif', 'w', **meta) as dst:
    dst.write(tb, 1) 


#1 D example: 
cohs = []
for c in coh_medians:
    #coh = c[1200,1400]
    coh = c[100, 925]
    cohs.append(coh)

x = np.asarray(tbs)
y = np.asarray(cohs)

# Plot the data and the fitted line
plt.scatter(x, y, label='Data')
#plt.plot(x, y_pred, label='Fitted line', color='r')
#plt.title(f'slope error: {slope_error: .3f}, intercept error {intercept_error: .3f}, rmse: {rmse: .3f}')
plt.ylabel('Coherence')
plt.xlabel('Temporal baseline')
plt.title('Coherence decay')
#plt.legend()
#plt.show()
plt.savefig('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/Coherence_decay_timeseries.png')

#plt.plot(tbs, cohs, '.')
#plt.show()

# Synthetic example
# Define the exponential decay function
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Function from Kellndorfer et al:
def coherence_decay(roh_inf, t, tau):
    return (1-roh_inf)*np.exp(-t/tau)+roh_inf

# Using linear function:
def linear_decay(x, m, c):
    return m*x+c

# Generate sample data


# Fit the curve to the data
params, covariance = curve_fit(linear_decay, x, y)

# Extract the slope and intercept
slope, intercept = params

# Extract the standard errors from the covariance matrix
slope_error, intercept_error = np.sqrt(np.diag(covariance))

# Compute the predictions
y_pred = linear_decay(x, slope, intercept)

# Compute the residual (difference between observed and predicted values)
residuals = y - y_pred

# Compute the root mean square error
rmse = np.sqrt(np.mean(residuals**2))


'''


'''
Combining coherence and visibility

Determine threshold for coherence (e.g. 0.5) and set for each pixel
when value drops below this threshold

Assume very slow motion: (half a fringe ~1.5cm?) that could be visible over 
the longest time-period.
Calculate how much of this motion would be visible.

Say coherence drops below threshold after x days:
24 days and 1.5 cm = 0.6mm / day = 219 mm/ year
72 days and 1.5 cm = 0.2mm / day = 76 mm / year


'''