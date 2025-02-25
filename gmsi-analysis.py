import rasterio 
import numpy as np
import dask.array as da
import glob
import os
import matplotlib.pyplot as plt

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
   
# deal with zeros and inf in scale factor maps:

def preprocess_scalefactors(scale_f):
    sf_zeros = da.where(scale_f == 0, np.nan, scale_f)
    sf_neg_inf = da.where(scale_f < -1000, np.nan, sf_zeros)
    sf_array = da.where(sf_zeros>10e3, np.nan, sf_neg_inf)
    return sf_array

################## visible terrain fraction analysis (gmsi-v1) ########################

dem = read_raster_as_dask_array('/Volumes/Science/CCAMM/CH-DEM/dhm200_gmsi_extent.tif')

# load all visibility files
vis_path = '/Volumes/Science/CCAMM/visibility'
vis_files = glob.glob(os.path.join(vis_path, '*/*.norm_scale_factor_masked.tif'))
ls_files = glob.glob(os.path.join(vis_path, '*/*.lsmap.tif'))

gmsi_path = '/Volumes/Science/CCAMM/gmsi-v1'
gmsi_files = glob.glob(os.path.join(gmsi_path, 'gmsi_norm_*.tif'))
tracks = ['A-15', 'A-88', 'D-66', 'D-139', 'D168']

# load visbility
visibility = [read_raster_as_dask_array(f) for f in vis_files]
sf_arrays = [preprocess_scalefactors(v) for v in visibility]

# load gmsi orbit tracks:
gmsi_tracks = [read_raster_as_dask_array(f) for f in gmsi_files]

# load ls maps to combine into map of data coverage
ls_maps = [read_raster_as_dask_array(f) for f in ls_files]
mask15 = ls_maps[0]>0
mask88 = ls_maps[1]>0
mask66 = ls_maps[2]>0
mask139 = ls_maps[3]>0
mask168 = ls_maps[4]>0
# Combine the masks using a logical OR operation
combined_mask = mask15 | mask88 | mask66 | mask139 | mask168

# Create the final raster with values set to 1 where the combined mask is True
data_coverage = da.where(combined_mask, 1, 0)

#load gmsi
gmsi = read_raster_as_dask_array('/Volumes/Science/CCAMM/gmsi-v1/gmsi_composite.tif')
############# set thresholds ###################

vis_threshold = 0.25
gmsi_good = 0.4
gmsi_excellent = 0.6

################ alpine area ###############
elev_threshold = 1500
alpine_mask = (dem > elev_threshold) & (data_coverage == 1) #anything above subalpine
pixel_area = 10*10
alpine_area = (alpine_mask.sum()*pixel_area).compute() / 1e6 #roughly 28 500 km2

################################ first look at full picture ####################

good_gmsi = (gmsi > gmsi_good) & (alpine_mask == 1)
excellent_gmsi = (gmsi > gmsi_excellent) & (alpine_mask == 1)

good_gmsi_area = (good_gmsi.sum()*pixel_area).compute()/1e6 
excellent_gmsi_area = (excellent_gmsi.sum()*pixel_area).compute()/1e6

frac_good = good_gmsi_area/alpine_area
frac_excellent = excellent_gmsi_area/alpine_area

############ calculate visibility above an elevation ###########
def per_orbit_stats(images, dem, qual_threshold, elev_threshold, orbit_masks, tracks):
    elevation_masks = [(dem > elev_threshold) & (om == 1) for om in orbit_masks]
    area_above = [(img > qual_threshold) & (em == 1) for img,em in zip(images, elevation_masks)]
    #area_all = [(img > 0) & (dem > elev_threshold) for img in images]

    area_above_km2 = [(vm.sum()*pixel_area).compute()/1e6 for vm in area_above]
    orbit_area_above_elev = [(vm.sum()*pixel_area).compute()/1e6 for vm in elevation_masks]

    visible_area_per_orbit = [ok/all for ok,all in zip(area_above_km2, orbit_area_above_elev)]

    #create results dictionary
    stats = {tracks[i]: visible_area_per_orbit[i] for i in range(len(tracks))} 

    print(f'Percent area with visibility/gmsi > {qual_threshold} above {elev_threshold} per orbit (shadow and layover masked): {stats}')
    print(orbit_area_above_elev)
    print(area_above_km2)
    return stats, elevation_masks

orbit_masks = [mask15, mask88, mask66, mask139, mask168]


# good > 1500
gmsi_stats = per_orbit_stats(gmsi_tracks, dem, gmsi_good, elev_threshold, orbit_masks, tracks)
# excellent > 1500
gmsi_stats = per_orbit_stats(gmsi_tracks, dem, gmsi_excellent, elev_threshold, orbit_masks, tracks)

# good > 3000
gmsi_stats = per_orbit_stats(gmsi_tracks, dem, gmsi_good, 3000, orbit_masks, tracks)
# excellent > 3000
gmsi_stats = per_orbit_stats(gmsi_tracks, dem, gmsi_excellent, 3000, orbit_masks, tracks)

vis_stats = per_orbit_stats(sf_arrays, dem, vis_threshold, elev_threshold, orbit_masks, tracks)



####################### print results #########################

print(f'Percent area with visibility > {vis_threshold} above {elev_threshold} per orbit (shadow and layover masked): {vis_stats}')
print(f'Percent area with gmsi > {gmsi_good} above {elev_threshold} per orbit (shadow and layover masked): {gmsi_stats}')
print(f'Fraction good (gmsi-all-CH): {frac_good}')
print(f'Fraction excellent (gmsi-all-CH): {frac_excellent}')





########################################################################################
################### visible terrain fraction analysis (Kandersteg test version) ##################
########################################################################################

#import deformation fraction
with rasterio.open('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/def_frac_ascending88.tif') as def_frac:
    def_frac_asc = def_frac.read(1)

#import mask
with rasterio.open('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/TR88A_clipped_Kandersteg.tif') as mask:
    slmask = mask.read(1)

#import gmsi:
with rasterio.open('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/gmsi_v1.tif') as gmsi:
    gmsi_asc = gmsi.read(1)

#upsample lsmask to grid of def_frac:
slmask_upsampled = resize(slmask, def_frac_asc.shape, order=0)

#create boolean masking:
slmask_upsampled = slmask_upsampled.astype(float)
slmask_upsampled[slmask_upsampled > 249] = np.nan
slmask_bool = np.isnan(slmask_upsampled)

# in def_frac, mask out shadow and layover areas
def_frac_slmask = def_frac_asc.copy()
def_frac_slmask[slmask_bool] = np.nan


#Total area
tot_area = (def_frac_asc.shape[0]*10*def_frac_asc.shape[1]*10)/1000000 #km2

# Area with def_frac > 0.5
def_frac_mask = def_frac_asc > 0.5
cells_above = np.sum(def_frac_mask)
good_visibility = (cells_above * 100) / 1000000 #km2

# Area with def_frac > 0.5 and shadow and layover masked
def_frac_slmask_mask = def_frac_slmask > 0.5
cells_above = np.sum(def_frac_slmask_mask)
good_visibility_slmask = (cells_above * 100) / 1000000 #km2

# Taking into account coherence:
# Upsample shadow mask to gmsi grid:
slmask_gmsi = resize(slmask, gmsi_asc.shape, order=0)

#create boolean masking:
slmask_gmsi = slmask_gmsi.astype(float)
slmask_gmsi[slmask_gmsi > 249] = np.nan
slmask_gmsi_bool = np.isnan(slmask_gmsi)

# in gmsi, mask out shadow and layover areas
gmsi_slmask = gmsi_asc.copy()
gmsi_slmask[slmask_gmsi_bool] = np.nan

gmsi_slmask_mask = gmsi_slmask > 0.5
cells_above_gmsi = np.sum(gmsi_slmask_mask)
good_gmsi = (cells_above_gmsi * 100) / 1000000 #km2

plt.imshow(gmsi_slmask_mask, cmap='Reds', vmin=0, vmax=1)
plt.show()