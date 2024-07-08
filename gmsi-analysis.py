import rasterio 
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

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

plt.imshow(gmsi_slmask_mask)
plt.show()