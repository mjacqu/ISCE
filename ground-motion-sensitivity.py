import rasterio 
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap

#import necessary files:
with rasterio.open('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/def_frac_ascending88.tif') as src_av:
    asc_vis = src_av.read(1)
    av_tf = src_av.transform

with rasterio.open('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/temporal_baselines_asc88.tif') as src_ad:
    asc_decay = src_ad.read(1)
    ad_tf = src_ad.transform

with rasterio.open('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/def_frac_descending139.tif') as src_dv:
    desc_vis = src_dv.read(1)
    dv_tf = src_dv.transform

with rasterio.open('/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/temporal_baselines_desc139.tif') as src_dd:
    desc_decay = src_dd.read(1)
    dd_tf = src_dd.transform

#reproject second dataset to match the first:
def reproject_onto(src1, src2, data1, data2, transf1, transf2):
    data2_reprojected = np.zeros_like(data1)
    reproject(
        source=data2,
        destination=data2_reprojected,
        src_transform=transf2,
        src_crs=src2.crs,
        dst_transform=transf1,
        dst_crs=src1.crs,
        resampling=Resampling.bilinear
    )
    return data2_reprojected, transf1


asc_vis_reproj, asc_vis_tf = reproject_onto(src_ad, src_av, asc_decay, asc_vis, ad_tf, av_tf)
desc_vis_reproj, desc_vis_tf = reproject_onto(src_dd, src_dv, desc_decay, desc_vis, dd_tf, dv_tf)

#calculate snr
snr_asc = (asc_vis_reproj * asc_decay)
snr_desc = (desc_vis_reproj * desc_decay)

#normalize snr
snr_asc_norm = snr_asc / np.nanmax(snr_asc)
snr_desc_norm = snr_desc / np.nanmax(snr_desc)

f, axs = plt.subplots(1,2)
axs[0].imshow(snr_asc_norm, cmap='Reds')
axs[1].imshow(snr_desc_norm, cmap='Reds')
f.show()

#create maximum value composite
snr_composite = np.where(np.isnan(snr_asc_norm), snr_desc_norm, np.maximum(snr_desc_norm, snr_asc_norm))

pref_orbit = np.where(snr_desc_norm > snr_asc_norm, 2.0, 1.0)
pref_orbit[np.isnan(snr_composite)] = np.nan

f, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(8, 6))
snr = axs[0,0].imshow(snr_asc_norm, cmap='Reds', vmin=0, vmax=1)
axs[0,0].set_title('Ascending suitability')
axs[0,1].imshow(snr_desc_norm, cmap='Reds', vmin=0, vmax=1)
axs[0,1].set_title('Descending suitability')
axs[1,0].imshow(snr_composite, cmap='Reds', vmin=0, vmax=1)
axs[1,0].set_title('Combined visibility')
axs[1,1].imshow(pref_orbit, cmap='bwr')
axs[1,1].set_title('Preferred orbit')
f.tight_layout()
f.show()

#apply water mask

# export to 4-channel geotiff
meta = {
    'dtype': 'float32',
    'nodata': -9999,
    'width': snr_composite.shape[1],
    'height': snr_composite.shape[0],
    'count': 4,
    'crs': 'EPSG:2056',  # You should use the appropriate CRS for your data
    'transform': asc_vis_tf  # Define the transformation
}


with rasterio.open(
    '/Users/mistral/Documents/ETHZ/Science/CCAMM/InSAR/Kandersteg/gmsi_v1.tif',
    'w', 
    driver='GTiff',
    **meta
) as dst:
    dst.write(snr_asc_norm, 1) 
    dst.write(snr_desc_norm, 2)
    dst.write(snr_composite, 3)
    dst.write(pref_orbit, 4)
