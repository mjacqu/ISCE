import interferogram
import os
import matplotlib.pyplot as plt
from pyproj import Proj
from datetime import datetime


ifg = interferogram.Interferogram(
    path ='/Volumes/Science/SpitzerStein/asc_88/20180628_20180716'
    #path = '/scratch_net/vierzack05_third/mylenej/radar/Chamoli/results/des63/20200803_20200815',
    #projection = "+proj=utm +zone=44 +ellps=WGS84 +datum=WGS84 +units=m +nodefs"
)


#masked_phase = ifg.mask_phase(0.28) #mask phase with a coherence threshold of 0.25
#amp = ifg.get_amplitude() #calculate intensity of backscatter (dB)

los_dx, los_dy, fd_dx, fd_dy = ifg.get_los_vec(1000) #for plotting los arrows

f, ax = plt.subplots(figsize=(6,8))
im = ax.imshow(ifg.phase, cmap = ifg.cmap, extent = ifg.extent)
#ax.arrow(374000, 3358000, los_dx, los_dy, head_width = 100, fc = 'k')
#ax.arrow(374000, 3358000, fd_dx, fd_dy, head_width = 100, fc = 'k')
ax.set_title(f"Wrapped interferogram {os.linesep}"
    f"{datetime.strftime(ifg.dates[0], format='%Y/%m/%d')} to "
    f"{datetime.strftime(ifg.dates[1], format='%Y/%m/%d')}")
ax.set_xlabel('UTM Easting (km)')
ax.set_ylabel('UTM Northing (km)')
f.colorbar(im)
f.show()

f, ax = plt.subplots(figsize=(6,8))
im = ax.imshow(amp, cmap = 'Greys', extent = ifg.extent)
ax.set_title('Backscatter amplitude (dB)')
ax.set_xlabel('UTM Easting (km)')
ax.set_ylabel('UTM Northing (km)')
f.colorbar(im)
f.show()

meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': -9999,
    'width': ifg.phase.shape[1],
    'height': ifg.phase.shape[0],
    'count': 1,
    'crs': 'EPSG:4326',  # You should use the appropriate CRS for your data
    'transform': transform  # Define the transformation
}

with rasterio.open('wrapped_phase.tif', 'w', **meta) as dst:
    dst.write(ifg.phase, 1) 

