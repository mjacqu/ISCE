import sys
import os
import numpy as np
sys.path.append('./')
import interferogram
import matplotlib.pyplot as plt
from matplotlib import colors
import rasterio
import earthpy.spatial as es


fp ='/scratch-third/mylenej/radar/SpitzerStein/results/asc_88'
fls = os.listdir(fp)

coh_list = []
for f in fls:
    ifg =  interferogram.Interferogram(
        path=os.path.join(fp,f)
    )
    coh = ifg.coherence
    coh_list.append(coh)

coh_ts = np.stack(coh_list, axis=2)

coh_med = np.median(coh_ts, axis=2)
plt.imshow(coh_med)
plt.show()

coh_var = np.var(coh_ts, axis=2)
scaled_var = coh_var/np.max(coh_var)
plt.imshow(scaled_var)
plt.colorbar()
plt.show()

#usefulness scale
bad = (coh_med<0.35) & (scaled_var<0.1) # areas with low average coherence and low variability: these areas are always bad.
ok = (coh_med> 0.35) & (scaled_var>0.5) # areas with high coherence but high variabiliy. You never know what you will get.
great = (coh_med>0.7) & (scaled_var<0.5) # areas with very high average coherence and realtively low variability. These areas are mostly reliable

low=np.ones(coh_med.shape)
low[~bad]=np.nan
mid=np.ones(coh_med.shape)
mid[~ok]=np.nan
high=np.ones(coh_med.shape)
high[~great]=np.nan

dempath = path = '//scratch-third/mylenej/radar/SpitzerStein/data/dem/kandersteg10m_wgs84'
with rasterio.open(dempath) as src:
    dtm = src.read(1)
    # Set masked values to np.nan
    dtm[dtm < 0] = np.nan

hillshade = es.hillshade(dtm)

plt.imshow(hillshade, cmap='Greys')
plt.imshow(low, alpha=0.8, cmap='Reds_r')
plt.imshow(mid, alpha=0.8, cmap='Oranges_r')
plt.imshow(high, alpha=0.8, cmap='Blues_r')
plt.show()
