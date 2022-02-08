import sys
sys.path.append('../MudCreek')
import los_projection as lp
import numpy as np
from osgeo import gdal
import scipy.signal
import richdem
from mpmath import *
import rasterio
import earthpy
import earthpy.spatial as es
import earthpy.plot as ep
import matplotlib.pyplot as plt


'''
Outline:
1. Create shadow and layover mask following Rees 2000
2. Apply compression-factor analysis following Cigna 2014 et al.,

Define geometric parameters:

β: Terrain slope relative to horizontal
α: Terrain aspect clockwise from NORTH
γ: Satellite heading angle clockwise from NORTH
θ: Angle between radar beam and vertical (look incidence angle)
_: Angle between radar beam and terrain normal (LOCAL incidence angle)
_: Look direction clockwise from NORTH (normal to γ)


'''
#Spitzer Stein DEM (note: needs to be in meters for slope calculation to work)
path = '/Users/mistral/Documents/ETHZ/Science/SpitzerStein/swissALTI3D_2019_Doldenstock_2m_lv95.tif'

# Load DEM
dem = richdem.LoadGDAL(path)

# Load LOS
# note: Azimuth from los.rdr.geo == LOOK direction!
inc, azi, los = lp.read_los('./ascending_los/los.rdr.geo') #test file from Columbia Glacier not actually fitting for Spitzer Stein

# calculate slope and aspect of DEM
slope = richdem.TerrainAttribute(dem, attrib='slope_riserun')
aspect = richdem.TerrainAttribute(dem, attrib='aspect')

# Shadowing where dh/dx <= cot(inc)
look_angle = np.radians(inc.mean())
shadow_threshold = np.degrees(1/np.tan(look_angle)) #all areas where slope <= threshold
layover_threshold = np.degrees(np.tan(look_angle)) #all areas where slope > threshold

#Aspects at which shadowing/layover is possile
#Assumption Nr. 1: Shadowing happens in area +/- 90 degrees from look direction
#Assumption Nr. 2: Layover happens in area +/- 90 degress opposite from look direction
look_direction = azi.mean()*-1

def get_sl_interval(look_direction):
    s_interval = [look_direction+90, look_direction-90]
    l_interval = [look_direction-90, look_direction+90]
    return s_interval, l_interval


s_interval, l_interval = get_sl_interval(look_direction)

#Hillshade for plotting results onto
# Open the DEM with Rasterio
with rasterio.open(path) as src:
    dtm = src.read(1)
    # Set masked values to np.nan
    dtm[dtm < 0] = np.nan

hillshade = es.hillshade(dtm)
# Plot the data
ep.plot_bands(
    hillshade,
    cbar=False,
    figsize=(10, 6),
)
plt.show()
