import rasterio
import os
import numpy
import re
from osgeo import gdal
from osgeo import gdal_array
from matplotlib.colors import LinearSegmentedColormap
import insarhelpers
from pyproj import Proj
from datetime import datetime



class Interferogram(object):
    """
    A class to easily plot outputs from ISCE


    Attributes
    ----------
        path : str
            path to ISCE output directory with date in filename
        dates : tuple
            start and enddate of interferogram as datetime objects
        metadata : gdal object
            metadata information for georeferencing etc.
        complex_wrapped : ndarray
            complex wrapped phase + intensity (ISCE output filt_topophase.flat.geo)
        coherence : ndarray
            interferometric coherence (ISCE output phsig.cor.geo)
        los : ndarray
            radar geometry information (los[0] = incidence angle from vertical at
            target; los[1]= azimuth measured from North in anti-clockwise direction)
        phase : ndarray
            wrapped phase from -pi/2 to +pi/2
        projection : str
            proj-string of desired projection. Default is EPSG:4326 (projection=None)
        extent : list
            x-min, x-max, y-min, y-max coordinates of image extent for plotting
        cmap : matplotlib.colors.LinearSegmentedColormap
            circular colormap for plotting phase

    Methods
    -------
        mask_phase(coh_th)
            masks the phase based on a given coherence threshold
        get_amplitude()
            calculate amplitude for plotting intensity image
        load_unwrapped()
            load unwrapped phase (ISCE output filt_topophase.unw.geo)
        get_los_vec(l)
            generate line of sight and flight direction vectors for plotting

    """
    def __init__(self, path, projection=None):
        self.path = os.path.join(path,'merged')
        self.dates = parse_dates(path)
        self.metadata = load_metadata(self.path, 'filt_topophase.flat.geo')
        self.complex_wrapped = load_rdrdata(self.path, 'filt_topophase.flat.geo')
        self.coherence = load_rdrdata(self.path, 'phsig.cor.geo')
        self.los = load_rdrdata(self.path, 'los.rdr.geo')
        self.phase = numpy.arctan2(self.complex_wrapped.imag,self.complex_wrapped.real)
        self.projection = projection
        self.extent = get_extent(self.metadata, self.projection)
        self.cmap = LinearSegmentedColormap.from_list('mycmap',['cyan', 'magenta', 'yellow', 'cyan'])

    def mask_phase(self, coh_th):
        """Mask the phase based on a coherence threshold

        Parameters
        ----------
            coh_th : float
                threshold used for masking (between 0 and 1)

        Returns
        -------
            masked_phase : ndarray
        """
        masked_phase = numpy.where(self.coherence > coh_th, self.phase, numpy.nan)
        return masked_phase

    def get_amplitude(self):
        """Calculate intensity image (dB)

        Returns
        -------
            amplitude : ndarray
        """
        #amp = 20*numpy.log(numpy.sqrt(self.complex_wrapped.real**2 + self.complex_wrapped.imag**2))
        amp = numpy.sqrt(self.complex_wrapped.real**2 + self.complex_wrapped.imag**2)
        return amp

    def load_unwrapped(self):
        """ Load unwrapped phase (filt_topophase.unw.geo)

        Returns
        ------
            unwrapped phase : ndarray
        """
        unw_ifg = load_rdrdata(self.path, 'filt_topophase.unw.geo')
        unw_ph = unw_ifg[1]
        return unw_ph

    def get_los_vec(self, l, factor=1.5):
        """ Generate dx and dy distances for line of sight and flight direction
            arrows (for plotting purposes)

        Parameters
        ----------
            l : float
                length of line of sight vector (in units of projection)
            factor : int
                factor that flight direction vector is longer than los vector.
                default = 1.5

        Returns
        ------
        los_dx, los_dy, fd_dx, fd_dy : floats
        """
        los_dx, los_dy = insarhelpers.get_los(self.los[1],l)
        fd_dy, fd_dx = insarhelpers.get_los(self.los[1],l*factor)
        return los_dx, los_dy, -fd_dx, fd_dy


#--------------------helpers---------------------------------------------------
#parse interferogram dates (e.g., for plots)
def parse_dates(path):
    """ Parse dates from filenames

    Returns
    -------
    datetimes : tuple of datetime objects
    """
    d = os.path.basename(path)
    dates = d.split('_')
    datetimes = [datetime.strptime(d, '%Y%m%d') for d in dates]
    return datetimes


#load radar data
def load_rdrdata(path, fn):
    """ Load data from path

    Returns
    -------
    data : ndarray
    """
    path_to_fn = os.path.join(path,fn)
    file = gdal_array.LoadFile(path_to_fn)
    return file

#load radar metadata
def load_metadata(path, fn):
    """ Load metadata from path

    Returns
    -------
    metadata : gdal object
    """
    path_to_fn = os.path.join(path,fn)
    metadata=gdal.Open(os.path.join(path,fn))
    return metadata

#create list with coordinates for plotting by extent
def get_extent(ds, projection=None):
    """ Get extent of radar image and project if projection is provided

    Returns
    -------
    extent : list
    """

    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ext = insarhelpers.GetExtent(gt, cols, rows)
    if projection is not None:
        myProj = Proj(projection)
        proj_ext = [myProj(c[0], c[1]) for c in ext]
        radar_extent = [proj_ext[0][0], proj_ext[2][0], proj_ext[1][1], proj_ext[0][1]]
    else:
        radar_extent = [ext[0][0], ext[2][0], ext[1][1], ext[0][1]]
    return radar_extent
