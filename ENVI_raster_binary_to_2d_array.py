from __future__ import division
import os
import sys

from osgeo import gdal, gdalconst 
from osgeo.gdalconst import * 



def ENVI_raster_binary_to_2d_array(file_name):
	'''
	Converts a binary file of ENVI type to a numpy array.
	Lack of an ENVI .hdr file will cause this to crash.
	'''
	driver = gdal.GetDriverByName('ENVI') 

	driver.Register()

	inDs = gdal.Open(file_name, GA_ReadOnly)
	
	if inDs is None:
		print "Couldn't open this file: " + file_name
		print '\nPerhaps you need an ENVI .hdr file? '			
		sys.exit("Try again!")
	else:
		print "%s opened successfully" %file_name
			
		print '~~~~~~~~~~~~~~'
		print 'Get image size'
		print '~~~~~~~~~~~~~~'
		cols = inDs.RasterXSize
		rows = inDs.RasterYSize
		bands = inDs.RasterCount
	
		print "columns: %i" %cols
		print "rows: %i" %rows
		print "bands: %i" %bands
	
		print '~~~~~~~~~~~~~~'
		print 'Get georeference information'
		print '~~~~~~~~~~~~~~'
		geotransform = inDs.GetGeoTransform()
		originX = geotransform[0]
		originY = geotransform[3]
		pixelWidth = geotransform[1]
		pixelHeight = geotransform[5]
	
		print "origin x: %i" %originX
		print "origin y: %i" %originY
		print "width: %2.2f" %pixelWidth
		print "height: %2.2f" %pixelHeight
	
		# Set pixel offset.....
		print '~~~~~~~~~~~~~~' 
		print 'Convert image to 2D array'
		print '~~~~~~~~~~~~~~'
		band = inDs.GetRasterBand(1)
		image_array = band.ReadAsArray(0, 0, cols, rows)
		image_array_name = file_name
		print type(image_array)
		print image_array.shape
		
		return image_array, pixelWidth,pixelHeight # (geotransform, inDs)


# The function can be called as follows:
# image_array, post, envidata =  ENVI_raster_binary_to_2d_array(file_name) 
#
# Notes:
# Notice the tuple (geotransform, inDs) - this contains all of your map information 
# (xy tie points, postings and coordinate system information)
# pixelWidth is assumed to be the same as pixelHeight in the above example, therefore 
# representing the surface posting - if this is not the case for your data then you 
# must change the returns to suit