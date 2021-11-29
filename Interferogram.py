import numpy as np
import rasterio
import matplotlib.pyplot as plt

class Interferogram(object):
    """
    Attributes:
        startdate (datetime): Date of first image acquisition
        enddate (datetime): Date of second image acquisition
        extent (array): #format to match imshow extent
        wrapped (array): wrapped phase
        unwrapped (array): unwrapped phase
        coherence (array): radar coherence
        los (array): radar view direction and incidence angle
        
    """
