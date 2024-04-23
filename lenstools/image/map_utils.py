import numpy as np
from scipy.ndimage import filters
import sys
import copy

class MapUtils():
    
    def __init__(self, np_data, side_angle_arcmin):
        self.np_data=np_data
        self.side_angle_arcmin=side_angle_arcmin
    
    def smooth(self,scale_angle_arcmin_FWHM,kind="gaussian",**kwargs):
    
        """
        Performs a smoothing operation on the convergence map

        :param scale_angle_arcmin_FWHM: FWHM for gaussian smoothing in arcmin
        :type scale_angle_arcmin_FWHM: float.

        :param kind: type of smoothing to be performed. Select "gaussian" for regular Gaussian smoothing in real space
        :type kind: str.

        :param kwargs: the keyword arguments are passed to the filter function
        :type kwargs: dict.

        :returns: smoothed data in a numpy array

        """

        if kind=='gaussian':
            
            FWHM_to_sigma=2.0*np.sqrt(2.0*np.log(2.0))
            scale_angle_arcmin=scale_angle_arcmin_FWHM/FWHM_to_sigma

            smoothing_scale_pixel=scale_angle_arcmin*self.np_data.shape[0]/self.side_angle_arcmin   
            smoothed_data = filters.gaussian_filter(self.np_data,smoothing_scale_pixel,**kwargs)

        else:
            sys.exit("Only Gaussian smoothing implemented so far")
        
        return smoothed_data

