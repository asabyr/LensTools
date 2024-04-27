import numpy as np
from scipy.ndimage import filters
from scipy import interpolate
import sys
import copy
from ..utils.fft import NUMPYFFTPack
fftengine = NUMPYFFTPack()
import astropy.units as u
import GaussianNoiseGenerator


FWHM_to_sigma=2.0*np.sqrt(2.0*np.log(2.0))
arcmin_to_rad=1.0/60.0*np.pi/180.0

class MapUtils():
    
    def __init__(self, np_data, side_angle_arcmin):
        
        self.np_data=np_data
        self.side_angle_arcmin=side_angle_arcmin
    
    def smooth(self,scale_angle_arcmin_FWHM,kind="gaussian",**kwargs):
    
        """
        Performs a smoothing operation on a numpy map

        :param scale_angle_arcmin_FWHM: FWHM for gaussian smoothing in arcmin
        :type scale_angle_arcmin_FWHM: float.

        :param kind: type of smoothing to be performed. Select "gaussian" for regular Gaussian smoothing in real space
        or "gaussianFFT" if you want the smoothing to be performed via FFTs.
        :type kind: str.

        :param kwargs: the keyword arguments are passed to the filter function
        :type kwargs: dict.

        :returns: smoothed data in a numpy array

        """
        if kind=='gaussian':
			#convert FWHM to sigma
            scale_angle_arcmin=scale_angle_arcmin_FWHM/FWHM_to_sigma
            smoothing_scale_pixel=scale_angle_arcmin*self.np_data.shape[0]/self.side_angle_arcmin   
            smoothed_data = filters.gaussian_filter(self.np_data,smoothing_scale_pixel,**kwargs)
            
        elif kind=='gaussianFFT':
            
            scale_angle_arcmin=scale_angle_arcmin_FWHM/FWHM_to_sigma
            smoothing_scale_pixel=scale_angle_arcmin*self.np_data.shape[0]/self.side_angle_arcmin   
            
            lx = fftengine.fftfreq(self.np_data.shape[0])
            ly = fftengine.rfftfreq(self.np_data.shape[1])
            l_squared = lx[:,None]**2 + ly[None,:]**2
            smoothed_data = fftengine.irfft2(np.exp(-0.5*l_squared*(2*np.pi*smoothing_scale_pixel)**2)*fftengine.rfft2(self.np_data))
            
        else:
            
            sys.exit("Only Gaussian smoothing implemented so far")
        
        return smoothed_data
		
	
    def tanh_ell(self, ells, **kwargs):
        """
        This function helps smoothly filter out the low ells.

        :param ells: multipoles over which to apply "tanh"
        :type ells: numpy array of floats or ints

        :param range_to_zero: range of multipoles over which to apply "tanh"
                            (2-element array) 
        :param range_to_zero: numpy array of floats or ints with len=2

        :returns: filter to be applied to ells (numpy array of floats)
        """

        center_ell=np.mean(self.range_to_zero)
        tanh_y=0.5*(np.tanh(self.tanh_steep*(ells-center_ell))+1)
        return tanh_y


    def low_ell_zero(self, ell_map, **kwargs):
        """
        This function helps smoothly filter out the low ells.

        :param ells: map of multipoles
        :type ells: numpy array of floats or ints

        :param range_to_zero: range of multipoles over which to apply "tanh"
                            (2-element array) 
        :param range_to_zero: numpy array of floats or ints with len=2

        :param tanh_x_bound: 
        :type tanh_x_bound: float

        :returns: filter map (numpy array of floats)

        """
        #filter map
        output=np.empty_like(ell_map)
        
        #set to 0 below range_to_zero[0], 
        # to 1 above range_to_zero[-1] and use tanh within specified region

        ind_zero=np.where(ell_map<self.range_to_zero[0])
        ind_one=np.where(ell_map>=self.range_to_zero[-1])
        ind_tanh=np.where((ell_map<self.range_to_zero[-1]) & (ell_map>self.range_to_zero[0]))
        
        
        output[ind_zero[0],ind_zero[1]]=0.0
        output[ind_one[0], ind_one[1]]=1.0
        output[ind_tanh[0],ind_tanh[1]]=self.tanh_ell(ell_map[ind_tanh[0],ind_tanh[1]], **kwargs)
        
        return output


    def filter_low_ell(self, range_to_zero=[80,90], tanh_steep=0.75, return_filter=False, **kwargs):
        
        self.range_to_zero=range_to_zero
        self.tanh_steep=tanh_steep

        side_angle_rad=self.side_angle_arcmin*arcmin_to_rad

        lx = fftengine.fftfreq(self.np_data.shape[0])
        ly = fftengine.rfftfreq(self.np_data.shape[1])

        lx*=(self.np_data.shape[0]/side_angle_rad)*2*np.pi
        ly*=(self.np_data.shape[1]/side_angle_rad)*2*np.pi

        l_map = np.sqrt((lx[:,None]**2 + ly[None,:]**2))
        
        ell_filter=self.low_ell_zero(ell_map=l_map)

        filtered_data=fftengine.irfft2(ell_filter*fftengine.rfft2(self.np_data))
        # filtered_data=fftengine.ifft2(ell_filter*fftengine.fft2(self.np_data))
        
        if return_filter==True:
            return filtered_data, ell_filter, l_map
        else:
            return filtered_data
    
    def low_ell_zero_gauss_smooth(self, ell_map, **kwargs):
        """
        This function helps smoothly filter out the low ells.

        :param ells: map of multipoles
        :type ells: numpy array of floats or ints

        :param range_to_zero: range of multipoles over which to apply "tanh"
                            (2-element array) 
        :param range_to_zero: numpy array of floats or ints with len=2

        :param tanh_x_bound: 
        :type tanh_x_bound: float

        :returns: filter map (numpy array of floats)

        """
        #filter map
        output=np.empty_like(ell_map)
        
        #set to 0 below range_to_zero[0], 
        # gaussian smooth above range_to_zero[-1] and use tanh within specified region

        ind_zero=np.where(ell_map<self.range_to_zero[0])
        ind_gauss=np.where(ell_map>=self.range_to_zero[-1])
        ind_tanh=np.where((ell_map<self.range_to_zero[-1]) & (ell_map>self.range_to_zero[0]))
        
        
        output[ind_zero[0],ind_zero[1]]=0.0
        output[ind_gauss[0], ind_gauss[1]]=np.exp(-0.5*ell_map[ind_gauss[0], ind_gauss[1]]**2.0*self.gauss_smooth_sigma_rad**2.0)
        output[ind_tanh[0],ind_tanh[1]]=self.tanh_ell(ell_map[ind_tanh[0],ind_tanh[1]], **kwargs)
        
        return output

    def filter_low_ell_gauss_smooth(self, gauss_smooth_FWHM_arcmin, range_to_zero=[80,90], tanh_steep=0.75, return_filter=False, **kwargs):
        
        #figure out gauss sigma radians + define some attributes
        gauss_smooth_sigma_arcmin=gauss_smooth_FWHM_arcmin/FWHM_to_sigma
        self.gauss_smooth_sigma_rad=gauss_smooth_sigma_arcmin*arcmin_to_rad       
        self.range_to_zero=range_to_zero
        self.tanh_steep=tanh_steep

        #make a map of physical ells
        side_angle_rad=self.side_angle_arcmin*arcmin_to_rad

        lx = fftengine.fftfreq(self.np_data.shape[0])
        ly = fftengine.rfftfreq(self.np_data.shape[1])

        lx*=(self.np_data.shape[0]/side_angle_rad)*2*np.pi
        ly*=(self.np_data.shape[1]/side_angle_rad)*2*np.pi

        l_map = np.sqrt((lx[:,None]**2 + ly[None,:]**2))

        #apply low + smooth filter & convert to real space
        ell_filter=self.low_ell_zero_gauss_smooth(ell_map=l_map)

        filtered_data=fftengine.irfft2(ell_filter*fftengine.rfft2(self.np_data))
        # filtered_data=fftengine.ifft2(ell_filter*fftengine.fft2(self.np_data))
        
        if return_filter==True:
            return filtered_data, ell_filter, l_map
        else:
            return filtered_data
    
    

