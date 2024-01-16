
from scipy.ndimage import filters
import sys

class MapUtils():
	
	def __init__(self, np_data, side_angle_arcmin):
		self.np_data=np_data
		self.side_angle_arcmin=side_angle_arcmin
	
	def smooth(self,scale_angle_arcmin,kind="gaussian",**kwargs):
	
		"""
		Performs a smoothing operation on the convergence map

		:param scale_angle: size of the smoothing kernel (must have units)
		:type scale_angle: float.

		:param kind: type of smoothing to be performed. Select "gaussian" for regular Gaussian smoothing in real space or "gaussianFFT" if you want the smoothing to be performed via FFTs (advised for large scale_angle)
		:type kind: str.

		:param inplace: if set to True performs the smoothing in place overwriting the old convergence map
		:type inplace: bool.

		:param kwargs: the keyword arguments are passed to the filter function
		:type kwargs: dict.

		:returns: ConvergenceMap instance (or None if inplace is True)

		"""

		if kind=='gaussian':
			smoothing_scale_pixel=scale_angle_arcmin*self.np_data.shape[0]/self.side_angle_arcmin
			smoothed_data = filters.gaussian_filter(self.np_data,smoothing_scale_pixel,**kwargs)

		else:
			sys.exit("Only Gaussian smoothing implemented so far")
		
		return smoothed_data