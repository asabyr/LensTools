"""

.. module:: shear 
	:platform: Unix
	:synopsis: This module implements a set of operations which are usually performed on weak lensing shear maps


.. moduleauthor:: Andrea Petri <apetri@phys.columbia.edu>


"""

from __future__ import division

from external import _topology

import numpy as np
from astropy.io import fits

##########################################
#####Default Fits loader##################
##########################################
def load_fits_default(*args):
	"""
	This is the default fits file loader, it assumes that the two components of the shear are stored in two different image FITS files, which have an ANGLE keyword in the header

	:param gamma_file: Name of the FITS file that contains the shear map
	:type gamma1_file: str.

	:returns: tuple -- (angle,ndarray -- gamma; gamma[0] is the gamma1 map, gamma[1] is the gamma2 map)

	:raises: IOError if the FITS files cannot be opened or do not exist

	"""

	#Open the files
	gamma_file = fits.open(args[0])

	#Read the ANGLE keyword from the header
	angle = gamma_file[0].header["ANGLE"]

	#Create the array with the shear map
	gamma = gamma_file[0].data.astype(np.float)

	#Close files and return
	gamma_file.close()

	return angle,gamma


##########################################
########ShearMap class####################
##########################################

class ShearMap(object):

	"""
	A class that handles 2D shear maps and allows to perform a set of operations on them

	:param loader: FITS file loading utility, must match the signature and return type of load_fits_default
	:type loader: keyword argument, function

	:param args: positional arguments must be the exact same as the ones that loader takes

	>>> from shear import ShearMap
	>>> test = ShearMap("shear.fit",loader=load_fits_default)
	>>>test.side_angle
	1.95
	>>>test.gamma
	#The actual map values

	"""

	def __init__(self,*args,**kwargs):

		if not("loader" in kwargs.keys()):
			loader = load_fits_default
		else:
			loader = kwargs["loader"]

		self.side_angle,self.gamma = loader(*args)

	def decompose(self,l_edges,keep_fourier=False):

		"""
		Decomposes the shear map into its E and B modes components and returns the respective power spectal densities at the specified multipole moments

		:param l_edges: Multipole bin edges
		:type l_edges: array

		:param keep_fourier: If set to True, holds the Fourier transforms of the E and B mode maps into the E and B attributes of the ShearMap instance
		:type keep_fourier: bool. 

		:returns: :returns: tuple -- (l -- array,P_EE,P_BB,P_EB -- arrays) = (multipole moments, EE,BB power spectra and EB cross power)

		"""

		#Perform Fourier transforms
		ft_gamma1 = np.fft.rfft2(self.gamma[0])
		ft_gamma2 = np.fft.rfft2(self.gamma[1])

		#Compute frequencies
		lx = np.fft.fftfreq(ft_gamma1.shape[0])
		ly = np.fft.rfftfreq(ft_gamma1.shape[0])

		#Safety check
		assert len(lx)==ft_gamma1.shape[0]
		assert len(ly)==ft_gamma1.shape[1]

		#Compute sines and cosines of rotation angles
		l_squared = lx[:,np.newaxis]**2 + ly[np.newaxis,:]**2
		l_squared[0,0] = 1.0

		sin_2_phi = 2.0 * lx[:,np.newaxis] * ly[np.newaxis,:] / l_squared
		cos_2_phi = (lx[:,np.newaxis]**2 - ly[np.newaxis,:]**2) / l_squared

		#Compute E and B components
		ft_E = cos_2_phi * ft_gamma1 + sin_2_phi * ft_gamma2
		ft_B = -1.0 * sin_2_phi * ft_gamma1 + cos_2_phi * ft_gamma2

		ft_E[0,0] = 0.0
		ft_B[0,0] = 0.0

		assert ft_E.shape == ft_B.shape
		assert ft_E.shape == ft_gamma1.shape

		#Compute and return power spectra
		l = 0.5*(l_edges[:-1] + l_edges[1:])
		P_EE = _topology.rfft2_azimuthal(ft_E,ft_E,self.side_angle,l_edges)
		P_BB = _topology.rfft2_azimuthal(ft_B,ft_B,self.side_angle,l_edges)
		P_EB = _topology.rfft2_azimuthal(ft_E,ft_B,self.side_angle,l_edges)

		if keep_fourier:
			self.fourier_E = ft_E
			self.fourier_B = ft_B

		return l,P_EE,P_BB,P_EB

