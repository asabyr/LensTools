from __future__ import division
from lenstools.extern import _topology
from lenstools.utils.fft import NUMPYFFTPack
import numpy as np
import scipy
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import copy

fftengine = NUMPYFFTPack()

class MapStats():

  def __init__(self, np_data):

    self.np_data=np_data


  def powerSpectrum(self,l_edges,side_angle_deg,scale=None):

    """
    Measures the power spectrum of a 2-D numpy array (i.e. any map)
    at the multipole moments specified in the input.

    :param side_angle_deg: the length of one side of the map in degrees
    :param l_edges: Multipole bin edges
    :param scale=None: scaling to apply to the square of the Fourier pixels before harmonic azimuthal averaging.
    Must be a function that takes the array of multipole magnitudes as an input and returns an array of real numbers

    :returns: a dictionary, which containts 'ell', 'power_spectrum', and 'Cl'.

    """
    ell = 0.5*(l_edges[:-1] + l_edges[1:])

    #Calculate the Fourier transform of the map with numpy FFT
    ft_map = fftengine.rfft2(self.np_data)

    #Compute the power spectrum with the C backend implementation
    power_spectrum = _topology.rfft2_azimuthal(ft_map,ft_map,side_angle_deg,l_edges, scale)

    #Output the power spectrum
    ps={}
    ps['ell']=ell
    ps['cl']=power_spectrum
    cl_factor=ell*(ell+1.)/(2.*np.pi)*1.e12
    ps['cl_scaled']=power_spectrum*cl_factor

    return ps

  def gradient(self, return_grad=False):

    """
    Computes the gradient of the map and sets the gradient_x, gradient_y attributes accordingly.
    :param return_grad=False: if set to True, returns a dictionary that contains 'gradient_x' and 'gradient_y'.
    """
    i = None
    j = None

    #Call the C backend
    gradient_x,gradient_y = _topology.gradient(self.np_data,j,i)

    #Return the gradients
    if return_grad==True:

        gradients={}
        gradients['gradient_x']=gradient_x
        gradients['gradient_y']=gradient_y

        return gradients

    #save as attributes
    self.gradient_x = gradient_x
    self.gradient_y = gradient_y

  def hessian(self, return_hess=False):

    """
    Computes the hessian of the map and sets the hessian_xx,hessian_yy,hessian_xy attributes accordingly
    :param return_hess=False: if set to True, returns a dictionary that contains 'hessian_xx', 'hessian_yy', and 'hessian_xy'.

    """
    i = None
    j = None

    #Call the C backend
    hessian_xx,hessian_yy,hessian_xy = _topology.hessian(self.np_data,j,i)

    #Return the hessian
    if return_hess==True:

      hessian={}
      hessian['hessian_xx']=hessian_xx
      hessian['hessian_yy']=hessian_yy
      hessian['hessian_xy']=hessian_xy

      return hessian

    self.hessian_xx = hessian_xx
    self.hessian_yy = hessian_yy
    self.hessian_xy = hessian_xy

  def minkowskiFunctionals(self,thresholds,norm=False):

    """
    Measures the three Minkowski functionals (area, perimeter and genus characteristic)
    of the specified map excursion sets.

    :param thresholds: thresholds that define the excursion sets to consider
    :param norm=False: normalization; if set to a True, interprets the thresholds array
    as units of sigma (the map standard deviation)
    :returns: a dictionary that contains 'midpoints', 'v0', 'v1', and 'v2'.
    """
    midpoints = 0.5 * (thresholds[:-1] + thresholds[1:])
    mask_profile = None

    #Decide if normalize thresholds or not
    if norm==True:
      sigma = self.np_data.std()
    else:
      sigma = 1.0

    #compute hessian and gradient
    self.gradient()
    self.hessian()

    #Compute the Minkowski functionals
    v0,v1,v2 = _topology.minkowski(self.np_data, mask_profile, self.gradient_x, self.gradient_y, self.hessian_xx,self.hessian_yy,self.hessian_xy, thresholds, sigma)

    MF={}
    MF['midpoints']=midpoints
    MF['v0']=v0
    MF['v1']=v1
    MF['v2']=v2

    return MF

  def countPeaks_slow(self, thresholds, offset=1):
    """
    Measures peak counts using loops so quite slow.
    :param thresholds: thresholds for the peak histogram.
    :param offset: how many pixels to ignore at the edges.
    :returns: a dictionary that contains 'peak_heights' (i.e. midpoints of the thresholds),
    'peak_counts', 'peak_values' and 'peak_locs'.
    """
    peaks=[]
    loc=np.array([])

    for i in range(len(self.np_data[:,0])-2*offset):

      for j in range(len(self.np_data[0,:])-2*offset):

        k=i+offset
        c=j+offset

        point=self.np_data[k,c]

        if point>self.np_data[k+1,c] and point>self.np_data[k,c+1]:
          if point>self.np_data[k-1,c] and point>self.np_data[k,c-1]:
            if point>self.np_data[k+1,c+1] and point>self.np_data[k-1,c-1]:
              if point>self.np_data[k+1,c-1] and point>self.np_data[k-1,c+1]:
                peaks.append(point)
                if len(loc)==0:
                  loc=np.array([[k,c]])
                else:
                  loc=np.concatenate((loc,[[k,c]]))

    counts, bin_edges=np.histogram(np.array(peaks), bins=thresholds)
    centers=(bin_edges[:-1] + bin_edges[1:]) / 2

    peaks_dict={}
    peaks_dict['peak_heights']=centers
    peaks_dict['peak_counts']=counts
    peaks_dict['peak_values']=np.array(peaks)
    peaks_dict['peak_locs']=loc

    return peaks_dict


  def peakCount_AP(self, thresholds, norm=False):

    """
    Counts the peaks in the map

    :param thresholds: thresholds extremes that define the binning of the peak histogram
    :type thresholds: array

    :param norm: normalization; if set to a True, interprets the thresholds array as units of sigma (the map standard deviation)
    :type norm: bool.

    :returns: tuple -- (threshold midpoints -- array, differential peak counts at the midpoints -- array)

    :raises: AssertionError if thresholds array is not provided

    >>> test_map = ConvergenceMap.load("map.fit")
    >>> thresholds = np.arange(map.data.min(),map.data.max(),0.05)
    >>> nu,peaks = test_map.peakCount(thresholds)

    """
    midpoints = 0.5 * (thresholds[:-1] + thresholds[1:])
    mask_profile = None
    sigma = 1.0

    peaks={}
    peaks['peak_heights']=midpoints
    peaks['peak_counts']=_topology.peakCount(self.np_data,mask_profile,thresholds,sigma)

    return peaks


  def countPeaks(self,thresholds, offset=1):

    """
    Measures peak counts.
    :param thresholds: thresholds for the peak histogram.
    :param offset: how many pixels to ignore at the edges.
    :returns: a dictionary that contains 'peak_heights' (i.e. midpoints of the thresholds),
    'peak_counts', 'peak_values' and 'peak_locs'.

    faster version based on https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    & https://stackoverflow.com/questions/56148387/finding-image-peaks-using-ndimage-maximum-filter-and-skimage-peak-local-max
    & https://stackoverflow.com/questions/57555171/how-do-you-apply-a-maximum-filter-that-ignores-ties
    """

    #define filter to be over 8 neighboring pixels
    neighborhood = generate_binary_structure(2,2)

    #makes all maxima=1
    #local max is an array of boolean values
    #mode determines what to do with edges but
    #this doesn't matter if you trim the edges

    local_max_all = maximum_filter(self.np_data, footprint=neighborhood, mode='nearest')==self.np_data
    
    #remove repeating values
    kernel=np.ones((3, 3))
    local_max = scipy.ndimage.convolve(np.array(local_max_all, dtype=np.float64), kernel)
    local_max = np.where(local_max>1, 0, local_max)
    local_max = np.array(local_max * local_max_all, dtype='bool')

    #default is to trim the outermost pixels
    if offset>0:

        image_trim=copy.deepcopy(self.np_data)[offset:-offset,offset:-offset]
        local_max_trim=copy.deepcopy(local_max)[offset:-offset,offset:-offset]
        peaks=image_trim[local_max_trim]

        local_max_offset=copy.deepcopy(local_max)
        local_max_offset[:,0]=False
        local_max_offset[0,:]=False
        local_max_offset[-1,:]=False
        local_max_offset[:,-1]=False

        X,Y = np.where(local_max_offset==True)
        loc=np.column_stack((X,Y))

    else:
        peaks=self.np_data[local_max]
        X,Y = np.where(local_max==True)
        loc=np.column_stack((X,Y))

    #bin into histogram
    counts, bin_edges=np.histogram(np.array(peaks), bins=thresholds)
    centers=(bin_edges[:-1] + bin_edges[1:]) / 2

    #output a dictionary
    peaks_dict={}
    peaks_dict['peak_heights']=centers
    peaks_dict['peak_counts']=counts
    peaks_dict['peak_values']=np.array(peaks)
    peaks_dict['peak_locs']=loc

    return peaks_dict


  def pdf(self,thresholds,norm=False):

    """
    Computes the one point probability distribution function of a map.

    :param thresholds: thresholds extremes that define the binning of the pdf
    :type thresholds: array

    :param norm: normalization; if set to a True, interprets the thresholds array as units of sigma (the map standard deviation)
    :type norm: bool.
    """

    midpoints = 0.5 * (thresholds[:-1] + thresholds[1:])

    if norm:
      sigma = self.np_data.std()
    else:
      sigma = 1.0

    #Compute the histogram
    hist, bin_edges = np.histogram(self.np_data, bins=thresholds*sigma, density=True)

    #Return dictionary
    pdf_dict={}
    pdf_dict['pdf_midpoints']=midpoints
    pdf_dict['pdf_values']=hist*sigma

    return pdf_dict

  def moments(self,connected=False,dimensionless=False):
    """
    Measures the first nine moments of the convergence map 
    (two quadratic, three cubic and four quartic)

    :param connected: if set to True returns only the connected part of the moments
    :type connected: bool.

    :param dimensionless: if set to True returns the dimensionless moments, normalized by the appropriate powers of the variance
    :type dimensionless: bool. 

    :returns: array -- (sigma0,sigma1,S0,S1,S2,K0,K1,K2,K3)

    >>> test_map = ConvergenceMap.load("map.fit")
    >>> var0,var1,sk0,sk1,sk2,kur0,kur1,kur2,kur3 = test_map.moments()
    >>> sk0,sk1,sk2 = test_map.moments(dimensionless=True)[2:5]
    >>> kur0,kur1,kur2,kur3 = test_map.moments(connected=True,dimensionless=True)[5:]

    """

    #compute hessian and gradient
    self.gradient()
    self.hessian()
    
    #Quadratic moments
    sigma0 = self.np_data.std()
    sigma1 = np.sqrt((self.gradient_x**2 + self.gradient_y**2).mean())

    #Cubic moments
    S0 = (self.np_data**3).mean()
    S1 = ((self.np_data**2)*(self.hessian_xx + self.hessian_yy)).mean()
    S2 = ((self.gradient_x**2 + self.gradient_y**2)*(self.hessian_xx + self.hessian_yy)).mean()

    #Quartic moments
    K0 = (self.np_data**4).mean()
    K1 = ((self.np_data**3) * (self.hessian_xx + self.hessian_yy)).mean()
    K2 = ((self.np_data) * (self.gradient_x**2 + self.gradient_y**2) * (self.hessian_xx + self.hessian_yy)).mean()
    K3 = ((self.gradient_x**2 + self.gradient_y**2)**2).mean()

    #Compute connected moments (only quartic affected)
    if connected:
      K0 -= 3 * sigma0**4
      K1 += 3 * sigma0**2 * sigma1**2
      K2 += sigma1**4
      K3 -= 2 * sigma1**4

    
    #Normalize moments to make them dimensionless
    if dimensionless:
      S0 /= sigma0**3
      S1 /= (sigma0 * sigma1**2)
      S2 *= (sigma0 / sigma1**4)

      K0 /= sigma0**4
      K1 /= (sigma0**2 * sigma1**2)
      K2 /= sigma1**4
      K3 /= sigma1**4

      sigma0 /= sigma0
      sigma1 /= sigma1

    #Return the array
    return np.array([sigma0,sigma1,S0,S1,S2,K0,K1,K2,K3])


  def countMinima(self,thresholds, offset=1):
    """
    Measures minima counts.
    """
    #define filter to be over 8 neighboring pixels
    neighborhood = generate_binary_structure(2,2)

    #makes all maxima=1
    #local max is an array of boolean values
    #mode determines what to do with edges but
    #this doesn't matter if you trim the edges

    local_min_all = minimum_filter(self.np_data, footprint=neighborhood, mode='nearest')==self.np_data
    
    #remove repeating values
    kernel=np.ones((3, 3))
    local_min = scipy.ndimage.convolve(np.array(local_min_all, dtype=np.float64), kernel)
    local_min = np.where(local_min>1, 0, local_min)
    local_min = np.array(local_min * local_min_all, dtype='bool')

    #default is to trim the outermost pixels
    if offset>0:

        image_trim=copy.deepcopy(self.np_data)[offset:-offset,offset:-offset]
        local_min_trim=copy.deepcopy(local_min)[offset:-offset,offset:-offset]
        minima=image_trim[local_min_trim]

        local_min_offset=copy.deepcopy(local_min)
        local_min_offset[:,0]=False
        local_min_offset[0,:]=False
        local_min_offset[-1,:]=False
        local_min_offset[:,-1]=False

        X,Y = np.where(local_min_offset==True)
        loc=np.column_stack((X,Y))

    else:
        minima=self.np_data[local_min]
        X,Y = np.where(local_min==True)
        loc=np.column_stack((X,Y))

    #bin into histogram
    counts, bin_edges=np.histogram(np.array(minima), bins=thresholds)
    centers=(bin_edges[:-1] + bin_edges[1:]) / 2

    #output a dictionary
    minima_dict={}
    minima_dict['minima_heights']=centers
    minima_dict['minima_counts']=counts
    minima_dict['minima_values']=np.array(minima)
    minima_dict['minima_locs']=loc

    return minima_dict

  def bispectrum(self,l_edges,side_angle_deg,ratio=0.5,configuration="equilateral",scale=None):
    """
    Calculates the bispectrum of the map in the equilateral or folded configuration

    :param l_edges: Multipole bin edges: these are the side of the triangle in the equilateral configuration or the base of the triangle in the folded configuration
    :type l_edges: array

    :param ratio: ratio between one of the triangle sides and the base in the folded configuration. Must be between 0 and 1
    :type ratio: float.

    :param configuration: must be either "equilateral" or "folded"
    :type configuration: str.

    :param scale: scaling to apply to the cube of the Fourier pixels before harmonic azimuthal averaging. Must be a function that takes the array of multipole magnitudes as an input and returns an array of real positive numbers
    :type scale: callable.

    :returns: (multipoles, bispectrum at multipoles)
    :rtype: tuple.

    """
        
    #Check folding ratio
    if (configuration=="folded") and not(ratio>0 and ratio<1):
      raise ValueError("Folding ratio should be between 0 and 1!")

    #Multipole edges
    l = 0.5*(l_edges[:-1] + l_edges[1:])

    #Calculate FFT of the map via FFT
    ft_map = fftengine.rfft2(self.np_data)

    #Calculate bispectrum
    if configuration in ("equilateral","folded"):
      bispectrum = _topology.bispectrum(ft_map,ft_map,ft_map,side_angle_deg,l_edges,configuration,ratio)
    else:
      raise NotImplementedError("Bispectrum configuration '{0}' not implemented!".format(configuration))

    #Return
    
    bispectrum_dict={}
    bispectrum_dict['ell']=l
    bispectrum_dict['bispectrum']=bispectrum
    
    return bispectrum_dict