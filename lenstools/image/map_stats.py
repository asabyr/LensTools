from __future__ import division
from lenstools.extern import _topology
from lenstools.utils.fft import NUMPYFFTPack
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
    ps['power_spectrum']=power_spectrum
    cl_factor=ell*(ell+1.)/(2.*np.pi)*1.e12
    ps['Cl']=power_spectrum*cl_factor

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
    Measures the three Minkowski functionals (area, perimeter and genus characteristic) of the specified map excursion sets.

    :param thresholds: thresholds that define the excursion sets to consider
    :param norm=False: normalization; if set to a True, interprets the thresholds array as units of sigma (the map standard deviation)
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

  def countPeaks(self, thresholds, offset=1):
    """
    Measures peak counts. 
    :param thresholds: thresholds for the peak histogram.
    :param offset: how many pixels to ignore at the edges.
    :returns: a dictionary that contains 'peak_heights' (i.e. midpoints of the thresholds),
    'peak_counts', 'peak_values' and 'peak_locs'.
    """
    peaks=[]
    
    for i in range(len(self.np_data:,0])-2*offset):
        
        for j in range(len(self.np_data0,:])-2*offset):
            
            k=i+offset
            c=j+offset
            
            point=self.np_datak,c]
            
            if point>self.np_datak+1,c] and point>self.np_datak,c+1]:
                if point>self.np_datak-1,c] and point>self.np_datak,c-1]:
                    if point>self.np_datak+1,c+1] and point>self.np_datak-1,c-1]:
                        if point>self.np_datak+1,c-1] and point>self.np_datak-1,c+1]:
                            peaks.append(point)
                            if len(loc)==0:
                              loc=np.array([[k,c]])
                            else:
                              loc=np.concatenate((loc,[[k,c]]))

    counts, bin_edges=np.histogram(np.array(peaks), bins=thresholds)
    centers=(bin_edges[:-1] + bin_edges[1:]) / 2
    
    peaks={}
    peaks['peak_heights']=centers
    peaks['peak_counts']=counts
    peaks['peak_values']=np.array(peaks)
    peaks['peak_locs']=loc

    return peaks
  

  def peakCount_AP(self,thresholds,norm=False):
		
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

      return midpoints, _topology.peakCount(self.np_data,mask_profile,thresholds,sigma)    



      


