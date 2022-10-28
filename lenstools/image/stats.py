from __future__ import division
from lenstools.extern import _topology
from lenstools.utils.fft import NUMPYFFTPack
fftengine = NUMPYFFTPack()

def powerSpectrum(data,l_edges,side_angle_deg,scale=None):

  """
  Measures the power spectrum of the convergence map at the multipole moments specified in the input

  :param l_edges: Multipole bin edges
  :type l_edges: array

  :param scale: scaling to apply to the square of the Fourier pixels before harmonic azimuthal averaging. Must be a function that takes the array of multipole magnitudes as an input and returns an array of real numbers
  :type scale: callable.

  :returns: (l -- array,Pl -- array) = (binned multipole moments, power spectrum at multipole moments)
  :rtype: tuple.

  >>> test_map = ConvergenceMap.load("map.fit")
  >>> l_edges = np.arange(200.0,5000.0,200.0)
  >>> l,Pl = test_map.powerSpectrum(l_edges)

  """
  l = 0.5*(l_edges[:-1] + l_edges[1:])

  #Calculate the Fourier transform of the map with numpy FFT
  ft_map = fftengine.rfft2(data)



  #Compute the power spectrum with the C backend implementation
  power_spectrum = _topology.rfft2_azimuthal(ft_map,ft_map,side_angle_deg,l_edges, scale)

  #Output the power spectrum
  return l,power_spectrum
