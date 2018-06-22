# generate rho and T from model
import numpy as np

# commonly used const. strength_factor, convert in mb^(-1) MeV^(-2)
strength_factor = 8.6737E-08   

def SLO(E, E0, Gamma0, sigma0):
    # Standard Lorentzian,
    # adapted from Kopecky & Uhl (1989) eq. (2.1)
    f = strength_factor * sigma0 * E * Gamma0**2 / ( (E**2 - E0**2)**2 + E**2 * Gamma0**2 )
    return f

def rho_CT(Ex, E0, T):
  # constant temperature NLD formula
  Ex = np.atleast_1d(Ex)
  Eff = Ex - E0
  rho = np.zeros(len(Ex))
  for i in range(len(Ex)):
    if Eff[i]>0:
      rho[i] = np.exp(Eff[i] / T) / T
  return rho

def generateRhoT(Emid):
	# generate NLD
	E1 = -0.7   # shift
	Tct  = 0.7  # temperature
	rho_true = rho_CT(Emid,E1,Tct)

	# generate gsf
	#                   MeV, MeV, mb                        
	E0, Gamma0, sigma0 = 9., 4., 12.
	E1, Gamma1, sigma1 = 4., 2., 3.
	gsf_true = SLO(Emid, E0, Gamma0, sigma0) +  SLO(Emid, E1, Gamma1, sigma1)
	T_true = gsf_true * pow(Emid,3.) # transcoeff = gsf * E^(2L+1)

	return rho_true, T_true, gsf_true