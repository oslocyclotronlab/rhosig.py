from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

# Script to decompose the frist generations matrix P 
# into the NLD $\rho$ and transmission coefficient $T$
# (or $\gamma$-ray strength function $gsf$) respectivly

def div0( a, b ):
  """ division function designed to ignore / 0, i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
  with np.errstate(divide='ignore', invalid='ignore'):
    c = np.true_divide( a, b )
    c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
  return c

def PfromRhoT(rho, T, type="transCoeff"):
  # generate a first gernation matrix P from
  # given input:      rho and T  /or/ rho and gsf 
  # chosen by type= "transCoeff" /or/   "gsfL1"
  Nbins = len(rho) #-- TODO: update for different #bins for Ex and Eg
  P = np.zeros((Nbins,Nbins))
  for i_Ex in range(Nbins):
    for i_Eg in range(i_Ex+1):
      i_Ediff = i_Ex - i_Eg
      if i_Ediff>=0: # no gamma's with higher energy then the excitation energy
        P[i_Ex,i_Eg] = rho[i_Ediff] * T[i_Eg]
        if type=="gsfL1": # if input T was a gsf, not transmission coeff: * E^(2L+1)
          Eg = Emid[i_Eg]
          P[i_Ex,i_Eg] *= pow(Eg,3.)
  return P

def chi2(rho, T, Pexp):
    Pfit = PfromRhoT(rho, T)
    # chi^2 = (data - fit)^2 / unc.^2, where unc.^2 = #cnt for Poisson dist.
    chi2 = np.sum( div0(np.power((Pexp - Pfit), 2.),Pexp))
    return chi2

def rhoTfrom1D(x1D):
  # split 1D array to who equal length subarrays
  halflength = int(len(x1D)/2)
  rho = x1D[:halflength]
  T = x1D[halflength:]
  return rho, T

def objfun1D(x, *args):
  # 1D version of the chi2 function (needed for minimize function)
  # so x has one dimension only, but may be nested to contain rho and T
  Pexp = np.array(args)
  rho, T= rhoTfrom1D(x)
  return chi2(rho, T, Pexp)

def decompose_matrix(P_in, Emid, fill_value=0):
  # routine for the decomposition of the input 
  # inputs:
  # P_in: Matrix to be decomposed
  # Emin: Array of middle-bin values,
  print "attempt decomposition"

  # protect input arrays
  P_in = np.copy(P_in)
  Emid = np.copy(Emid)

  # TODO: update for different #bins for Ex and Eg
  Nbins = len(P_in) # hand adjusted

  # manipulation to try to improve the fit
  P_in[np.where(P_in == 0)] = fill_value # fill holes with a really small number
  P_in = np.tril(P_in) # set lower triangle to 0 -- due to array form <-> where Eg>Ex

  # creating some articifical holes -- simulating artifacts
  # P_in[40:50,40:50]=0
  # P_in[10:15,5:10]=0
  # P_in[0:35,0:35]=0

  # initial guesses
  rho0 = np.ones(Nbins)

  T0 = np.zeros(Nbins)     # inigial guess for T  following
  for i_Eg in range(Nbins): # eq(6) in Schiller2000
    i_start = max(0,i_Eg)   # !!!!! TODO: should be max(i_Exmin,i_Eg)
    T0[i_Eg] = np.sum(P_in[i_start:,i_Eg])  # and only up to i_Exmax

  p0 = np.append(rho0,T0) # create 1D array of the initial guess

  # minimization
  res = minimize(objfun1D, x0=p0, args=P_in, method="Powell",
    options={'disp': True})
  # different other methods tried:
  # res = minimize(objfun1D, x0=p0, args=P_in,
  #   options={'disp': True})
  # res = minimize(objfun1D, x0=p0, args=P_in, method="L-BFGS-B",
  #   options={'disp': True}) # does a bad job when you include the weightings
  # res = minimize(objfun1D, x0=p0, args=P_in, method="Nelder-Mead",
  #   options={'disp': True}) # does a bad job
  # res = minimize(objfun1D, x0=p0, args=P_in, method="BFGS",
  #   options={'disp': True}) # does a bad job

  p_fit = res.x
  rho_fit, T_fit= rhoTfrom1D(p_fit)

  return rho_fit, T_fit

