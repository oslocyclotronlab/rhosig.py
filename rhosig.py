from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

# Script to decompose the frist generations matrix P 
# into the NLD $\rho$ and transmission coefficient $T$
# (or $\gamma$-ray strength function $gsf$) respectivly
# Here: read in a "experimental" 1gen matrix

data = np.loadtxt("1Gen.m", comments="!")
# select fit region by hand
# important: for now, need to adjust Nbins further down!
oslo_matrix = data[20:-20,20:-20] 

# Set bin width and range
bin_width = 0.20
Emin = 0.  # Minimum and maximum excitation                  -- CURRENTLY ARBITRARY
Emax = 5.  # energy over which to extract strength function  -- CURRENTLY ARBITRARY
Nbins = len(oslo_matrix) # hand adjusted
Emax_adjusted = bin_width*Nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,Nbins+1)
Emid = (bins[0:-1]+bins[1:])/2 # Array of middle-bin values, to use for plotting gsf

# analysis parameters

def div0( a, b ):
  """ division function designed to ignore / 0, i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
  with np.errstate(divide='ignore', invalid='ignore'):
    c = np.true_divide( a, b )
    c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
  return c

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

def PfromRhoT(rho, T, type="transCoeff"):
  # generate a first gernation matrix P from
  # given input:      rho and T  /or/ rho and gsf 
  # chosen by type= "transCoeff" /or/   "gsfL1"
  P = np.zeros((Nbins,Nbins))
  for i_Ex in range(Nbins):
    for i_Eg in range(Nbins):
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

# generate true 1Gen matrix and add statistical noise
P_true = oslo_matrix

# maipulation to try to improve the fit
P_true[np.where(P_true == 0)] = 1e-1 # fill holes with a really small number
P_true = np.tril(P_true) # set lower triangle to 0 -- due to array form <-> where Eg>Ex

# creating some articifical holes -- simulating artifacts
# P_true[40:50,40:50]=0
# P_true[10:15,5:10]=0
# P_true[0:35,0:35]=0

# initial guesses
rho0 = np.ones(Nbins)

T0 = np.zeros(Nbins)     # inigial guess for T  following
for i_Eg in range(Nbins): # eq(6) in Schiller2000
  i_start = max(0,i_Eg)   # !!!!! TODO: should be max(i_Exmin,i_Eg)
  T0[i_Eg] = np.sum(P_true[i_start:,i_Eg])  # and only up to i_Exmax

p0 = np.append(rho0,T0) # create 1D array of the initial guess

# minimization
res = minimize(objfun1D, x0=p0, args=P_true, method="Powell",
  options={'disp': True})
# different other methods tried:
# res = minimize(objfun1D, x0=p0, args=P_true,
#   options={'disp': True})
# res = minimize(objfun1D, x0=p0, args=P_true, method="L-BFGS-B",
#   options={'disp': True}) # does a bad job when you include the weightings
# res = minimize(objfun1D, x0=p0, args=P_true, method="Nelder-Mead",
#   options={'disp': True}) # does a bad job
# res = minimize(objfun1D, x0=p0, args=P_true, method="BFGS",
#   options={'disp': True}) # does a bad job

p_fit = res.x
rho_fit, T_fit= rhoTfrom1D(p_fit)
gsf_fit = T_fit/pow(Emid,3)  # assuming dipoles only

# "normalize" -- or at least some approximation to this
# alpha = rho_true[0]/rho_fit[0]
# rho_fit *= alpha
# T_fit   /= alpha
# gsf_fit /= alpha

# Plot it

# New Figure: Oslo type matrix
f_mat, ax_mat = plt.subplots(2,1)

# true matrix
ax = ax_mat[0]
from matplotlib.colors import LogNorm # To get log scaling on the z axis
colorbar_object = ax.pcolormesh(bins, bins, P_true, norm=LogNorm())
f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

# fitted matrix
ax = ax_mat[1]
P_fit = PfromRhoT(rho_fit,T_fit)

from matplotlib.colors import LogNorm # To get log scaling on the z axis
colorbar_object = ax.pcolormesh(bins, bins, P_fit, norm=LogNorm())
f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

# New Figure: compare input and output NLD and gsf
f_mat, ax_mat = plt.subplots(2,1)

# NLD
ax = ax_mat[0]
# ax.plot(Emid,rho_true)
ax.plot(Emid,rho_fit,"o")

ax.set_yscale('log')
ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')

# gsf
ax = ax_mat[1]

# ax.plot(Emid,gsf_true)
ax.plot(Emid,gsf_fit,"o")

ax.set_yscale('log')
ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
ax.set_ylabel(r'$gsf \, \mathrm{(MeV**(-3)}$')

# # Show plot
plt.show()