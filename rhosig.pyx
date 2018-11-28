# cython: profile=True
"""
Script to decompose the frist generations matrix P
into the NLD $\rho$ and transmission coefficient $T$
(or $\gamma$-ray strength function $gsf$) respectivly

to compile, run following:
cython3 rhosig.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.5 -o rhosig.so rhosig.c

Copyright (C) 2018 Fabio Zeiser
University of Oslo
fabiobz [0] fys.uio.no

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
cimport cython
cimport numpy as np

def decompose_matrix(P_in, P_err, Emid, Emid_rho, Emid_Ex, fill_value=0):
    """ routine for the decomposition of the first generations spectrum P_in

    Parameters:
    -----------
    P_in : ndarray
        First generations matrix to be decomposed
    Emin : ndarray
        Array of middle-bin values

    Returns:
    --------
    rho_fit: ndarray
        fitted nld
    T_fit: ndarray
        fitted transmission coefficient

    """
    print("attempt decomposition")

    # protect input arrays
    P_in = np.copy(P_in)
    P_err = np.copy(P_err)
    Emid = np.copy(Emid)

    Nbins_Ex, Nbins_T = np.shape(P_in)
    Nbins_rho = Nbins_T

    # manipulation to try to improve the fit
    # TODO: imporvement might be to look for elements = 0 only in the trangle Ex<Eg
    #        + automate what value should be filled. Maybe 1/10 of smallest value in matrix?
    # if fill_value!=0:
    #   P_in[np.where(P_in == 0)] = fill_value # fill holes with a really small number
    #   P_in = np.tril(P_in,k=Nbins_T - Nbins_rho) # set lower triangle to 0 -- due to array form <-> where Eg>Ex

    # initial guesses
    rho0 = np.ones(Nbins_rho)

    T0 = np.zeros(Nbins_T)     # inigial guess for T  following
    for i_Eg in range(Nbins_T): # eq(6) in Schiller2000
        T0[i_Eg] = np.sum(P_in[:,i_Eg]) # no need for i_start; we trimmed the matrix already

    p0 = np.append(rho0,T0) # create 1D array of the initial guess

    # minimization
    res = minimize(objfun1D, x0=p0, args=(P_in,P_err,Emid,Emid_rho,Emid_Ex), method="Powell",
        options={'disp': True})
    # further optimization: eg through higher tolderaced xtol and ftol
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
    rho_fit, T_fit= rhoTfrom1D(p_fit, Nbins_rho)

    return rho_fit, T_fit


def objfun1D(x, *args):
    """
    1D version of the chi2 function (needed for minimize function)
    so x has one dimension only, but may be nested to contain rho and T

    Parameters:
    ----------
    x: ndarray
        workaround: 1D representation of the parameters rho and T
    args: tuple
        tuple of the fixed parameters needed to completely specify the function

    """

    Pexp, Perr, Emid, Emid_rho, Emid_Ex = args
    Pexp = np.asarray(Pexp)
    Perr = np.asarray(Perr)
    Emid = np.asarray(Emid)
    Emid_rho = np.asarray(Emid_rho)
    Emid_Ex = np.asarray(Emid_Ex)
    Pexp = Pexp.reshape(-1, Pexp.shape[-1])
    Nbins_Ex, Nbins_T = np.shape(Pexp)
    Nbins_rho = Nbins_T
    rho, T = rhoTfrom1D(x, Nbins_rho)
    return chi2(rho, T, Pexp, Perr, Emid, Emid_rho, Emid_Ex)


def chi2(np.ndarray rho, np.ndarray T, np.ndarray Pexp, np.ndarray Perr, np.ndarray Emid, np.ndarray Emid_rho, np.ndarray Emid_Ex):
    """ Chi^2 between experimental and fitted first genration matrix"""
    cdef float chi2
    cdef np.ndarray Pfit
    if np.any(rho<0) or np.any(T<0): # hack to implement lower boundary
        chi2 = 1e20
    else:
        Nbins_Ex, Nbins_T = np.shape(Pexp)
        Pfit = PfromRhoT(rho, T, Nbins_Ex, Emid, Emid_rho, Emid_Ex)
        # chi^2 = (data - fit)^2 / unc.^2, where unc.^2 = #cnt for Poisson dist.
        chi2 = np.sum( div0((Pexp - Pfit)**2,Perr**2))
    return chi2


@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(True)  # turn off negative index wrapping for entire function
def PfromRhoT(np.ndarray rho, np.ndarray T, int Nbins_Ex, np.ndarray Emid, np.ndarray Emid_rho, np.ndarray Emid_Ex, type="transCoeff"):
    """ Generate a first gernation matrix P from given nld and T (or gsf)

    Parameters:
    -----------
    rho: ndarray
        nld
    T: ndarray, optional
        transmission coefficient; either this or gsf must be specified
    gsf: ndarray, optional
        gamma-ray strength function; either this or gsf must be specified
    type: string, optional
        chosen by type= "transCoeff" /or/ "gsfL1"
    Nbins_Ex, Emid, Emid_rho, Emid_Ex:
        bin number and bin center values
    Note: rho and T must have the same bin width

    Returns:
    --------
    P: ndarray
        normalized first generations matrix (sum of each Ex bin = 1)
    """

    cdef int Nbins_T = len(T)
    cdef int i_Ex, i_Eg, i_Ef, Nbins
    cdef double Ef ,Ex
    cdef double Eg
    global Emid
    cdef np.ndarray P = np.zeros((Nbins_Ex,Nbins_T))
    # for i_Ex in range(Nbins_Ex):
    #     for i_Eg in range(Nbins_T):
    for i_Ex in range(Nbins_Ex):
        Ex = Emid_Ex[i_Ex]
        Nbins = (np.abs(Emid-Ex)).argmin() + 1
        for i_Eg in range(Nbins):
            Ef = Emid_Ex[i_Ex] - Emid[i_Eg]
            i_Ef = (np.abs(Emid_rho-Ef)).argmin()
            if i_Ef>=0: # no gamma's with higher energy then the excitation energy
                P[i_Ex,i_Eg] = rho[i_Ef] * T[i_Eg]
                if type=="gsfL1": # if input T was a gsf, not transmission coeff: * E^(2L+1)
                    Eg = Emid[i_Eg]
                    P[i_Ex,i_Eg] *= np.power(Eg,3.)
    # normalize each Ex row to 1 (-> get decay probability)
    for i, normalization in enumerate(np.sum(P,axis=1)):
        P[i,:] /= normalization
    return P


def div0(np.ndarray a, np.ndarray b ):
    """ division function designed to ignore / 0,
    i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    cdef np.ndarray c
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c


def rhoTfrom1D(np.ndarray x1D, int Nbins_rho):
    """ split 1D array to who equal length subarrays """
    cdef np.ndarray rho = x1D[:Nbins_rho]
    cdef np.ndarray T = x1D[Nbins_rho:]
    return rho, T


