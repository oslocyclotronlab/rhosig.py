import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from spinfunctions import SpinFunctions
import utilities as ut

# normalization of NLD and GSF with the Oslo method

class NormNLD:
    """ Normalize nld according to nld' = nld * A * np.exp(alpha * Ex)
    Note: This is the transformation eq (3), Schiller2000

    Parameters:
    -----------
    nld : ndarray
        Nuclear level density before normalization, format: [Ex_i, nld_i]
    method : string
        Method for normalization
    pnorm : dict
        Parameters needed for the chosen normalization method
    nldModel : string
        NLD Model for extrapolation
    pext : dict
        Parameters needed for the chosen extrapolation method
    """
    def __init__(self, nld, method, pnorm, nldModel, pext):
        self.nld = nld
        self.method = method
        self.pnorm = pnorm
        self.pext = pext
        self.nldModel = nldModel

        if method is "2points":
            pars_req = {"nldE1", "nldE2"}
            nld_norm, A_norm, alpha_norm = ut.call_model(self.norm_2points,pnorm,pars_req)
            nld_ext = self.extrapolate()
            levels_smoothed, _ = self.discretes(Emids=nld[:,0],resolution=0.1)
            levels_smoothed = levels_smoothed[0:13]
            self.discretes = np.c_[nld[0:13,0],levels_smoothed]

            self.nld_norm = nld_norm
            self.A_norm = A_norm
            self.alpha_norm = alpha_norm
            self.nld_ext = nld_ext
        elif method is "find_norm":
            self.A_norm, self.alpha_norm, self.T= self.find_norm()
            # print(T)
            self.nld_ext = self.extrapolate()
            self.nld_norm = self.normalize(self.nld, self.A_norm, self.alpha_norm)
            self.nld_norm = self.nld_norm
        else:
            raise TypeError("\nError: Normalization model not supported; check spelling\n")

    def norm_2points(self, **kwargs):
        """ Normalize to two given fixed points within "exp". nld Ex-trange

        Input:
        ------
        nldE1 : np.array([E1, nldE1])
        nldE2 : np.array([E2, nldE2])


        """
        Ex = self.nld[:,0]
        nld = self.nld[:,1]
        E1, nldE1 = self.pnorm["nldE1"]
        E2, nldE2 = self.pnorm["nldE2"]

        fnld = ut.log_interp1d(Ex,nld, bounds_error=True)

        # find alpha and A from the normalization points
        alpha = np.log( (nldE2 * fnld(E1)) / (nldE1*fnld(E2)) ) / (E2-E1)
        A = nldE2 / fnld(E2) * np.exp(- alpha * E2)
        print(A)
        print("Normalization parameters: \n alpha={0:1.2e} \t A={1:1.2e} ".format(alpha, A))

        # apply the transformation
        nld_norm = nld * A * np.exp(alpha*Ex)
        return nld_norm, A, alpha

    def extrapolate(self):
        """ Get Extrapolation values """

        model = self.nldModel
        pars = self.pext

        # Earr for extrapolation
        Earr = np.linspace(pars["ext_range"][0],pars["ext_range"][1], num=50)

        # # different extrapolation models
        # def CT(T, Eshift, **kwargs):
        #     """ Constant Temperature"""
        #     return np.exp((Earr-Eshift) / T) / T;

        if model=="CT":
            pars_req = {"T", "Eshift"}
            if pars.has_key("nld_Sn") and pars.has_key("Eshift")==False:
                pars["Eshift"] = self.EshiftFromT(pars["T"], pars["nld_Sn"])
            pars["Earr"] = Earr
            values = ut.call_model(self.CT,pars,pars_req)
        else:
            raise TypeError("\nError: NLD model not supported; check spelling\n")

        extrapolation = np.c_[Earr,values]
        return extrapolation

    @staticmethod
    def normalize(nld, A, alpha):
        """ Normalize nld

        Parameters:
        -----------
        nld : Unnormalized nld, format [Ex, nld]_i
        A : Transformation parameter
        alpha : Transformation parameter

        Returns:
        --------
        nld_norm : Normalized NLD
        """
        Ex = nld[:,0]
        nld_val = nld[:,1]
        if nld.shape[1]==3:
            rel_unc = nld[:,2]/nld[:,1]
        nld_norm = nld_val * A * np.exp(alpha*Ex)
        if nld.shape[1]==3:
            nld_norm = np.c_[nld_norm,nld_norm * rel_unc]
        nld_norm = np.c_[Ex,nld_norm]
        return nld_norm

    @staticmethod
    def CT(Earr, T, Eshift, **kwargs):
        """ Constant Temperature nld"""
        return np.exp((Earr-Eshift) / T) / T;

    @staticmethod
    def EshiftFromT(T, nld_Sn):
        """ Eshift from T for CT formula """
        return nld_Sn[0] - T*np.log(nld_Sn[1]*T)

    def find_norm(self):
        """
        Automatically find best normalization taking into account
        discrete levels at low energy and extrapolation at high energies
        via chi^2 minimization.

        TODO: Check validity of the assumed chi^2 "cost" function

        Returns:
        --------
        res.x: ndarray
            Best fit normalization parameters `[A, alpha, T]`
        """

        from scipy.optimize import curve_fit

        nld = self.nld
        #further parameters
        pnorm = self.pnorm
        E1_low = pnorm["E1_low"]
        E2_low = pnorm["E2_low"]

        E1_high = pnorm["E1_high"]
        E2_high = pnorm["E2_high"]
        nld_Sn = pnorm["nld_Sn"]

        # slice out comparison regions
        idE1 = np.abs(nld[:,0]-E1_low).argmin()
        idE2 = np.abs(nld[:,0]-E2_low).argmin()
        data_low = nld[idE1:idE2,:]

        # Get discretes (for the lower energies)
        levels_smoothed, _ = self.discretes(Emids=nld[:,0],resolution=0.1)
        levels_smoothed = levels_smoothed[idE1:idE2]
        self.discretes = np.c_[nld[idE1:idE2,0],levels_smoothed]

        idE1 = np.abs(nld[:,0]-E1_high).argmin()
        idE2 = np.abs(nld[:,0]-E2_high).argmin()
        data_high = nld[idE1:idE2,:]

        if self.nldModel == "CT":
            nldModel = self.CT
        else:
            print("Other models not yet supported in this fit")

        from scipy.optimize import differential_evolution
        res = differential_evolution(self.chi2_disc_ext,
                                     bounds=[(-10,10),(-10,10),(0.01,1)],
                                     args=(nldModel, nld_Sn, data_low, data_high, levels_smoothed))
        print("Result from find_norm / differential evolution:\n", res)

        T = res.x[2]
        self.pext["T"] = T
        self.pext["Eshift"] = self.EshiftFromT(T, nld_Sn)

        return res.x

    @staticmethod
    def discretes(Emids, fname="discrete_levels.txt", resolution=0.1):
        """ Get discrete levels, and smooth by some resolution [MeV]
        and the bins centers [MeV]
        For now: Assume linear binning """
        energies = np.loadtxt(fname)
        energies /= 1e3 # convert to MeV

        # Emax = energies[-1]
        # nbins = int(np.ceil(Emax/binsize))
        # bins = np.linspace(0,Emax,nbins+1)
        binsize = Emids[1] - Emids[0]
        bin_edges = np.append(Emids, Emids[-1]+binsize)
        bin_edges -= binsize/2.

        hist, _ = np.histogram(energies,bins=bin_edges)
        hist = hist.astype(float)/binsize # convert to levels/MeV

        from scipy.ndimage import gaussian_filter1d
        hist_smoothed = gaussian_filter1d(hist, sigma=resolution/binsize)

        return hist_smoothed, hist

    @staticmethod
    def chi2_disc_ext(x,
                     nldModel, nld_Sn, data_low, data_high, levels_smoothed):
        """
        Chi^2 between discrete levels at low energy and extrapolation at high energies

        TODO: Check validity of the assumed chi^2 "cost" function
        Currently, I assume that I should weight with the number of points
        in each dataset (highvs low energies), such that the discretes
        and nld(Sn) will be weighted the same. But unsure whether that will produce the correct uncertainties (once we figured how to calculate them.)

        Note: Currently working with CT extrapolation only, but should be little effort to change.

        Parameters:
        -----------
        x : ndarray
            Optimization argument in form of a 1D array
        nldModel : string
            NLD Model for extrapolation
        nld_Sn : tuple
            nld at Sn of form `[Ex, value]`
        data_low : ndarray
            Unnormalized nld at lower energies to be compared to discretes of form `[Ex, value]`
        data_high : ndarray
            Unnormalized nld at higher energies to be compared to `nldModel` of form `[Ex, value]`
        levels_smoothed: ndarray
            Discrete levels smoothed by experimental resolution of form `[value]`

        """
        A, alpha = x[:2]
        T = x[2]

        data = NormNLD.normalize(data_low, A, alpha)
        n_low = len(data)
        chi2 = (data[:,1] - levels_smoothed)**2.
        if data.shape[1] == 3: # weight with uncertainty, if existent
            chi2 /= data[:,2]**2
        chi2_low = np.sum(chi2)/n_low

        data = NormNLD.normalize(data_high, A, alpha)
        n_high = len(data)
        Eshift = NormNLD.EshiftFromT(T, nld_Sn)
        chi2 = (data[:,1] - nldModel(data[:,0], T, Eshift))** 2.
        if data.shape[1] == 3: # weight with uncertainty, if existent
            chi2 /= data[:,2]**2
        chi2_high = np.sum(chi2)/n_high

        chi2 = (chi2_low + chi2_high)*(n_high+n_low)
        return chi2


# extrapolations of the gsf
def gsf_extrapolation(pars, ext_range):
    """finding and plotting extraploation of the gsf
      input parameters:
      pars: dictionary with saved parameters
      ext_range: (plot) range for extrapolation

      return: np.array of lower extrapolation, np.array of lower extrapolation
    """

    def f_gsf_ext_low(Eg, c, d):
        return np.exp(c*Eg+d)

    def f_gsf_ext_high(Eg, a, b):
        return np.exp(a*Eg+b) / np.power(Eg,3)

    Emin_low, Emax_low, Emin_high, Emax_high = ext_range
    Emid_ext_low = np.linspace(Emin_low,Emax_low)
    Emid_ext_high = np.linspace(Emin_high,Emax_high)
    ext_a, ext_b =  pars['gsf_ext_high']
    ext_c, ext_d =  pars['gsf_ext_low']

    gsf_ext_low = np.column_stack((Emid_ext_low, f_gsf_ext_low(Emid_ext_low, ext_c, ext_d)))
    gsf_ext_high = np.column_stack((Emid_ext_high, f_gsf_ext_high(Emid_ext_high, ext_a, ext_b)))

    return gsf_ext_low, gsf_ext_high


def transformGSF(Emid_Eg, Emid_nld, rho_in, gsf_in,
                 nld_ext,
                 gsf_ext_low, gsf_ext_high,
                 Jtarget, D0, Gg, Sn, alpha_norm,
                 normMethod,
                 spincutModel, spincutPars={}):
  # transform the gsf extracted with the Oslo method
  # to the average total radiative width <Gg>
  # returns normalized GSF (L=1) from an input gamma-ray strength function gsf
  # inputs:
    # Emid_Eg, rho_in, gsf in MeV, MeV^-1, MeV^-3 -- -- important: gsf needs to be "shape corrected" by alpha_norm
    # nld_ext: extrapolation of nld
    # gsf_ext_low, high: extrapolations of gsf -- important: need to be "shape corrected" bs alpha_norm
    # Jtarget in 1
    # D0 in eV
    # Gg in meV
    # Sn in MeV
  # assuming dipole radiation
      # interpolate NLD and T

  def SpinDist(Ex, J):
    return SpinFunctions(Ex=Ex, J=J, model=spincutModel, pars=spincutPars).distibution()

  def GetNormFromGgD0(Gg, D0, Jtarget):
    # get the normaliation, see eg. eq (26) in Larsen2011; but converted T to gsf

    # compose nld and gsf function of data & extrapolation
    frho_exp = ut.log_interp1d(Emid_nld,rho_in)
    fnld_ext = ut.log_interp1d(nld_ext[:,0], nld_ext[:,1])

    fgsf_exp   = ut.log_interp1d(Emid_Eg, gsf_in)
    fgsf_ext_low = ut.log_interp1d(gsf_ext_low[:,0], gsf_ext_low[:,1])
    fgsf_ext_high = ut.log_interp1d(gsf_ext_high[:,0], gsf_ext_high[:,1])

    # extapolate "around" dataset
    def frho(E):
      if E==0:
        val = 1
      elif E <= Emid_nld[-1]:
        val = frho_exp(E)
      else:
        val = fnld_ext(E)
      return val

    def fgsf(E):
      if E < Emid_Eg[0]:
        val = fgsf_ext_low(E)
      elif E <= Emid_Eg[-1]:
        val = fgsf_exp(E)
      else:
        val = fgsf_ext_high(E)
      return val

    # calculate integral
    Eint_min = 0
    Eint_max = Sn # Sn + Exres # following routine by Magne; Exres is the energy resolution (of SiRi)
    Eintegral, stepSize = np.linspace(Eint_min,Eint_max,num=100, retstep=True) # number of interpolation points

    if(normMethod=="standard"):
      # equalts "old" (normalization.f) version in the Spin sum
      # get the normaliation, see eg. eq (26) in Larsen2011; but converted T to gsf
      # further assumptions: s-wave (currently) and equal parity
      def SpinSum(Ex, Jtarget):
        if Jtarget == 0: #  if(Jtarget == 0.0)      I_i = 1/2 => I_f = 1/2, 3/2
          return SpinDist(Ex,Jtarget+1/2) + SpinDist(Ex,Jtarget+3/2)
        elif Jtarget == 1/2: #  if(Jtarget == 0.5)  I_i = 0, 1  => I_f = 0, 1, 2
          return SpinDist(Ex,Jtarget-1/2) + 2*SpinDist(Ex,Jtarget+1/2) + SpinDist(Ex,Jtarget+3/2)
        elif Jtarget == 1: #  if(Jtarget == 0.5)     I_i = 1/2, 3/2  => I_f = 1/2, 3/2, 5/2
          return 2*SpinDist(Ex,Jtarget-1/2) + 2*SpinDist(Ex,Jtarget+1/2) + SpinDist(Ex,Jtarget+3/2)
        elif Jtarget > 1: #  J_target > 1   ->       I_i = Jt-1/2, Jt+1/2  => I_f = Jt-3/2, Jt-1/2, Jt+3/2, Jt+5/2
          return SpinDist(Ex,Jtarget-3/2) + 2*SpinDist(Ex,Jtarget-1/2) + 2*SpinDist(Ex,Jtarget+1/2) + SpinDist(Ex,Jtarget+3/2)
        else:
          ValueError("Negative J not supported")
      # perform integration by summation
      integral = 0
      for Eg in Eintegral:
          Ex = Sn - Eg
          if Eg<=Emid_nld[0]: print("warning: Eg < {0}; check rho interpolate".format(Emid_nld[0]))
          if Ex<=Emid_nld[0]: print("warning: at Eg = {0}: Ex <{1}; check rho interpolate".format(Eg, Emid_nld[0]))
          integral += np.power(Eg,3) * fgsf(Eg) * frho(Ex) * SpinSum(Ex, Jtarget)
      integral *= stepSize
      # factor of 2 because of equi-parity (we use total nld in the
      # integral above, instead of the "correct" nld per parity)
      # Units: G / (D) = meV / (eV*1e3) = 1
      norm = 2 * Gg / ( integral * D0*1e3)

    elif(normMethod=="test"):
      # experimental new version of the spin sum and integration
      # similar to (26) in Larsen2011, but derived directly from the definition in Bartholomew ; but converted T to gsf
      # further assumptions: s-wave (currently) and equal parity

      # input checks
      rho01plus = 1/2 * frho(Sn) * (SpinDist(Sn,Jtarget-1/2)+SpinDist(Sn,Jtarget+1/2))
      D0_from_frho = 1/rho01plus *1e6
      D0_diff = abs((D0 - D0_from_frho))
      if (D0_diff > 0.1 * D0): ValueError("D0 from extrapolation ({}) and from given D0 ({}) don't match".format(D0_from_frho,D0))

      # Calculating the nlds per J and parity in the residual nucleus before decay, and the accessible spins
      # (by dipole decay <- assumption)
      if Jtarget == 0:     #  J_target = 0   ->    I_i = 1/2 => I_f = 1/2, 3/2
        # I_residual,i = 1/2 -> I_f = 0, 1
        rho0pi = 1/2 * frho(Sn) * SpinDist(Sn,Jtarget0+1/2)
        accessible_spin0 = lambda Ex, Jtarget: SpinDist(Ex,Jtarget-1/2) + SpinDist(Ex,Jtarget+1/2)
      elif Jtarget == 1/2: #  J_target = 1/2   ->      I_i = 0, 1  => I_f = 0, 1, 2
        # I_residual,i = 0 -> I_f = 1
        rho0pi = 1/2 * frho(Sn) * SpinDist(Sn,Jtarget-1/2)
        accessible_spin0 = lambda Ex, Jtarget: SpinDist(Ex,Jtarget+1/2)
        # I_residual,i = 1 -> I_f = 0,1,2
        rho1pi = 1/2 * frho(Sn) * SpinDist(Sn,Jtarget+1/2)
        accessible_spin1 = lambda Ex, Jtarget: SpinDist(Ex,Jtarget-1/2) + SpinDist(Ex,Jtarget+1/2) + SpinDist(Ex,Jtarget+3/2)
      elif Jtarget == 1: #  J_target = 1  ->       I_i = 1/2, 3/2  => I_f = 1/2, 3/2, 5/2
        # I_residual,i = 1/2 -> I_f = 1/2, 3/2
        rho0pi = 1/2 * frho(Sn) * SpinDist(Sn,Jtarget-1/2)
        accessible_spin0 = lambda Ex, Jtarget: SpinDist(Ex,Jtarget-1/2) + SpinDist(Ex,Jtarget+1/2)
        # I_residual,i = 3/2 -> I_f = 1/2, 3/2, 5/2
        rho1pi = 1/2 * frho(Sn) * SpinDist(Sn,Jtarget+1/2)
        accessible_spin1 = lambda Ex, Jtarget: SpinDist(Ex,Jtarget-1/2) + SpinDist(Ex,Jtarget+1/2) + SpinDist(Ex,Jtarget+3/2)
      elif Jtarget > 1: #J_target > 1   ->       I_i = Jt-1/2, Jt+1/2  => I_f = Jt-3/2, Jt-1/2, Jt+3/2, Jt+5/2
        # I_residual,i = Jt-1/2 -> I_f = Jt-3/2, Jt-1/2, Jt+1/2
        rho0pi = 1/2 * frho(Sn) * SpinDist(Sn,Jtarget-1/2)
        accessible_spin0 = lambda Ex, Jtarget: SpinDist(Ex,Jtarget-3/2) + SpinDist(Ex,Jtarget-1/2) + SpinDist(Ex,Jtarget+1/2)
        # I_residual,i = Jt+1/2 -> I_f = Jt-1/2, Jt+1/2, Jt+3/2
        rho1pi = 1/2 * frho(Sn) * SpinDist(Sn,Jtarget+1/2)
        accessible_spin1 = lambda Ex, Jtarget: SpinDist(Ex,Jtarget-1/2) + SpinDist(Ex,Jtarget+1/2) + SpinDist(Ex,Jtarget+3/2)
      else:
        ValueError("Negative J not supported")
      # perform integration by summation
      integral0 = 0
      integral1 = 0
      for Eg in Eintegral:
          Ex = Sn - Eg
          if Eg<=Emid_nld[0]: print("warning: Eg < {0}; check rho interpolate".format(Emid_nld[0]))
          if Ex<=Emid_nld[0]: print("warning: at Eg = {0}: Ex <{1}; check rho interpolate".format(Eg, Emid_nld[0]))
          integral0 += np.power(Eg,3) * fgsf(Eg) * frho(Ex) * accessible_spin0(Ex, Jtarget)
          integral1 += np.power(Eg,3) * fgsf(Eg) * frho(Ex) * accessible_spin1(Ex, Jtarget)
      # simplification: <Gg>_experimental is usually reported as the average over all individual
      # Gg's. Due to a lack of further knowledge, we assume that there are equally many transisions from target states
      # with It+1/2 as from It-1/2. Then we find:
      # <Gg> = ( <Gg>_(I+1/2) + <Gg>_(I+1/2) ) / 2
      integral = (1/rho0pi * integral0 + 1/rho1pi * integral1)/2
      integral *= stepSize
      # factor of 2 because of equi-parity (we use total nld in the
      # integral above, instead of the "correct" nld per parity)
      # Units: G / (integral) = meV / (MeV*1e9) = 1
      norm = 2 * Gg / ( integral *1e9)

    return norm

  b_norm = GetNormFromGgD0(Gg, D0, Jtarget)
  gsf_norm = gsf_in * b_norm

  gsf_ext_low_norm = gsf_ext_low
  gsf_ext_high_norm = gsf_ext_high
  gsf_ext_low_norm[:,1] *= b_norm
  gsf_ext_high_norm[:,1] *= b_norm

  print("alpha_norm: {0}".format(alpha_norm))
  print("b_norm: {0}".format(b_norm))

  return gsf_norm, gsf_ext_low_norm, gsf_ext_high_norm, b_norm


def normalizeGSF(Emid_Eg, Emid_nld, rho_in, gsf_in,
                 nld_ext,
                 gsf_ext_range, pars,
                 Jtarget, D0, Gg, Sn, alpha_norm,
                 normMethod,
                 makePlot, interactive,
                 spincutModel, spincutPars={}):
  # normalize the gsf extracted with the Oslo method
  # to the average total radiative width <Gg>
  # returns normalized GSF (L=1) from an input gamma-ray strength function gsf
  # inputs:
    # Emid_Eg, rho_in, gsf in MeV, MeV^-1, MeV^-3
    # nld_ext: extrapolation of nld
    # gsf_ext_range: extrapolation ranges of gsf
    # Jtarget in 1
    # D0 in eV
    # Gg in meV
    # Sn in MeV
  # assuming dipole radiation
      # interpolate NLD and T


  #   def trans_extrapolation(Emid_Eg, gsf_fit, pars, ext_range, makePlot, interactive):
  # """finding and plotting extraploation of the transmission coefficient/gsf
  #   input parameters:
  #   Emid and gsf_fit = reference transmission coefficient to plot
  #   pars: dictionary with saved parameters, if existing
  #   ext_range: (plot) range for extrapolation
  #   makePlot: flag for creating plots
  #   interactive: flag for to enable interactive change of pars

  #   return: np.array of lower extrapolation, np.array of lower extrapolation
  # """

  # # find parameters
  # key = 'gsf_ext_low'
  # if key not in pars:
  #     pars['gsf_ext_low']= np.array([0.99,10])
  # key = 'trans_ext_high'
  # if key not in pars:
  #     pars['gsf_ext_high']= np.array([0.99,10])

  # def f_gsf_ext_low(Eg, c, d):
  #     return np.exp(c*Eg+d)

  # def f_gsf_ext_high(Eg, a, b):
  #     return np.exp(a*Eg+b) / np.exp(Eg,3)

  # Emin_low, Emax_low, Emin_high, Emax_high = ext_range
  # Emid_ext_low = np.linspace(Emin_low,Emax_low)
  # Emid_ext_high = np.linspace(Emin_high,Emax_high)
  # ext_a, ext_b =  pars['trans_ext_high']
  # ext_c, ext_d =  pars['trans_ext_low']

  # trans_ext_low = np.column_stack((Emid_ext_low, f_trans_ext_low(Emid_ext_low, ext_c, ext_d)))
  # trans_ext_high = np.column_stack((Emid_ext_high, f_trans_ext_high(Emid_ext_high, ext_a, ext_b)))

   # find parameters
  key = 'gsf_ext_low'
  if key not in pars:
      pars['gsf_ext_low']= np.array([2.,-25.])
  key = 'gsf_ext_high'
  if key not in pars:
      pars['gsf_ext_high']= np.array([2.,-25.])

  # creating the defaults
  gsf_shape = gsf_in * np.exp(alpha_norm * Emid_Eg) # "shape" - correction of the transformation

  gsf_ext_low, gsf_ext_high = gsf_extrapolation(pars, gsf_ext_range)
  gsf, gsf_ext_low, gsf_ext_high, b_norm =  transformGSF(Emid_Eg=Emid_Eg, Emid_nld=Emid_nld, rho_in=rho_in, gsf_in=gsf_shape,
                                                       nld_ext=nld_ext,
                                                       gsf_ext_low=gsf_ext_low, gsf_ext_high=gsf_ext_high,
                                                       Jtarget=Jtarget, D0=D0, Gg=Gg, Sn=Sn,
                                                       alpha_norm=alpha_norm,
                                                       normMethod=normMethod,
                                                       spincutModel=spincutModel, spincutPars=spincutPars)

  if makePlot:
      fig, ax = plt.subplots()
      plt.subplots_adjust(left=0.25, bottom=0.35)

      # gsf
      [gsf_plot]=ax.plot(Emid_Eg,gsf,"o")
      [gsf_ext_high_plt] = ax.plot(gsf_ext_high[:,0],gsf_ext_high[:,1],"r--", label="ext. high")
      [gsf_ext_low_plt] = ax.plot(gsf_ext_low[:,0],gsf_ext_low[:,1],"b--", label="ext. high")

      Emin_low, Emax_low, Emin_high, Emax_high = gsf_ext_range
      ax.set_xlim([Emin_low,Emax_high])
      ax.set_yscale('log')
      ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
      ax.set_ylabel(r'$gsf [MeV^-1]$')

      legend = ax.legend()

      # load "true" gsf
      gsf_true_all = np.loadtxt("compare/240Pu/GSFTable_py.dat")
      gsf_true_tot = gsf_true_all[:,1] + gsf_true_all[:,2]
      gsf_true = np.column_stack((gsf_true_all[:,0],gsf_true_tot))
      if gsf_true is not None: ax.plot(gsf_true[:,0],gsf_true[:,1])

      if interactive: # interactively change the extrapolation
          # Define an axes area and draw a slider in it
          axis_color = 'lightgoldenrodyellow'
          ext_a, ext_b =  pars['gsf_ext_high']
          ext_c, ext_d =  pars['gsf_ext_low']
          ext_a_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axis_color)
          ext_b_slider_ax  = fig.add_axes([0.25, 0.10, 0.65, 0.03], facecolor=axis_color)
          ext_c_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
          ext_d_slider_ax  = fig.add_axes([0.25, 0.20, 0.65, 0.03], facecolor=axis_color)

          sext_a = Slider(ext_a_slider_ax, 'a', 0., 2., valinit=ext_a)
          sext_b = Slider(ext_b_slider_ax, 'b', 0, 5, valinit=ext_b)
          sext_c = Slider(ext_c_slider_ax, 'c', 0, 2., valinit=ext_c)
          sext_d = Slider(ext_d_slider_ax, 'd', 0, 5, valinit=ext_d)

          def slider_update(val):
              # nonlocal gsf, gsf_ext_low, gsf_ext_high, b_norm
              ext_a = sext_a.val
              ext_b = sext_b.val
              ext_c = sext_c.val
              ext_d = sext_d.val
              # save the values
              pars['gsf_ext_low'] = np.array([ext_c,ext_d])
              pars['gsf_ext_high'] = np.array([ext_a,ext_b])
              # apply
              gsf_ext_low, gsf_ext_high = gsf_extrapolation(pars, gsf_ext_range)
              gsf, gsf_ext_low, gsf_ext_high, b_norm =  transformGSF(Emid_Eg=Emid_Eg, Emid_nld=Emid_nld, rho_in=rho_in, gsf_in=gsf_shape,
                                                                   nld_ext=nld_ext,
                                                                   gsf_ext_low=gsf_ext_low, gsf_ext_high=gsf_ext_high,
                                                                   Jtarget=Jtarget, D0=D0, Gg=Gg, Sn=Sn,
                                                                   alpha_norm=alpha_norm,
                                                                   normMethod=normMethod,
                                                                   spincutModel=spincutModel, spincutPars=spincutPars)
              gsf_plot.set_ydata(gsf)
              gsf_ext_high_plt.set_ydata(gsf_ext_high[:,1])
              gsf_ext_low_plt.set_ydata(gsf_ext_low[:,1])
              fig.canvas.draw_idle()

          sext_a.on_changed(slider_update)
          sext_b.on_changed(slider_update)
          sext_c.on_changed(slider_update)
          sext_d.on_changed(slider_update)

          reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
          button = Button(reset_ax, 'Reset', color=axis_color, hovercolor='0.975')

          def reset(event):
              sext_a.reset()
              sext_b.reset()
              sext_c.reset()
              sext_d.reset()
          button.on_clicked(reset)
      plt.show()

      # repead this, to get the values from the end of the plot
      ext_a, ext_b =  pars['gsf_ext_high']
      ext_c, ext_d =  pars['gsf_ext_low']
      gsf_ext_low, gsf_ext_high = gsf_extrapolation(pars, gsf_ext_range)
      gsf, gsf_ext_low, gsf_ext_high, b_norm =  transformGSF(Emid_Eg=Emid_Eg, Emid_nld=Emid_nld, rho_in=rho_in, gsf_in=gsf_shape,
                                                           nld_ext=nld_ext,
                                                           gsf_ext_low=gsf_ext_low, gsf_ext_high=gsf_ext_high,
                                                           Jtarget=Jtarget, D0=D0, Gg=Gg, Sn=Sn,
                                                           alpha_norm=alpha_norm,
                                                           normMethod=normMethod,
                                                           spincutModel=spincutModel, spincutPars=spincutPars)

  return gsf, b_norm, gsf_ext_low, gsf_ext_high
