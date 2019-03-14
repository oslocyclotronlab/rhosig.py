from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import scipy.stats as stats

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

        self.nld_norm = None # Normalized nld

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
            popt, samples = self.find_norm()
            self.A_norm = popt["A"][0]
            self.alpha_norm = popt["alpha"][0]
            self.T = popt["T"][0]
            self.normalize_scanning_samples(popt, samples)

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
            if ("nld_Sn" in pars) and ("Eshift" in pars == False):
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
        chi2_args = (nldModel, nld_Sn, data_low, data_high, levels_smoothed)
        res = differential_evolution(self.chi2_disc_ext,
                                     bounds=[(-10,10),(-10,10),(0.01,1)],
                                     args=chi2_args)
        print("Result from find_norm / differential evolution:\n", res)

        import multinest as ml
        p0 = dict(zip(["A","alpha","T"], (res.x).T))
        popt, samples = ml.run_nld_2regions(p0=p0,
                                   chi2_args=chi2_args)

        # set extrapolation as the median values used
        self.pext["T"] = popt["T"][0]
        self.pext["Eshift"] = self.EshiftFromT(popt["T"][0],
                                               self.pnorm["nld_Sn"])
        self.nld_ext = self.extrapolate()

        return popt, samples

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
        bin_edges -= binsize/2

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
        chi2_low = np.sum(chi2)

        data = NormNLD.normalize(data_high, A, alpha)
        n_high = len(data)
        Eshift = NormNLD.EshiftFromT(T, nld_Sn)
        chi2 = (data[:,1] - nldModel(data[:,0], T, Eshift))** 2.
        if data.shape[1] == 3: # weight with uncertainty, if existent
            chi2 /= (data[:,2])**2
        chi2_high = np.sum(chi2)

        chi2 = (chi2_low + chi2_high)
        return chi2

    def normalize_scanning_samples(self, popt, samples):
        """
        Normalize NLD given the transformation parameter samples from multinest

        Parameters:
        -----------
        popt: (dict of str: (float,float)
            Dictionary of median and stddev of the parameters
        samples : dnarray
            Equally weighted samples from the chain
        """
        nld = self.nld
        Ex = self.nld[:,0]

        # self.A_norm = self.popt["A"][0]
        # self.alpha_norm = self.popt["alpha"][0]
        # self.T = self.popt["T"][0]

        # combine uncertainties from nld (from 1Gen fit) and transformation
        if nld.shape[1] == 3:
            N_samples_max = 100
            N_loop = min(N_samples_max, len(samples["A"]))
            nld_samples = np.zeros((N_loop,len(Ex)))
            for i in range(N_loop):
                nld_tmp = stats.norm.rvs(self.nld[:,1],self.nld[:,2])
                nld_tmp = self.normalize(np.c_[Ex, nld_tmp],
                                     samples["A"][i],
                                     samples["alpha"][i])
                nld_samples[i] = nld_tmp[:,1]
            median = np.median(nld_samples,axis=0)
            std = nld_samples.std(axis=0)
            self.nld_norm = np.c_[Ex,median,std]

        # no uncertainties on nld provided
        if nld.shape[1]==2:
            self.nld_norm = self.normalize(self.nld, self.A_norm,
                                           self.alpha_norm)


class NormGSF:
    """ Normalize GSF to <Gg>

    Attributes:
    -----------
    gsf : ndarray
        gsf before normalization
        format: [E_i, value_i] or [E_i, value_i, yerror_i]
    method : string
        Method for normalization
    extModel : dict of string: string
        Model for extrapolation at ("low", "high") energies
    pext : dict
        Parameters needed for the chosen extrapolation method
    ext_range : ndarrat
        (plot) range for extrapolation ([low1,low2.,high1, high2])
    gsf_norm : ndarray
        normalized/trasformed gsf
        format: [E_i, value_i] or [E_i, value_i, yerror_i]

    nld : ndarray
        normalized NLD
        format: [E_i, value_i] or [E_i, value_i, yerror_i]

    Further inputs & units:
    # Emid_Eg, nld, gsf in MeV, MeV^-1, MeV^-3
        important!!: gsf needs to be "shape corrected" by alpha_norm
    # nld_ext: extrapolation of nld
    # gsf_ext_range: extrapolation ranges of gsf
        important!!: need to be "shape corrected" bs alpha_norm
    # Jtarget in 1
    # D0 in eV
    # Gg in meV
    # Sn in MeV

    TODO: Propper implementation taking into account uncertainties

    """
    def __init__(self, gsf, method,
                 Jtarget, D0, Gg, Sn, alpha_norm,
                 pext, ext_range, extModel=None,
                 spincutModel=None, spincutPars={},
                 nld=None, nld_ext=None):
        self.gsf_in = self.transform(gsf, B=1, alpha=alpha_norm) # shape corrected
        self.gsf = np.copy(self.gsf_in)
        self.method = method
        self.Jtarget, self.D0, self.Gg = Jtarget, D0, Gg
        self.Sn = Sn

         # define defaults
        key = 'gsf_ext_low'
        if key not in pext:
            print("Set {} to a default".format(key))
            pext['gsf_ext_low']= np.array([2.,-25.])
        key = 'gsf_ext_high'
        if key not in pext:
            print("Set {} to a default".format(key))
            pext['gsf_ext_high']= np.array([2.,-25.])
        self.pext = pext

        self.ext_range = ext_range
        self.spincutModel = spincutModel
        self.spincutPars = spincutPars
        # if extModel = None:
        #     self.extModel = {"low": "exp_gsf", "high": "exp_trans"}
        self.nld = nld
        #self.nld = nld
        self.nld_ext = nld_ext


        # self.gsf_norm = None # could be the same as self.gsf
        # only usage is now in gsf = transform(self.gsf[...]

        # initial extrapolation
        gsf_ext_low, gsf_ext_high = self.gsf_extrapolation(self.pext)
        self.gsf_ext_low = gsf_ext_low
        self.gsf_ext_high = gsf_ext_high



    @staticmethod
    def transform(gsf, B, alpha):
        """ Normalize gsf

        Parameters:
        -----------
        gsf : ndarray
            Unnormalized gsf, [Ex_i, gsf_i] or [Ex_i, gsf_i, yerror_i]
        B, alpha : float
            Transformation parameters

        Returns:
        --------
        gsf_norm : Normalized gsf
        """
        E_array = gsf[:,0]
        gsf_val = gsf[:,1]
        if gsf.shape[1]==3:
            rel_unc = gsf[:,2]/gsf[:,1]
        gsf_norm = gsf_val * B * np.exp(alpha*E_array)
        if gsf.shape[1]==3:
            gsf_norm = np.c_[gsf_norm,gsf_norm * rel_unc]
        gsf_norm = np.c_[E_array,gsf_norm]
        return gsf_norm


    def gsf_extrapolation(self, pars):
        """finding and plotting extraploation of the gsf
          Parameters:
          -----------
          pars: dictionary with saved parameters

          Returns:
          --------
          gsf_ext_low : np.array of lower extrapolation
          gsf_ext_high : np.array of higher extrapolation

          TODO: Automatic choice of ext model according to method from dict.
        """

        # assert self.extModel==["exp_gsf","exp_trans"]
        ext_range = self.ext_range
        Emin_low, Emax_low, Emin_high, Emax_high = ext_range
        ext_a, ext_b =  pars['gsf_ext_high']
        ext_c, ext_d =  pars['gsf_ext_low']

        def f_gsf_ext_low(Eg, c, d):
            return np.exp(c*Eg+d)

        def f_gsf_ext_high(Eg, a, b):
            return np.exp(a*Eg+b) / np.power(Eg,3)

        Emid = np.linspace(Emin_low,Emax_low)
        value = f_gsf_ext_low(Emid, ext_c, ext_d)
        gsf_ext_low = np.c_[Emid, value]

        Emid = np.linspace(Emin_high,Emax_high)
        value = f_gsf_ext_high(Emid, ext_a, ext_b)
        gsf_ext_high = np.c_[Emid, value]

        return gsf_ext_low, gsf_ext_high


    def fnld(self, E):
        """ compose nld of data & extrapolation

        TODO: Implement uncertainties
        """
        nld = self.nld
        nld_ext = self.nld_ext
        Earr = nld[:,0]
        nld = nld[:,1]

        fexp = ut.log_interp1d(Earr,nld)
        fext = ut.log_interp1d(nld_ext[:,0], nld_ext[:,1])

        conds = [E <= Earr[-1], E > Earr[-1]]
        funcs = [fexp, fext]
        return np.piecewise(E, conds, funcs)


    def fgsf(self, E, gsf, gsf_ext_low, gsf_ext_high):
        """ compose gsf of data & extrapolation

        TODO: Implement uncertainties
        """
        # gsf = self.gsf
        # gsf_ext_low = self.gsf_ext_low
        # gsf_ext_high = self.gsf_ext_high
        Earr = gsf[:,0]
        gsf = gsf[:,1]

        fexp     = ut.log_interp1d(Earr, gsf)
        fext_low = ut.log_interp1d(gsf_ext_low[:,0], gsf_ext_low[:,1])
        fext_high = ut.log_interp1d(gsf_ext_high[:,0], gsf_ext_high[:,1])

        conds = [E < Earr[0], (E >= Earr[0]) & (E <= Earr[-1]), E > Earr[-1]]
        funcs = [fext_low, fexp, fext_high]
        return np.piecewise(E, conds, funcs)


    def spin_dist(self, Ex, J):
        return SpinFunctions(Ex=Ex, J=J,
                             model=self.spincutModel,
                             pars=self.spincutPars).distibution()


    def GetNormFromGgD0(self):
        """
        Get the normaliation, see eg. eq (26) in Larsen2011;
        Note however, that we use gsf instead of T

        Returns:
        --------
        norm : float
            Absolute normalization constant by which we need to scale
            the non-normalized gsf, such that we get the correct <Gg>
        """
        # setup of the integral (by summation)
        Eint_min = 0
        #TODO: Integrate up to Sn + Exres? And/Or from - Exres
        # Sn + Exres # following routine by Magne; Exres is the energy resolution (of SiRi)
        Eint_max = self.Sn
        Nsteps = 100 # number of interpolation points
        Eintegral, stepSize = np.linspace(Eint_min,Eint_max,
                                          num=Nsteps, retstep=True)

        if(self.method=="standard"):
            norm = self.Gg_Norm_standard(Eintegral, stepSize)
        elif(self.method=="test"):
            norm = self.Gg_Norm_test(Eintegral, stepSize)

        return norm


    def Gg_Norm_standard(self, Eintegral, stepSize):
        """ Compute normalization from Gg integral, the "standard" way

        Equals "old" (normalization.f) version in the Spin sum
        get the normaliation, see eg. eq (26) in Larsen2011; but converted T to gsf
        further assumptions: s-wave (currently) and equal parity

        Parameterts:
        ------------
        Eintegral : ndarray
            (Center)bin of Ex enegeries for the integration
        stepSize : ndarray
            Step size for integration by summation

        Returns:
        --------
        norm : float
            Absolute normalization constant
        """

        Gg, D0, Jtarget = self.Gg, self.D0, self.Jtarget
        fnld = self.fnld
        def fgsf(E):
            return self.fgsf(E, self.gsf_in, self.gsf_ext_low,
                             self.gsf_ext_high)
        spin_dist = self.spin_dist
        Enld_exp_min = self.nld[0,0] # lowest energy of experimental nld points
        Sn = self.Sn

        def SpinSum(Ex, Jtarget):
            if Jtarget == 0:
                # if(Jtarget == 0.0) I_i = 1/2 => I_f = 1/2, 3/2
                return spin_dist(Ex,Jtarget+1/2) \
                       + spin_dist(Ex,Jtarget+3/2)
            elif Jtarget == 1/2:
                # if(Jtarget == 0.5)    I_i = 0, 1  => I_f = 0, 1, 2
                return spin_dist(Ex,Jtarget-1/2) \
                       + 2.*spin_dist(Ex,Jtarget+1/2) \
                       + spin_dist(Ex,Jtarget+3/2)
            elif Jtarget == 1.:
                # if(Jtarget == 0.5) I_i = 1/2, 3/2  => I_f = 1/2, 3/2, 5/2
                return 2.*spin_dist(Ex,Jtarget-1/2) \
                       + 2.*spin_dist(Ex,Jtarget+1/2) \
                       + spin_dist(Ex,Jtarget+3/2)
            elif Jtarget > 1.:
            # J_target > 1 > I_i = Jt-1/2, Jt+1/2
            #                    => I_f = Jt-3/2, Jt-1/2, Jt+3/2, Jt+5/2
                return spin_dist(Ex,Jtarget-3/2) \
                       + 2.*spin_dist(Ex,Jtarget-1/2) \
                       + 2.*spin_dist(Ex,Jtarget+1/2) \
                       + spin_dist(Ex,Jtarget+3/2)
            else:
                ValueError("Negative J not supported")

        # perform integration by summation
        # TODO: Revise warnings: do they make sense?
        integral = 0
        for Eg in Eintegral:
                Ex = Sn - Eg
                if Eg<=Enld_exp_min:
                    print("warning: Eg < {0}; check rho interpolate"
                          .format(Enld_exp_min))
                if Ex<=Enld_exp_min:
                    print("warning: at Eg = {0}: Ex < {1}; " +
                           "check rho interpolate".format(Eg, Enld_exp_min))
                integral += np.power(Eg,3) * fgsf(Eg) * fnld(Ex) \
                            * SpinSum(Ex, Jtarget)
        integral *= stepSize

        # factor of 2 because of equi-parity (we use total nld in the
        # integral above, instead of the "correct" nld per parity)
        # Units: G / (D) = meV / (eV*1e3) = 1
        norm = 2. * Gg / ( integral * D0*1e3)
        return norm


    def Gg_Norm_test(self, Eintegral, stepSize):
        """ Compute normalization from Gg integral, "test approach"

        Experimental new version of the spin sum and integration
         similar to (26) in Larsen2011, but derived directly from the definition in Bartholomew ; but converted T to gsf
        Further assumptions: s-wave (currently) and equal parity

        Parameterts:
        ------------
        Eintegral : ndarray
            (Center)bin of Ex enegeries for the integration
        stepSize : ndarray
            Step size for integration by summation

        Returns:
        --------
        norm : float
            Absolute normalization constant
        """

        Gg, D0, Jtarget = self.Gg, self.D0, self.Jtarget
        fnld = self.fnld
        def fgsf(E):
            return self.fgsf(E, self.gsf_in, self.gsf_ext_low, self.gsf_ext_high)
        spin_dist = self.spin_dist
        Enld_exp_min = self.nld[0,0] # lowest energy of experimental nld points
        Sn = self.Sn

        # input checks
        rho01plus = 1/2 * fnld(Sn) \
                    (spin_dist(Sn,Jtarget-1/2) + spin_dist(Sn,Jtarget+1/2))
        D0_from_fnld = 1/rho01plus *1e6
        D0_diff = abs((D0 - D0_from_fnld))
        if (D0_diff > 0.1 * D0):
            ValueError("D0 from extrapolation ({}) " +
                       "and from given D0 ({}) don't match"
                       .format(D0_from_fnld,D0))

        # Calculating the nlds per J and parity in the residual nucleus before decay, and the accessible spins
        # (by dipole decay <- assumption)
        if Jtarget == 0:
            # J_target = 0 > I_i = 1/2 => I_f = 1/2, 3/2
            # I_residual,i = 1/2 -> I_f = 0, 1
            rho0pi = 1/2 * fnld(Sn) * spin_dist(Sn,Jtarget+1/2)
            accessible_spin0 = lambda Ex, Jtarget: spin_dist(Ex,Jtarget-1/2) + spin_dist(Ex,Jtarget+1/2)
            # only one spin accessible
            rho1pi = None
            accessible_spin1 = lambda Ex, Jtarget: None
        elif Jtarget == 1/2:
            # J_target = 1/2  >  I_i = 0, 1  => I_f = 0, 1, 2
            # I_residual,i = 0 -> I_f = 1
            rho0pi = 1/2 * fnld(Sn) * spin_dist(Sn,Jtarget-1/2)
            accessible_spin0 = lambda Ex, Jtarget: spin_dist(Ex,Jtarget+1/2)
            # I_residual,i = 1 -> I_f = 0,1,2
            rho1pi = 1/2 * fnld(Sn) * spin_dist(Sn,Jtarget+1/2)
            accessible_spin1 = lambda Ex, Jtarget: spin_dist(Ex,Jtarget-1/2) + spin_dist(Ex,Jtarget+1/2) + spin_dist(Ex,Jtarget+3/2)
        elif Jtarget == 1:
            # J_target = 1 > I_i = 1/2, 3/2    => I_f = 1/2, 3/2, 5/2
            # I_residual,i = 1/2 -> I_f = 1/2, 3/2
            rho0pi = 1/2 * fnld(Sn) * spin_dist(Sn,Jtarget-1/2)
            accessible_spin0 = lambda Ex, Jtarget: spin_dist(Ex,Jtarget-1/2) + spin_dist(Ex,Jtarget+1/2)
            # I_residual,i = 3/2 -> I_f = 1/2, 3/2, 5/2
            rho1pi = 1/2 * fnld(Sn) * spin_dist(Sn,Jtarget+1/2)
            accessible_spin1 = lambda Ex, Jtarget: spin_dist(Ex,Jtarget-1/2) + spin_dist(Ex,Jtarget+1/2) + spin_dist(Ex,Jtarget+3/2)
        elif Jtarget > 1:
            #J_target > 1 > I_i = Jt-1/2, Jt+1/2
            # => I_f = Jt-3/2, Jt-1/2, Jt+3/2, Jt+5/2
            # I_residual,i = Jt-1/2 -> I_f = Jt-3/2, Jt-1/2, Jt+1/2
            rho0pi = 1/2 * fnld(Sn) * spin_dist(Sn,Jtarget-1/2)
            accessible_spin0 = lambda Ex, Jtarget: spin_dist(Ex,Jtarget-3/2) + spin_dist(Ex,Jtarget-1/2) + spin_dist(Ex,Jtarget+1/2)
            # I_residual,i = Jt+1/2 -> I_f = Jt-1/2, Jt+1/2, Jt+3/2
            rho1pi = 1/2 * fnld(Sn) * spin_dist(Sn,Jtarget+1/2)
            accessible_spin1 = lambda Ex, Jtarget: spin_dist(Ex,Jtarget-1/2) + spin_dist(Ex,Jtarget+1/2) + spin_dist(Ex,Jtarget+3/2)
        else:
            ValueError("Negative J not supported")

        # perform integration by summation
        # TODO: Revise warnings: do they make sense?
        integral0 = 0
        integral1 = 0
        for Eg in Eintegral:
            Ex = Sn - Eg
            if Eg<=Enld_exp_min:
                print("warning: Eg < {0}; check rho interpolate"
                      .format(Enld_exp_min))
            if Ex<=Enld_exp_min:
                print("warning: at Eg = {0}: Ex <{1}; check rho interpolate"
                      .format(Eg, Enld_exp_min))
            integral0 += np.power(Eg,3) * fgsf(Eg) * fnld(Ex) * accessible_spin0(Ex, Jtarget)
            if rho1pi is not None:
                    integral1 += np.power(Eg,3) * fgsf(Eg) * fnld(Ex) * accessible_spin1(Ex, Jtarget)
        # simplification: <Gg>_experimental is usually reported as the average over all individual
        # Gg's. Due to a lack of further knowledge, we assume that there are equally many transisions from target states
        # with It+1/2 as from It-1/2 Then we find:
        # <Gg> = ( <Gg>_(I+1/2) + <Gg>_(I+1/2) ) / 2
        if rho1pi is None:
            integral = 1./rho0pi * integral0
        else:
            integral = (1./rho0pi * integral0 + 1./rho1pi * integral1)/2
        integral *= stepSize
        # factor of 2 because of equi-parity (we use total nld in the
        # integral above, instead of the "correct" nld per parity)
        # Units: G / (integral) = meV / (MeV*1e9) = 1
        norm = 2. * Gg / ( integral *1e9)
        return norm


    def normalizeGSF(self, makePlot, interactive, gsf_referece=None):
        """
        # normalize the gsf extracted with the Oslo method
        # to the average total radiative width <Gg>
        # returns normalized GSF (L=1) from an input gamma-ray strength function gsf

        makePlot: bool
            Plot the normalized gsf
        interactive : bool
            Create interactive plot to change the extraploation parameters
        gsf_referece : ndarray
            Refernce for plotting and normalization during code debugging
        """
        # check input
        if interactive:
            assert interactive == makePlot

        transform = self.transform

        # "shape" - correction  of the transformation
        gsf_ext_low, gsf_ext_high = self.gsf_extrapolation(self.pext)
        self.gsf_ext_low, self.gsf_ext_high = gsf_ext_low, gsf_ext_high
        norm = self.GetNormFromGgD0()
        gsf = transform(self.gsf_in, B=norm, alpha=0)
        gsf_ext_low = transform(gsf_ext_low, B=norm, alpha=0)
        gsf_ext_high = transform(gsf_ext_high, B=norm, alpha=0)

        self.gsf = gsf
        self.gsf_ext_low = gsf_ext_low
        self.gsf_ext_high = gsf_ext_high

        if makePlot:
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.25, bottom=0.35)

            # gsf
            [gsf_plot]=ax.plot(gsf[:,0],gsf[:,1],"o")
            [gsf_ext_high_plt] = ax.plot(gsf_ext_high[:,0],gsf_ext_high[:,1],
                                         "r--", label="ext. high")
            [gsf_ext_low_plt] = ax.plot(gsf_ext_low[:,0],gsf_ext_low[:,1],
                                        "b--", label="ext. high")

            Emin_low, Emax_low, Emin_high, Emax_high = self.ext_range
            ax.set_xlim([Emin_low,Emax_high])
            ax.set_yscale('log')
            ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
            ax.set_ylabel(r'$gsf [MeV^-1]$')

            legend = ax.legend()

            # load referece gsf
            if gsf_referece is not None:
                ax.plot(gsf_referece[:,0],gsf_referece[:,1])
            if not interactive:
                plt.show()

        if interactive:
            # Define an axes area and draw a slider in it
            axis_color = 'lightgoldenrodyellow'
            ext_a, ext_b =  self.pext['gsf_ext_high']
            ext_c, ext_d =  self.pext['gsf_ext_low']
            ext_a_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03],
                                            facecolor=axis_color)
            ext_b_slider_ax  = fig.add_axes([0.25, 0.10, 0.65, 0.03],
                                            facecolor=axis_color)
            ext_c_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03],
                                            facecolor=axis_color)
            ext_d_slider_ax  = fig.add_axes([0.25, 0.20, 0.65, 0.03],
                                            facecolor=axis_color)

            sext_a = Slider(ext_a_slider_ax, 'a', 0., 2., valinit=ext_a)
            sext_b = Slider(ext_b_slider_ax, 'b', -30, 5, valinit=ext_b)
            sext_c = Slider(ext_c_slider_ax, 'c', 0, 2., valinit=ext_c)
            sext_d = Slider(ext_d_slider_ax, 'd', -30, 5, valinit=ext_d)

            def slider_update(val):
                ext_a = sext_a.val
                ext_b = sext_b.val
                ext_c = sext_c.val
                ext_d = sext_d.val
                # save the values
                self.pext['gsf_ext_low'] = np.array([ext_c,ext_d])
                self.pext['gsf_ext_high'] = np.array([ext_a,ext_b])

                # apply
                gsf_ext_low, gsf_ext_high = self.gsf_extrapolation(self.pext)
                self.gsf_ext_low, self.gsf_ext_high = gsf_ext_low, gsf_ext_high
                norm = self.GetNormFromGgD0()
                gsf = transform(self.gsf_in, B=norm, alpha=0)
                gsf_ext_low = transform(gsf_ext_low, B=norm, alpha=0)
                gsf_ext_high = transform(gsf_ext_high, B=norm, alpha=0)

                self.gsf = gsf
                self.gsf_ext_low = gsf_ext_low
                self.gsf_ext_high = gsf_ext_high

                gsf_plot.set_ydata(gsf[:,1])
                gsf_ext_high_plt.set_ydata(gsf_ext_high[:,1])
                gsf_ext_low_plt.set_ydata(gsf_ext_low[:,1])
                fig.canvas.draw_idle()

            sext_a.on_changed(slider_update)
            sext_b.on_changed(slider_update)
            sext_c.on_changed(slider_update)
            sext_d.on_changed(slider_update)

            reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
            button = Button(reset_ax, 'Reset', color=axis_color,
                            hovercolor='0.975')

            def reset(event):
                sext_a.reset()
                sext_b.reset()
                sext_c.reset()
                sext_d.reset()
            button.on_clicked(reset)

            plt.show()
