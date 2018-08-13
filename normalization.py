import numpy as np
from scipy.interpolate import interp1d
import sys

# normalization of NLD and GSF with the Oslo method

# Normalization of the NLD
def normalizeNLD(E1, nldE1, E2, nldE2, Emid_rho, rho):
  # normalization of the NLD according to the transformation eq (3), Schiller2000
  # iputs: unnormalized nld rho, and their mid-energy bins Emid
  #        E1(2) and nldE1(2): normalization points: Energy and NLD

  # find corresponding energy bins of E1 and E2
  i_E1 = (np.abs(Emid_rho-E1)).argmin()
  i_E2 = (np.abs(Emid_rho-E2)).argmin()
  print i_E1, i_E2, Emid_rho[i_E1], Emid_rho[i_E2]

  # find alpha and A from the normalization points
  alpha = np.log( (nldE2 * rho[i_E1]) / (nldE1*rho[i_E2]) ) / (E2-E1)
  A = nldE2 / rho[i_E2] * np.exp(- alpha * E2)
  print A
  print "Normalization parameters: \n alpha={0:1.2e} \t A={1:1.2e} ".format(alpha, A)

  # apply the transformation
  rho *= A * np.exp(alpha*Emid_rho)
  return rho, alpha, A


# normalize the transmission coefficient extracted with the Oslo method
# to the average total radiative width <Gg>

def normalizeGSF(Emid, Emid_rho,rho_in, T_in, Jtarget, D0, Gg, Sn, alpha_norm, spincutModel, spincutPars={}):
  # returns normalized GSF (L=1) from an input transmission coefficient T
  # inputs:
    # Emid, rho_in, T_in in MeV, MeV^-1, 1
    # Jtarget in 1
    # D0 in eV
    # Gg in meV
    # Sn in MeV
  # assuming dipole radiation

  def SpinDist(Ex, J, model=spincutModel, pars=spincutPars):
    # Get Spin distribution given a spin-cut sigma
    # note: input is sigma2
    # assuming equal parity

    def GetSigma2(Ex, J, model=model, pars=pars):
      # Get the square of the spin cut for a specified model

      # different spin cut models
      def EB05(mass, NLDa, Eshift): 
      # Von Egidy & B PRC72,044311(2005)
      # The rigid moment of inertia formula (RMI)
      # FG+CT
        Eeff = Ex - Eshift
        if Eeff<0: Eeff=0
        sigma2 =  (0.0146 * np.power(mass, 5.0/3.0)
                    * (1 + np.sqrt(1 + 4 * NLDa * Eeff)) 
                    / (2 * NLDa))
        return sigma2
      def EB09_CT(mass): 
        # The constant temperature (CT) formula Von Egidy & B PRC80,054310 and NPA 481 (1988) 189
        sigma2 =  np.power(0.98*(A**(0.29)),2)
        return sigma2
      def EB09_emp(mass,Pa_prime): 
        # Von Egidy & B PRC80,054310
        # FG+CT
        Eeff = Ex - 0.5 * Pa_prime
        if Eeff<0: Eeff=0
        sigma2 = 0.391 * np.power(mass, 0.675) * np.power(mass,0.312)
        return sigma2

      # call a model
      def CallModel(fsigma,pars,pars_req):
        if pars_req <= set(pars): # is the required parameters are a subset of all pars given
            return fsigma(**pars)
        else:
            raise TypeError("Error: Need following arguments for this method: {0}".format(pars_req))

      if model=="EB05": 
        pars_req = {"mass", "NLDa", "Eshift"}
        return CallModel(EB05,pars,pars_req)
      if model=="EB09_CT": 
        pars_req = {"mass"}
        return CallModel(EB09_CT,pars,pars_req)
      if model=="EB09_emp": 
        pars_req = {"mass","Pa_prime"}
        return CallModel(EB09_emp,pars,pars_req) 

      else:
        raise TypeError("\nError: Spincut model not supported; check spelling\n")
        return 1.

    sigma2 = GetSigma2(Ex,J)

    # following Gilbert1965, eq (E4)
    return (2.*J+1.) /(2.*sigma2) * np.exp(-np.power(J+0.5, 2.) / (2.*sigma2))

  def CalcIntegralSwave(Jtarget, Exres=0.2):
    # Calculate normalization integral, see eg. eq (26) in Larsen2011
    
    if Jtarget != 0: 
      print "Formula needs to be extended to more Itargets; take care of Clebsh-Gordan coeffs"
      error() # todo - throw error

    # interpolate NLD and T
    rho = interp1d(Emid_rho,rho_in, bounds_error=False, fill_value=0) # defualt: linear interpolation
    T   = interp1d(Emid, T_in, bounds_error=False, fill_value=0)  # default:linear interpolation

    print "RHOS", rho(1), rho(2), rho(2.5)
    print "T", T(1), T(2), T(2.5), T(3.5)

    # calculate integral
    Eint_min = 0
    Eint_max = Sn + Exres # following routine by Magne; Exres is the energy resolution (of SiRi)
    Eintegral, stepSize = np.linspace(Eint_min,Eint_max,num=100, retstep=True) # number of interpolation points 
   
    #  if(Jtarget == 0.0){      /*I_i = 1/2 => I_f = 1/2, 3/2 */
    norm = 0
    for Eg in Eintegral:
        Ex = Sn - Eg
        if Eg<=Emid_rho[0]: print "warning: Eg < {0}; check rho interpolate".format(Emid_rho[0])
        if Ex<=Emid_rho[0]: print "warning: at Eg = {0}: Ex <{1}; check rho interpolate".format(Eg, Emid_rho[0])
        norm += T(Eg) * rho(Ex) * (SpinDist(Ex,Jtarget+0.5) + SpinDist(Ex,Jtarget+1.5) )
      
    return norm * stepSize

  def GetNormFromGgD0(Gg, D0):
    # get the normaliation, see eg. eq (26) in Larsen2011
    return CalcIntegralSwave(Jtarget) * D0 * 1e3 / Gg  # /* Units = a1*D/G = MeV*eV*1e3/(MeV*meV) = 1 */

  b_norm = 1./GetNormFromGgD0(Gg, D0)
  T_norm = T_in * b_norm * np.exp(alpha_norm * Emid)
  gsf = T_norm / (2.*np.pi*Emid**3) # here: assume dipole radiation

  print "alpha_norm: {0}".format(alpha_norm)
  print "b_norm: {0}".format(b_norm)

  print np.exp(alpha_norm * Emid)

  print gsf
  return gsf
  
