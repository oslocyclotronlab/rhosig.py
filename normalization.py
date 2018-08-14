import numpy as np
from scipy.interpolate import interp1d
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# normalization of NLD and GSF with the Oslo method

# Normalization of the NLD
def normalizeNLD(E1, nldE1, E2, nldE2, Emid_rho, rho):
  # normalization of the NLD according to the transformation eq (3), Schiller2000
  # iputs: unnormalized nld rho, and their mid-energy bins Emid
  #        E1(2) and nldE1(2): normalization points: Energy and NLD

  # find corresponding energy bins of E1 and E2
  i_E1 = (np.abs(Emid_rho-E1)).argmin()
  i_E2 = (np.abs(Emid_rho-E2)).argmin()
  print(i_E1, i_E2, Emid_rho[i_E1], Emid_rho[i_E2])

  # find alpha and A from the normalization points
  alpha = np.log( (nldE2 * rho[i_E1]) / (nldE1*rho[i_E2]) ) / (E2-E1)
  A = nldE2 / rho[i_E2] * np.exp(- alpha * E2)
  print(A)
  print("Normalization parameters: \n alpha={0:1.2e} \t A={1:1.2e} ".format(alpha, A))

  # apply the transformation
  rho *= A * np.exp(alpha*Emid_rho)
  return rho, alpha, A

# extrapolations of the gsf
def nld_extrapolation(Ex, nldModel, nldPars={},
                      makePlot=True):

    def GetExtrapolation(Ex, model=nldModel, pars=nldPars):
      # Get Extrapolation values

      # different extrapolation models
      def CT(T, Eshift): 
      # constant temperature
        return np.exp((Ex-Eshift) / T) / T;

      # call a model
      def CallModel(fun,pars,pars_req):
        if pars_req <= set(pars): # is the required parameters are a subset of all pars given
            return fun(**pars)
        else:
            raise TypeError("Error: Need following arguments for this method: {0}".format(pars_req))

      if model=="CT": 
        pars_req = {"T", "Eshift"}
        return CallModel(CT,pars,pars_req)
      else:
        raise TypeError("\nError: NLD model not supported; check spelling\n")
        return 1.

    nld_ext = GetExtrapolation(Ex)
    return np.column_stack((Ex,nld_ext))


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

class SpinFunctions:
  def GetSigma2(self, Ex, J, model, pars):
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

  def SpinDist(self, Ex, J, model, pars):
    # Get Spin distribution given a spin-cut sigma
    # note: input is sigma2
    # assuming equal parity

    sigma2 = self.GetSigma2(Ex, J, model=model, pars=pars)
    # following Gilbert1965, eq (E4)
    return (2.*J+1.) /(2.*sigma2) * np.exp(-np.power(J+0.5, 2.) / (2.*sigma2))


def transformGSF(Emid, Emid_rho, rho_in, gsf_in, 
                 nld_ext,
                 gsf_ext_low, gsf_ext_high,
                 Jtarget, D0, Gg, Sn, alpha_norm, spincutModel, spincutPars={}):
  # transform the gsf extracted with the Oslo method
  # to the average total radiative width <Gg>
  # returns normalized GSF (L=1) from an input gamma-ray strength function gsf
  # inputs:
    # Emid, rho_in, gsf in MeV, MeV^-1, MeV^-3 -- -- important: gsf needs to be "shape corrected" by alpha_norm
    # nld_ext: extrapolation of nld
    # gsf_ext_low, high: extrapolations of gsf -- important: need to be "shape corrected" bs alpha_norm
    # Jtarget in 1
    # D0 in eV
    # Gg in meV
    # Sn in MeV
  # assuming dipole radiation
      # interpolate NLD and T

  def SpinDist(Ex, J):
    return SpinFunctions().SpinDist(Ex=Ex, J=J, model=spincutModel, pars=spincutPars)

  def GetNormFromGgD0(Gg, D0, Jtarget):
    # get the normaliation, see eg. eq (26) in Larsen2011; but converted T to gsf

    # compose nld and gsf function of data & extrapolation
    frho_exp = interp1d(Emid_rho,rho_in) # defualt: linear interpolation
    fnld_ext = interp1d(nld_ext[:,0], nld_ext[:,1])  # default:linear interpolation

    fgsf_exp   = interp1d(Emid, gsf_in)  # default:linear interpolation
    fgsf_ext_low = interp1d(gsf_ext_low[:,0], gsf_ext_low[:,1])  # default:linear interpolation
    fgsf_ext_high = interp1d(gsf_ext_high[:,0], gsf_ext_high[:,1])  # default:linear interpolation

    # extapolate "around" dataset 
    def frho(E):
      if E <= Emid_rho[-1]:
        val = frho_exp(E)
      else:
        val = fnld_ext(E)
      return val

    def fgsf(E):
      if E < Emid[0]:
        val = fgsf_ext_low(E)
      elif E <= Emid[-1]:
        val = fgsf_exp(E)
      else:
        val = fgsf_ext_high(E)
      return val

    # calculate integral 
    Eint_min = 0
    Eint_max = Sn # Sn + Exres # following routine by Magne; Exres is the energy resolution (of SiRi)
    Eintegral, stepSize = np.linspace(Eint_min,Eint_max,num=100, retstep=True) # number of interpolation points 

    # equalts "old" (normalization.f) version in the Spin sum
    # get the normaliation, see eg. eq (26) in Larsen2011; but converted T to gsf
    # further assumptions: s-wave (currently)
    def SpinSum(Jtarget): 
      if Jtarget == 0: #  if(Jtarget == 0.0)      /*I_i = 1/2 => I_f = 1/2, 3/2 */
        return SpinDist(Ex,Jtarget+0.5) + SpinDist(Ex,Jtarget+1.5)
      elif Jtarget == 1/2: #  if(Jtarget == 0.5)      /*I_i = 0, 1  => I_f = 0, 1, 2 */
        return SpinDist(Ex,Jtarget-0.5) + 2*SpinDist(Ex,Jtarget+0.5) + SpinDist(Ex,Jtarget+1.5)
      elif Jtarget == 1: #  if(Jtarget == 0.5)      /*I_i = 1/2, 3/2  => I_f = 1/2, 3/2, 5/2 */
        return 2*SpinDist(Ex,Jtarget-0.5) + 2*SpinDist(Ex,Jtarget+0.5) + SpinDist(Ex,Jtarget+1.5)
      elif Jtarget > 1: #  if(Jtarget == 0.5)      /*I_i = 1/2, 3/2  => I_f = 1/2, 3/2, 5/2 */
        return SpinDist(Ex,Jtarget-1.5) + 2*SpinDist(Ex,Jtarget-0.5) + 2*SpinDist(Ex,Jtarget+0.5) + SpinDist(Ex,Jtarget+1.5)
      else:
        ValueError("Negative J not supported")

    integral = 0
    for Eg in Eintegral:
        Ex = Sn - Eg
        if Eg<=Emid_rho[0]: print("warning: Eg < {0}; check rho interpolate".format(Emid_rho[0]))
        if Ex<=Emid_rho[0]: print("warning: at Eg = {0}: Ex <{1}; check rho interpolate".format(Eg, Emid_rho[0]))
        integral += np.power(Eg,3) * fgsf(Eg) * frho(Ex) * SpinSum(Jtarget)
    integral *= stepSize

    return 2 * Gg / ( integral * D0*1e3)  # /* Units = G/ (D) = meV / (eV*1e3) = 1 */

  b_norm = GetNormFromGgD0(Gg, D0, Jtarget)
  gsf_norm = gsf_in * b_norm

  gsf_ext_low_norm = gsf_ext_low
  gsf_ext_high_norm = gsf_ext_high
  gsf_ext_low_norm[:,1] *= b_norm
  gsf_ext_high_norm[:,1] *= b_norm

  print("alpha_norm: {0}".format(alpha_norm))
  print("b_norm: {0}".format(b_norm))

  return gsf_norm, gsf_ext_low_norm, gsf_ext_high_norm, b_norm
  

def normalizeGSF(Emid, Emid_rho, rho_in, gsf_in, 
                 nld_ext,
                 gsf_ext_range, pars,
                 Jtarget, D0, Gg, Sn, alpha_norm,
                 makePlot, interactive,
                 spincutModel, spincutPars={}):
  # normalize the gsf extracted with the Oslo method
  # to the average total radiative width <Gg>
  # returns normalized GSF (L=1) from an input gamma-ray strength function gsf
  # inputs:
    # Emid, rho_in, gsf in MeV, MeV^-1, MeV^-3
    # nld_ext: extrapolation of nld
    # gsf_ext_range: extrapolation ranges of gsf
    # Jtarget in 1
    # D0 in eV
    # Gg in meV
    # Sn in MeV
  # assuming dipole radiation
      # interpolate NLD and T


  #   def trans_extrapolation(Emid, gsf_fit, pars, ext_range, makePlot, interactive):
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
  gsf_shape = gsf_in * np.exp(alpha_norm * Emid) # "shape" - correction of the transformation

  gsf_ext_low, gsf_ext_high = gsf_extrapolation(pars, gsf_ext_range)
  gsf, gsf_ext_lowa, gsf_ext_higha, b_norm =  transformGSF(Emid=Emid, Emid_rho=Emid_rho, rho_in=rho_in, gsf_in=gsf_shape, 
                                                       nld_ext=nld_ext,
                                                       gsf_ext_low=gsf_ext_low, gsf_ext_high=gsf_ext_high,
                                                       Jtarget=Jtarget, D0=D0, Gg=Gg, Sn=Sn, 
                                                       alpha_norm=alpha_norm, 
                                                       spincutModel=spincutModel, spincutPars=spincutPars)

  if makePlot:
      fig, ax = plt.subplots()
      plt.subplots_adjust(left=0.25, bottom=0.35)

      # gsf
      [gsf_plot]=ax.plot(Emid,gsf,"o")
      [gsf_ext_high_plt] = ax.plot(gsf_ext_higha[:,0],gsf_ext_higha[:,1],"r--", label="ext. high")
      [gsf_ext_low_plt] = ax.plot(gsf_ext_lowa[:,0],gsf_ext_lowa[:,1],"b--", label="ext. high")

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
          ext_a_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03], axisbg=axis_color)
          ext_b_slider_ax  = fig.add_axes([0.25, 0.10, 0.65, 0.03], axisbg=axis_color)
          ext_c_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axis_color)
          ext_d_slider_ax  = fig.add_axes([0.25, 0.20, 0.65, 0.03], axisbg=axis_color)

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
              gsf_ext_low1, gsf_ext_high1 = gsf_extrapolation(pars, gsf_ext_range)
              gsf, gsf_ext_lowa, gsf_ext_higha, b_norm =  transformGSF(Emid=Emid, Emid_rho=Emid_rho, rho_in=rho_in, gsf_in=gsf_shape, 
                                                                   nld_ext=nld_ext,
                                                                   gsf_ext_low=gsf_ext_low1, gsf_ext_high=gsf_ext_high1,
                                                                   Jtarget=Jtarget, D0=D0, Gg=Gg, Sn=Sn, 
                                                                   alpha_norm=alpha_norm, 
                                                                   spincutModel=spincutModel, spincutPars=spincutPars)
              gsf_plot.set_ydata(gsf)
              gsf_ext_high_plt.set_ydata(gsf_ext_higha[:,1])
              gsf_ext_low_plt.set_ydata(gsf_ext_lowa[:,1])
              fig.canvas.draw_idle()
          sext_a.on_changed(slider_update)
          sext_b.on_changed(slider_update)
          sext_c.on_changed(slider_update)
          sext_d.on_changed(slider_update)

          resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
          button = Button(resetax, 'Reset', color=axis_color, hovercolor='0.975')

          def reset(event):
              sext_a.reset()
              sext_b.reset()
              sext_c.reset()
              sext_d.reset()
          button.on_clicked(reset)
      plt.show()
      # repead this, to get the values from the end of the plot
      # ext_a, ext_b =  pars['gsf_ext_high']
      # ext_c, ext_d =  pars['gsf_ext_low']
      # gsf_ext_low = np.column_stack((Emid_ext_low, f_gsf_ext_low(Emid_ext_low, ext_c, ext_d)))
      # gsf_ext_high = np.column_stack((Emid_ext_high, f_gsf_ext_high(Emid_ext_high, ext_a, ext_b)))


  return gsf, b_norm, gsf_ext_low, gsf_ext_high