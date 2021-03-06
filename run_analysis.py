from __future__ import division
import numpy as np

import json_tricks as json # can handle np arrays
from uncertainties import unumpy

import pyximport; pyximport.install() # compile each time for dev. purpose
import rhosig as rsg

import generateRhoT as gen
import normalization as norm
import standard_plots as splot

import utilities as ut

# Analysis of first generations matrix
# by Oslo Method

# running the program
interactive = True
makePlot = True

# try loading parameter file
try:
	with open("parameters.json", "r") as read_file:
		pars = json.load(read_file)
	print(pars)
except IOError:
	pars = dict()

# load 1Gen matrix data
data_folder = "synthetic_data/Jint_EB06_mama_4res/"
print("Use exp 1Gen matrix")
fname1Gen = data_folder+"/1Gen.m"
oslo_matrix, cal, Ex_array, Ex_array = ut.read_mama_2D(fname1Gen)
# some checkes
if len(Ex_array)!=len(Ex_array):
    raise ValueError("For now, require Ny = Nx, otherwise routines need to be  adjusted")
if np.any([cal["a0x"],cal["a1x"],cal["a2x"]]
          != [cal["a0y"],cal["a1y"],cal["a2y"]]):
    raise ValueError("For now, require xcal = ycal, otherwise energy arrays needs to be adjusted")
else:
    Emid = Ex_array
    # bin_width = cal["a1x"]

## Rebin and cut matrix
pars_fg = {"Egmin" : 1.0,
           "Exmin" : 4.0,
           "Emax" : 6.0}

oslo_matrix, Nbins, Emid = ut.rebin_both_axis(oslo_matrix, Emid, rebin_fac = 4)
oslo_matrix, Emid_Eg, Emid_Ex, Emid_nld = ut.fg_cut_matrix(oslo_matrix,
                                                        Emid, **pars_fg)

oslo_matrix_err = np.sqrt(oslo_matrix)

## decomposition of first gereration matrix P in NLD rho and transmission coefficient T
try:

    rho_fit = np.load("rho_fit.npy")
    T_fit = np.load("T_fit.npy")
    # hotfix: throw away uncertainty estimate in T
    T_fit = T_fit[:,0]
except:
    rho_fit, T_fit = \
        rsg.decompose_matrix_with_unc(P_in=oslo_matrix,
                                      P_err=oslo_matrix_err,
                                      Emid_Eg=Emid_Eg,
                                      Emid_nld=Emid_nld,
                                      Emid_Ex=Emid_Ex,
                                      N_mc = 10,
                                      options={"disp" : "True",
                                               "maxfev" : 500 })

    np.save("rho_fit.npy",rho_fit)
    np.save("T_fit.npy",T_fit)
    # hotfix: throw away uncertainty estimate in T
    T_fit = T_fit[:,0]

##############
# normalization of input matrix [for plotting later]
u_oslo_matrix = unumpy.uarray(oslo_matrix, np.sqrt(oslo_matrix))

# normalize each Ex row to 1 (-> get decay probability)
for i, normalization in enumerate(np.sum(u_oslo_matrix,axis=1)):
  u_oslo_matrix[i,:] /= normalization
oslo_matrix = unumpy.nominal_values(u_oslo_matrix)
oslo_matrix_err = unumpy.std_devs(u_oslo_matrix)
##############

splot.rsg_plots(rho_fit[:,0], T_fit, P_in=oslo_matrix, Emid_Eg=Emid_Eg, Emid_nld=Emid_nld, Emid_Ex = Emid_Ex, rho_true=None, **pars_fg)

## normalize the NLD

###
# # 2points
# pnld_norm = {}
# pnld_norm["nldE1"] = np.array([1.0,74]) # Mev, Mev^-1; higher norm point
# pnld_norm["nldE2"] = np.array([2.5,2.9e3]) # Mev, Mev^-1; higher norm point
# pnld_ext = {}
# pnld_ext["ext_range"] = np.array([2.5,7.]) # extrapolation range
# pnld_ext['T'] =  0.425
# pnld_ext['Eshift'] =  -0.456
###

###
# # find_norm
pnld_norm = {}
pnld_norm["E1_low"] = 0.45
pnld_norm["E2_low"] = 1.
pnld_norm["E1_high"] = 3.
pnld_norm["E2_high"] = 4.5
pnld_norm["nld_Sn"] = np.array([6.543,32e6])
pnld_ext = {} # automatically found for CT
pnld_ext["ext_range"] = np.array([2.5,7.]) # extrapolation range
###


# fake an uncertainty
# nld_err = rho_fit_err
# nld_err = rho_fit * 0.1
# nld_err[:10] *= 2
# rho_fit = np.c_[rho_fit,nld_err]

nldInst = norm.NormNLD(nld=np.c_[Emid_nld, rho_fit],
                       method="find_norm", pnorm=pnld_norm,
                       nldModel="CT", pext=pnld_ext)

# nldInst.find_norm()

rho_fit = nldInst.nld_norm
nld_ext = nldInst.nld_ext
A_norm = nldInst.A_norm
alpha_norm = nldInst.alpha_norm
discretes = nldInst.discretes

splot.nld_plot(rho_fit[:,1], T_fit, Emid_nld=Emid_nld, nld_ext=nld_ext, discretes=discretes)
# rsg_plots(rho_fit, T_fit, P_in=oslo_matrix, rho_true=rho_true, gsf_true=T_true)

## normalization of the gsf
# choose a spincut model and give it's parameters
spincutModel="EB05"
spincutPars={"mass":240, "NLDa":25.16, "Eshift":0.12} # some dummy values

# spincutModel="EB09_emp"
# spincutPars={"mass":56, "Pa_prime":2.905} # some dummy values

# input parameters:
# Emid_Eg, rho_in, T_in in MeV, MeV^-1, 1
# Jtarget in 1
# D0 in eV
# Gg in meV
# Sn in MeV
Jtarget = 1/2
D0 = 2.2 # eV
Gg = 43 # meV should this be updated? --> div by 2.3 due to RAINIER input model
Sn = 6.534 # work-around for now! -- until energy calibration is set!

# extrapolations
gsf_ext_range = np.array([0,3.,4., Sn+1])
# trans_ext_low, trans_ext_high = norm.trans_extrapolation(Emid_Eg, T_fit=T_fit,
#                                                    pars=pars, ext_range=ext_range,
#                                                    makePlot=makePlot, interactive=interactive)

############
###################################################################
# calculate the gsf
# assumption: Dipole transition only (therefore: E^(2L+1) -> E^3)
gsf_fit = T_fit/(2.*np.pi*pow(Emid_Eg,3.))

# assumptions in normalization: swave (currently); and equal parity
normMethod="standard" #-- like in normalization.c/Larsen2011 eq (26)
# normMethod="test" # -- test derived directly from Bartolomew
try:
    dim = rho_fit.shape[1]
    if dim == 3:
        rho_fit_err = rho_fit[:,2]
        rho_fit = rho_fit[:,1]
    elif dim == 2:
        rho_fit = rho_fit[:,1]
except IndexError:
        pass

# load "true" gsf
gsf_true_all = np.loadtxt(data_folder+"/GSFTable_py.dat")
gsf_true_tot = gsf_true_all[:,1] + gsf_true_all[:,2]
gsf_true = np.column_stack((gsf_true_all[:,0],gsf_true_tot))

normGSF = norm.NormGSF(gsf=np.c_[Emid_Eg,gsf_fit],
                       method=normMethod,
                       Jtarget=Jtarget, D0=D0, Gg=Gg, Sn=Sn,
                       alpha_norm=alpha_norm,
                       pext=pars, ext_range = gsf_ext_range,
                       spincutModel=spincutModel, spincutPars=spincutPars,
                       nld=np.c_[Emid_nld,rho_fit], nld_ext=nld_ext)

normGSF.normalizeGSF(gsf_referece = gsf_true,
                     makePlot=makePlot,
                     interactive=interactive)

gsf_fit = normGSF.gsf[:,1]
gsf_ext_low = normGSF.gsf_ext_low
gsf_ext_high = normGSF.gsf_ext_high
pars = normGSF.pext

T_fit = 2.*np.pi*gsf_fit*pow(Emid_Eg,3.) # for completenes, calculate this, too

splot.rsg_plots(rho_fit, T_fit, P_in=oslo_matrix, Emid_Eg=Emid_Eg, Emid_nld=Emid_nld, Emid_Ex = Emid_Ex, nld_ext=nld_ext, rho_true=None, discretes=discretes, **pars_fg)

# Comparison to "true" nld and gsf
def load_NLDtrue(fdisc=data_folder+"/NLD_exp_disc.dat", fcont=data_folder+"NLDcont.dat"):
    # load the known leveldensity from file
    NLD_true_disc = np.loadtxt(fdisc)
    NLD_true_cont = np.loadtxt(fcont)
    # apply same binwidth to continuum states
    binwidth_goal = NLD_true_disc[1,0]-NLD_true_disc[0,0]
    print(binwidth_goal)
    binwidth_cont = NLD_true_cont[1,0]-NLD_true_cont[0,0]
    Emax = NLD_true_cont[-1,0]
    nbins = int(np.ceil(Emax/binwidth_goal))
    Emax_adjusted = binwidth_goal*nbins # Trick to get an integer number of bins
    bins = np.linspace(0,Emax_adjusted,nbins+1)
    hist, edges = np.histogram(NLD_true_cont[:,0],bins=bins,weights=NLD_true_cont[:,1]*binwidth_cont)
    NLD_true = np.zeros((nbins,2))
    NLD_true[:nbins,0] = bins[:nbins]
    NLD_true[:,1] = hist/binwidth_goal
    NLD_true[:len(NLD_true_disc),1] += NLD_true_disc[:,1]
    return NLD_true, binwidth_goal

rho_true, rho_true_binwith = load_NLDtrue()

splot.normalized_plots(rho_fit, gsf_fit,
                 gsf_ext_low, gsf_ext_high, Emid_Eg=Emid_Eg,
                 Emid_nld=Emid_nld,
                 rho_true=rho_true, gsf_true=gsf_true, rho_true_binwidth=rho_true_binwith)

# save parameters to file
with open("parameters.json", "w") as write_file:
	json.dump(pars, write_file)
