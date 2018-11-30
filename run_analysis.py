import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import json_tricks as json # can handle np arrays
from uncertainties import unumpy

import pyximport; pyximport.install() # compile each time for dev. purpose
import rhosig as rsg

import generateRhoT as gen
import normalization as norm

import utilities as ut

# Analysis of first generations matrix
# by Oslo Method

def rsg_plots(rho_fit, T_fit, P_in, nld_ext=None, rho_true=None):
	# creates
	# gsf_fit = T_fit/pow(Emid,3)  # assuming dipoles only
	# New Figure: Oslo type matrix
	f_mat, ax_mat = plt.subplots(2,1)

	# input matrix
	ax = ax_mat[0]
	from matplotlib.colors import LogNorm # To get log scaling on the z axis
	norm=LogNorm(vmin=1e-5, vmax=P_in.max())
	colorbar_object = ax.pcolormesh(pltbins_Eg, pltbins_Ex, P_in, norm=norm)
	f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

	ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
	ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

	# fitted matrix
	ax = ax_mat[1]
	Nbins_Ex, Nbins_T = np.shape(P_in)
	P_fit = rsg.PfromRhoT(rho_fit,T_fit, Nbins_Ex, Emid, Emid_rho, Emid_Ex)

	from matplotlib.colors import LogNorm # To get log scaling on the z axis
	colorbar_object = ax.pcolormesh(pltbins_Eg, pltbins_Ex, P_fit, norm=norm)
	f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

	ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
	ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

	# New Figure: compare input and output NLD
	f_mat, ax = plt.subplots(1,1)

	# NLD
	if nld_ext is not None: ax.plot(nld_ext[:,0],nld_ext[:,1],"b--")
	if rho_true is not None: ax.plot(Emid_rho,rho_true)
	ax.plot(Emid_rho,rho_fit,"o")

	ax.set_yscale('log')
	ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
	ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')

	plt.show()

def normalized_plots(rho_fit, gsf_fit, gsf_ext_low, gsf_ext_high, rho_true=None, rho_true_binwidth=None, gsf_true=None):
	# New Figure: compare input and output NLD and gsf
	f_mat, ax_mat = plt.subplots(2,1)

	# NLD
	ax = ax_mat[0]
	if rho_true is not None: ax.step(np.append(-rho_true_binwidth,rho_true[:-1,0])+rho_true_binwidth/2.,np.append(0,rho_true[:-1,1]), "k", where="pre",label="input NLD, binned")
	ax.plot(Emid_rho,rho_fit,"o")

	ax.set_yscale('log')
	ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
	ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')

	# gsf
	ax = ax_mat[1]
	if gsf_true is not None: ax.plot(gsf_true[:,0],gsf_true[:,1])
	ax.plot(Emid,gsf_fit,"o")
	[gsf_ext_high_plt] = ax.plot(gsf_ext_high[:,0],gsf_ext_high[:,1],"r--", label="ext. high")
	[gsf_ext_low_plt] = ax.plot(gsf_ext_low[:,0],gsf_ext_low[:,1],"b--", label="ext. high")

	ax.set_yscale('log')
	ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
	ax.set_ylabel(r'$gsf \, \mathrm{(MeV**(-3)}$')

	plt.show()

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

print("Use exp 1Gen matrix")
# load experimential data
fname1Gen = "1Gen.m"
oslo_matrix, cal, Ex_array, Ex_array = ut.read_mama_2D(fname1Gen)

# some checkes
if len(Ex_array)!=len(Ex_array):
    raise ValueError("For now, require Ny = Nx, otherwise routines need to be  adjusted")
if np.any([cal["a0x"],cal["a1x"],cal["a2x"]]
          != [cal["a0y"],cal["a1y"],cal["a2y"]]):
    raise ValueError("For now, require xcal = ycal, otherwise energy arrays needs to be adjusted")
else:
    Emid = Ex_array
    bin_width = cal["a1x"]

## Rebin and cut matrix
# cut array along ex and eg min/max
Egmin = 1.0
Exmin = 2.0
Emax = 5.0
# Rebin factor
Nbins = len(oslo_matrix) # before
rebin_fac = 4.
Nbins_final = int(Nbins/rebin_fac)
rebin_fac = Nbins/float(Nbins_final)

oslo_matrix, Emid_ = ut.rebin_and_shift(oslo_matrix, Emid, Nbins_final, rebin_axis=0) # rebin x axis
oslo_matrix, Emid_ = ut.rebin_and_shift(oslo_matrix, Emid, Nbins_final, rebin_axis=1) # rebin y axis
Emid = Emid_
bin_width *= rebin_fac
Nbins = Nbins_final
# Eg
i_Egmin = (np.abs(Emid-Egmin)).argmin()
i_Emax = (np.abs(Emid-Emax)).argmin()
# Ex
i_Exmin = (np.abs(Emid-Exmin)).argmin()

oslo_matrix = oslo_matrix[i_Exmin:i_Emax,i_Egmin:i_Emax]
Emid_Ex = Emid[i_Exmin:i_Emax]
Emid_rho = Emid[:i_Emax-i_Egmin]
Emid = Emid[i_Egmin:i_Emax]
print(np.shape(oslo_matrix),np.shape(Emid),np.shape(Emid_rho))

# def myfunc(N): return np.random.normal(N,np.sqrt(N))
# myfunc_vec = np.vectorize(myfunc)
# oslo_matrix=myfunc_vec(oslo_matrix)
# approximate uncertainty my sqrt of number of counts
u_oslo_matrix = unumpy.uarray(oslo_matrix, np.sqrt(oslo_matrix))

# normalize each Ex row to 1 (-> get decay probability)
for i, normalization in enumerate(np.sum(u_oslo_matrix,axis=1)):
	u_oslo_matrix[i,:] /= normalization
oslo_matrix = unumpy.nominal_values(u_oslo_matrix)
oslo_matrix_err = unumpy.std_devs(u_oslo_matrix)

Nbins_Ex, Nbins_Eg= np.shape(oslo_matrix) # after
Eup_max = Exmin + Nbins_Ex * bin_width # upper bound of last bin
pltbins_Ex = np.linspace(Exmin,Eup_max,Nbins_Ex+1) # array of (start-bin?) values used for plotting
Eup_max = Egmin + Nbins_Eg * bin_width # upper bound of last bin
pltbins_Eg = np.linspace(Egmin,Eup_max,Nbins_Eg+1) # array of (start-bin?) values used for plotting

print(oslo_matrix.shape)
## decomposition of first gereration matrix P in NLD rho and transmission coefficient T
rho_fit, T_fit = rsg.decompose_matrix(P_in=oslo_matrix, P_err=oslo_matrix_err, Emid=Emid, Emid_rho=Emid_rho, Emid_Ex=Emid_Ex, fill_value=1e-1)
print(rho_fit, T_fit)

## normalize the NLD
nldE1 = np.array([1.0,74]) # Mev, Mev^-1; higher normalization point
nldE2 = np.array([2.5,2.9e3]) # Mev, Mev^-1; higher normalization point

def load_NLDtrue(fdisc="compare/240Pu/NLD_exp_disc.dat", fcont="compare/240Pu/NLD_exp_cont.dat"):
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

# normalize
rho_fit, alpha_norm, A_norm = norm.normalizeNLD(nldE1[0], nldE1[1], nldE2[0], nldE2[1], Emid_rho=Emid_rho, rho=rho_fit)

# extrapolation
ext_range = np.array([2.5,7.])
nldPars = dict()
# find parameters
nldModel="CT"
key = 'nld_CT'
# if key not in pars:
pars['nld_CT']= np.array([0.55,-1.3])

nldPars['T'], nldPars['Eshift'] = pars['nld_CT']

# extrapolations of the gsf
Emid_ = np.linspace(ext_range[0],ext_range[1])
nld_ext = norm.nld_extrapolation(Emid_,
							 nldModel=nldModel, nldPars=nldPars,
							 makePlot=makePlot)


rsg_plots(rho_fit, T_fit, P_in=oslo_matrix, nld_ext=nld_ext, rho_true=None)
# rsg_plots(rho_fit, T_fit, P_in=oslo_matrix, rho_true=rho_true, gsf_true=T_true)

## normalization of the gsf
# choose a spincut model and give it's parameters
spincutModel="EB05"
spincutPars={"mass":240, "NLDa":25.16, "Eshift":0.12} # some dummy values

# spincutModel="EB09_emp"
# spincutPars={"mass":56, "Pa_prime":2.905} # some dummy values

# input parameters:
# Emid, rho_in, T_in in MeV, MeV^-1, 1
# Jtarget in 1
# D0 in eV
# Gg in meV
# Sn in MeV
Jtarget = 1/2
D0 = 2.2 # eV
Gg = 34. # meV --> div by 2.3 due to RAINIER input model
Sn = 6.534 # work-around for now! -- until energy calibration is set!

# extrapolations
gsf_ext_range = np.array([0,3.,4., Sn+1])
# trans_ext_low, trans_ext_high = norm.trans_extrapolation(Emid, T_fit=T_fit,
#                                                    pars=pars, ext_range=ext_range,
#                                                    makePlot=makePlot, interactive=interactive)

############
###################################################################
# calculate the gsf
# assumption: Dipole transition only (therefore: E^(2L+1) -> E^3)
gsf_fit = T_fit/(2*np.pi*pow(Emid,3.))

# assumptions in normalization: swave (currently); and equal parity
normMethod="standard" #-- like in normalization.c/Larsen2011 eq (26)
# normMethod="test" # -- test derived directly from Bartolomew
gsf_fit, b_norm, gsf_ext_low, gsf_ext_high = norm.normalizeGSF(Emid=Emid, Emid_rho=Emid_rho, rho_in=rho_fit, gsf_in=gsf_fit,
															   nld_ext = nld_ext,
															   gsf_ext_range=gsf_ext_range, pars=pars,
															   Jtarget=Jtarget, D0=D0, Gg=Gg, Sn=Sn, alpha_norm=alpha_norm,
															   normMethod=normMethod,
															   spincutModel=spincutModel, spincutPars=spincutPars,
															   makePlot=makePlot, interactive=interactive)
T_fit = 2*np.pi*gsf_fit*pow(Emid,3.) # for completenes, calculate this, too

# # load "true" gsf
# gsf_true_all = np.loadtxt("compare/240Pu/GSFTable_py.dat")
# gsf_true_tot = gsf_true_all[:,1] + gsf_true_all[:,2]
# gsf_true = np.column_stack((gsf_true_all[:,0],gsf_true_tot))

# normalized_plots(rho_fit, gsf_fit,
#                  gsf_ext_low, gsf_ext_high,
#                  rho_true=rho_true, gsf_true=gsf_true, rho_true_binwidth=rho_true_binwith)
# normalized_plots(rho_fit, gsf_fit, rho_true=rho_true, gsf_true=gsf_true)

# # save parameters to file
with open("parameters.json", "w") as write_file:
	json.dump(pars, write_file)
