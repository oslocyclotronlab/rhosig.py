import numpy as np
import matplotlib.pyplot as plt
from StringIO import StringIO

import rhosig as rsg
import generateRhoT as gen
import normalization as norm

import utilities as ut
# Analysis of first generations matrix
# by Oslo Method

def rsg_plots(rho_fit, T_fit, P_in, rho_true=None):
	# creates 
    # gsf_fit = T_fit/pow(Emid,3)  # assuming dipoles only
    # New Figure: Oslo type matrix
    f_mat, ax_mat = plt.subplots(2,1)

    # input matrix
    ax = ax_mat[0]
    from matplotlib.colors import LogNorm # To get log scaling on the z axis
    colorbar_object = ax.pcolormesh(pltbins, pltbins, P_in, norm=LogNorm())
    f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

    # fitted matrix
    ax = ax_mat[1]
    P_fit = rsg.PfromRhoT(rho_fit,T_fit)

    from matplotlib.colors import LogNorm # To get log scaling on the z axis
    colorbar_object = ax.pcolormesh(pltbins, pltbins, P_fit, norm=LogNorm())
    f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

    # New Figure: compare input and output NLD
    f_mat, ax = plt.subplots(1,1)

    # NLD
    if rho_true!=None: ax.plot(Emid_rho,rho_true)
    ax.plot(Emid_rho,rho_fit,"o")

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')

    plt.show()

def normalized_plots(rho_fit, gsf_fit, rho_true=None, rho_true_binwidth=None, gsf_true=None):
    # New Figure: compare input and output NLD and gsf
    f_mat, ax_mat = plt.subplots(2,1)

    # NLD
    ax = ax_mat[0]
    if rho_true!=None: ax.step(np.append(-rho_true_binwidth,rho_true[:-1,0])+rho_true_binwidth/2.,np.append(0,rho_true[:-1,1]), "k", where="pre",label="input NLD, binned")
    ax.plot(Emid_rho,rho_fit,"o")

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')

    # gsf
    ax = ax_mat[1]
    if gsf_true!=None: ax.plot(Emid,gsf_true)
    ax.plot(Emid,gsf_fit,"o")

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$gsf \, \mathrm{(MeV**(-3)}$')

    plt.show()

# running the program

print "Use exp 1Gen matrix"
# load experimential data
fname1Gen = "1Gen.m"
data = np.loadtxt(fname1Gen, comments="!")
print "loaded data"

# select fit region by hand
# important: for now, need to adjust Nbins further down!
oslo_matrix = data

# read pltbins/calibration from the mamafile
def EmidsFromMama(fname):
    # returns middle-bin values of the energy pltbins from a mama matrix
    # energies in MeV

    # calibration coefficients in MeV
    # cal0 + cal1*ch + cal2*ch**2 for x and y
    f = open(fname)
    lines = f.readlines()
    print lines[6]
    cal = np.genfromtxt(StringIO(lines[6]),dtype=object, delimiter=",")
    if cal[0]!="!CALIBRATION EkeV=6":
        raise ValueError("Could not read calibration")
    cal = cal[1:].astype("float64")
    cal *= 1e-3 # keV -> MeV
    xcal = cal[:3]
    ycal = cal[3:]
    print "Calibration read from mama: \n xcal{0} \t ycal {1}".format(xcal, ycal)
    # workaround until implemented otherwise
    if np.any(xcal!=ycal):
        raise ValueError("For now, require xcal = ycal, otherwise Emid needs to be adjusted")
    
    # read channel numbers
    print lines[8]
    line = lines[8].replace(b':',b',') # replace ":"" as delimiter by ","
    nChs = np.genfromtxt(StringIO(line),dtype=object, delimiter=",")
    print "Channels read from mama: \n nChx{0} \t nChy {1}".format(nChs[1:3], nChs[3:])
    if nChs[0]!="!DIMENSION=2":
        raise ValueError("Could not read calibration")
    nChs = nChs[1:].astype("int")
    if nChs[0]!=0 or nChs[2]!=0:
        raise ValueError("Not your day: First channel is not ch0")
    nChx = nChs[1]+1
    nChy = nChs[3]+1
    if nChx!=nChy:
        raise ValueError("For now, require nChx = nChy, otherwise Emid needs to be adjusted")

    Emid = np.array(range(nChx)) # Emid = [0, 1, 2,..., nChx-1]
    Emid = xcal[0] + xcal[1] * Emid + xcal[2] * Emid**2

    if xcal[0]!=0 or xcal[0]!=0 or ycal[0]!=0 or ycal[0]!=0:
        raise ValueError("Ohoh: Variable binzise not yet implemented")
    bin_width = Emid[1]-Emid[0]

    return Emid, bin_width

Emid, bin_width = EmidsFromMama(fname1Gen)

## Rebin and cut matrix
# cut array along ex and eg min/max
Emin = 2.0
Emax = 5.0
# Rebin factor
Nbins = len(oslo_matrix) # before
rebin_fac = 4.
Nbins_final = int(Nbins/rebin_fac)

oslo_matrix, Emid_ = ut.rebin_and_shift(oslo_matrix, Emid, Nbins_final, rebin_axis=0) # rebin x axis
oslo_matrix, Emid_ = ut.rebin_and_shift(oslo_matrix, Emid, Nbins_final, rebin_axis=1) # rebin y axis
Emid = Emid_
i_Emin = (np.abs(Emid-Emin)).argmin()
i_Emax = (np.abs(Emid-Emax)).argmin()
oslo_matrix = oslo_matrix[i_Emin:i_Emax,i_Emin:i_Emax]
Emid_rho = Emid[:i_Emax-i_Emin]
Emid = Emid[i_Emin:i_Emax]

Nbins = len(oslo_matrix) # after
Eup_max = Emid[-1] + bin_width/2. # upper bound of last bin
pltbins = np.linspace(0,Eup_max,Nbins+1) # array of (start-bin?) values used for plotting

## decomposition of first gereration matrix P in NLD rho and transmission coefficient T
rho_fit, T_fit = rsg.decompose_matrix(P_in=oslo_matrix, Emid=Emid, fill_value=1e-1)

## normalize the NLD
nldE1 = np.array([1.0,74]) # Mev, Mev^-1; higher normalization point
nldE2 = np.array([2.5,2.9e3]) # Mev, Mev^-1; higher normalization point

def load_NLDtrue(fdisc="compare/240Pu/NLD_exp_disc.dat", fcont="compare/240Pu/NLD_exp_cont.dat"):
    # load the known leveldensity from file
    NLD_true_disc = np.loadtxt(fdisc)
    NLD_true_cont = np.loadtxt(fcont)
    # apply same binwidth to continuum states
    binwidth_goal = NLD_true_disc[1,0]-NLD_true_disc[0,0]
    print binwidth_goal
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

rsg_plots(rho_fit, T_fit, P_in=oslo_matrix, rho_true=None)
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
D0 = 3.36e3 # eV 
Gg = 1900/2.3 # meV --> div by 2.3 due to RAINIER input model
Sn = 6.534 # work-around for now! -- until energy calibration is set!

gsf_fit = norm.normalizeGSF(Emid=Emid, Emid_rho=Emid_rho, rho_in=rho_fit, T_in=T_fit, Jtarget=0, D0=D0, Gg=Gg, Sn=Sn, alpha_norm=alpha_norm, spincutModel=spincutModel, spincutPars=spincutPars)
T_fit = gsf_fit*pow(Emid,3.)

normalized_plots(rho_fit, gsf_fit, rho_true=rho_true, rho_true_binwidth=rho_true_binwith)
# normalized_plots(rho_fit, gsf_fit, rho_true=rho_true, gsf_true=gsf_true)

rsg_plots(rho_fit, gsf_fit/pow(Emid,3.), P_in=oslo_matrix, rho_true=None)