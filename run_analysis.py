import numpy as np
import matplotlib.pyplot as plt
from StringIO import StringIO

import rhosig as rsg
import generateRhoT as gen
import normalization as norm

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
    if rho_true!=None: ax.plot(Emid,rho_true)
    ax.plot(Emid,rho_fit,"o")

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')

    plt.show()

def normalized_plots(rho_fit, gsf_fit, rho_true=None, gsf_true=None):
    # New Figure: compare input and output NLD and gsf
    f_mat, ax_mat = plt.subplots(2,1)

    # NLD
    ax = ax_mat[0]
    if rho_true!=None: ax.plot(Emid,rho_true)
    ax.plot(Emid,rho_fit,"o")

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

# only one of these can be set to true!
# UseExp1Gen = False
# UseSynthetic1Gen = True
UseExp1Gen = True
UseSynthetic1Gen = False

if UseExp1Gen:
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

    # if one wants to cut away some part of the matrix
    oslo_matrix = data[20:-30,20:-30]
    Emid = Emid[20:-30]

    Nbins = len(oslo_matrix) # hand adjusted
    Eup_max = Emid[-1] + bin_width/2. # upper bound of last bin
    pltbins = np.linspace(0,Eup_max,Nbins+1) # array of (start-bin?) values used for plotting

if UseSynthetic1Gen:
    print "Use synthetic 1Gen matrix"
    # Set bin width and range -- TODO: update for different #pltbins for Ex and Eg
    bin_width = 0.20
    Emin = 0.  # Minimum and maximum excitation                  -- CURRENTLY ARBITRARY
    Emax = 5.  # energy over which to extract strength function  -- CURRENTLY ARBITRARY
    Nbins = int(np.ceil(Emax/bin_width))
    # Nbins = len(oslo_matrix) # hand adjusted
    Emax_adjusted = bin_width*Nbins # Trick to get an integer number of pltbins
    pltbins = np.linspace(0,Emax_adjusted,Nbins+1)
    Emid = (pltbins[0:-1]+pltbins[1:])/2 # Array of middle-bin values, to use for plotting gsf

    rho_true, T_true, gsf_true= gen.generateRhoT(Emid)
    rho_true [3:5]=0. # simulte some artefacts
    oslo_matrix = rsg.PfromRhoT(rho_true,T_true)

# decomposition of first gereration matrix P in NLD rho and transmission coefficient T
# rho_fit, T_fit = rsg.decompose_matrix(P_in=oslo_matrix, Emid=Emid, fill_value=1e-11)
rho_fit, T_fit = rsg.decompose_matrix(P_in=oslo_matrix, Emid=Emid, fill_value=1e-1)
print "decomposed matrix to rho and T"

# normalize the NLD
# nldE1 = np.array([3.,282.]) # Mev, Mev^-1; higher normalization point
# nldE2 = np.array([11.197,2.18]) # Mev, Mev^-1; higher normalization point
nldE1 = np.array([3.24,7.]) # Mev, Mev^-1; higher normalization point
nldE2 = np.array([11.197,2.18e3]) # Mev, Mev^-1; higher normalization point
rho_fit, alpha_norm, A_norm = norm.normalizeNLD(nldE1[0], nldE1[1], nldE2[0], nldE2[1], Emid=Emid, rho=rho_fit)

rsg_plots(rho_fit, T_fit, P_in=oslo_matrix, rho_true=None)
# rsg_plots(rho_fit, T_fit, P_in=oslo_matrix, rho_true=rho_true, gsf_true=T_true)

# normalization of the gsf
# choose a spincut model and give it's parameters
# spincutModel="EB05"
# spincutPars={"mass":120, "NLDa":2.4, "Eshift":-0.7} # some dummy values

spincutModel="EB09_emp"
spincutPars={"mass":56, "Pa_prime":2.905} # some dummy values

# input parameters:
# Emid, rho_in, T_in in MeV, MeV^-1, 1
# Jtarget in 1
# D0 in eV
# Gg in meV
# Sn in MeV

D0 = 3.36e3 # eV 
Gg = 1900/2.3 # meV --> div by 2.3 due to RAINIER input model
Sn = Emid[-1] # work-around for now! -- until energy calibration is set!

gsf_fit = norm.normalizeGSF(Emid=Emid, rho_in=rho_fit, T_in=T_fit, Jtarget=0, D0=D0, Gg=Gg, Sn=Sn, alpha_norm=alpha_norm, spincutModel=spincutModel, spincutPars=spincutPars)
normalized_plots(rho_fit, gsf_fit)
# normalized_plots(rho_fit, gsf_fit, rho_true=rho_true, gsf_true=gsf_true)