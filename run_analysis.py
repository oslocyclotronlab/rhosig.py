import numpy as np
import rhosig as rsg
import matplotlib.pyplot as plt

# Analysis of first generations matrix
# by Oslo Method

def rsg_plots(rho_fit, T_fit, P_in, rho_true=None, gsf_true=None):
	# creates 
    gsf_fit = T_fit/pow(Emid,3)  # assuming dipoles only
    # New Figure: Oslo type matrix
    f_mat, ax_mat = plt.subplots(2,1)

    # input matrix
    ax = ax_mat[0]
    from matplotlib.colors import LogNorm # To get log scaling on the z axis
    colorbar_object = ax.pcolormesh(bins, bins, P_in, norm=LogNorm())
    f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

    # fitted matrix
    ax = ax_mat[1]
    P_fit = rsg.PfromRhoT(rho_fit,T_fit)

    from matplotlib.colors import LogNorm # To get log scaling on the z axis
    colorbar_object = ax.pcolormesh(bins, bins, P_fit, norm=LogNorm())
    f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

    # New Figure: compare input and output NLD and gsf
    f_mat, ax_mat = plt.subplots(2,1)

    # NLD
    ax = ax_mat[0]
    ax.plot(Emid,rho_fit,"o")
    if rho_true: ax.plot(Emid,rho_true)

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')

    # gsf
    ax = ax_mat[1]
    ax.plot(Emid,gsf_fit,"o")
    if gsf_true: ax.plot(Emid,gsf_true)

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$gsf \, \mathrm{(MeV**(-3)}$')

    plt.show()

# running the program

# load experimential data
data = np.loadtxt("1Gen.m", comments="!")
print "loaded data"

# select fit region by hand
# important: for now, need to adjust Nbins further down!
oslo_matrix = data[20:-20,20:-20] 

# Set bin width and range -- TODO: update for different #bins for Ex and Eg
bin_width = 0.20
Emin = 0.  # Minimum and maximum excitation                  -- CURRENTLY ARBITRARY
Emax = 5.  # energy over which to extract strength function  -- CURRENTLY ARBITRARY
Nbins = len(oslo_matrix) # hand adjusted
Emax_adjusted = bin_width*Nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,Nbins+1)
Emid = (bins[0:-1]+bins[1:])/2 # Array of middle-bin values, to use for plotting gsf

# decomposition of first gereration matrix P in NLD rho and transmission coefficient T
rho_fit, T_fit = rsg.decompose_matrix(P_in=oslo_matrix, Emid=Emid)
print "decomposed matrix to rho and T"

rsg_plots(rho_fit, T_fit, P_in=oslo_matrix, rho_true=None, gsf_true=None)