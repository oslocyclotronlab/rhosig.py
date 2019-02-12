import numpy as np
import matplotlib.pyplot as plt
import rhosig as rsg

def rsg_plots(rho_fit, T_fit, P_in, Emid_Eg, Emid_nld, Emid_Ex, Exmin, Emax, Egmin, rho_true=None, **kwargs):
    Nbins_Ex, Nbins_Eg= np.shape(P_in) # after
    bin_width = Emid_Eg[1]- Emid_Eg[0]
    Eup_max = Exmin + Nbins_Ex * bin_width # upper bound of last bin
    pltbins_Ex = np.linspace(Exmin,Eup_max,Nbins_Ex+1) # array of (start-bin?) values used for plotting
    Eup_max = Egmin + Nbins_Eg * bin_width # upper bound of last bin
    pltbins_Eg = np.linspace(Egmin,Eup_max,Nbins_Eg+1) # array of (start-bin?) values used for plotting

    try:
        dim = rho_fit.shape[1]
        if dim == 3:
            rho_fit_err = rho_fit[:,2]
            rho_fit = rho_fit[:,1]
        elif dim == 2:
            rho_fit = rho_fit[:,1]
    except IndexError:
        pass

    # creates
    # gsf_fit = T_fit/pow(Emid_Eg,3)  # assuming dipoles only
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
    P_fit = rsg.PfromRhoT(rho_fit,T_fit, Nbins_Ex, Emid_Eg, Emid_nld, Emid_Ex)

    from matplotlib.colors import LogNorm # To get log scaling on the z axis
    colorbar_object = ax.pcolormesh(pltbins_Eg, pltbins_Ex, P_fit, norm=norm)
    f_mat.colorbar(colorbar_object, ax=ax) # Add colorbar to plot

    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

    # "does it work" plots
    # = comparison of slices of the 1st gen matrixes
    Nx, Ny = 3, 2
    Ntot = Nx * Ny
    Exs = np.linspace(Exmin,Emax,num=Ntot)

    f_mat, ax_mat = plt.subplots(Ny,Nx)
    for i in range(Ntot):
        Ex_plt = Exs[i] # Ex for this plot
        iEx = np.abs(Emid_Ex-Ex_plt).argmin()
        i_plt, j_plt = map_iterator_to_grid(i, Nx)
        ax = ax_mat[i_plt,j_plt]
        ax.plot(P_in[iEx,:],"k-", label="input")
        ax.plot(P_fit[iEx,:],"b-", label="fit")
        ax.title.set_text('Ex = {:.1f}'.format(Ex_plt))

        ax.set_xlabel(r"$E_g \, \mathrm{(MeV)}$")
        ax.set_ylabel('Probability')

    ax_mat[0,0].legend()

    plt.tight_layout()
    plt.show()



def nld_plot(rho_fit, T_fit, Emid_nld, nld_ext=None, rho_true=None, discretes=None, **kwargs):
    # New Figure: compare input and output NLD
    f_mat, ax = plt.subplots(1,1)

    # NLD
    if nld_ext is not None: ax.plot(nld_ext[:,0],nld_ext[:,1],"b--")
    if rho_true is not None: ax.plot(Emid_nld,rho_true)

    try:
        ax.errorbar(Emid_nld,rho_fit,yerr=rho_fit_err,fmt="o")
    except:
        ax.plot(Emid_nld,rho_fit,"o")
    ax.plot(discretes[:,0],discretes[:,1],"k.-")

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')
    plt.show()

def map_iterator_to_grid(counter, Nx):
    # Returns i, j coordinate pairs to map a single iterator onto a 2D grid for subplots.
    # Counts along each row from left to right, then increments row number
    i = counter // Nx
    j = counter % Nx
    return i, j

def normalized_plots(rho_fit, gsf_fit, gsf_ext_low, gsf_ext_high, Emid_Eg, Emid_nld, rho_true=None, rho_true_binwidth=None, gsf_true=None):
    # New Figure: compare input and output NLD and gsf
    f_mat, ax_mat = plt.subplots(2,1)

    # NLD
    ax = ax_mat[0]
    if rho_true is not None: ax.step(np.append(-rho_true_binwidth,rho_true[:-1,0])+rho_true_binwidth/2.,np.append(0,rho_true[:-1,1]), "k", where="pre",label="input NLD, binned")
    ax.plot(Emid_nld,rho_fit,"o")

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_x \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$\rho \, \mathrm{(MeV)}$')

    # gsf
    ax = ax_mat[1]
    if gsf_true is not None: ax.plot(gsf_true[:,0],gsf_true[:,1])
    ax.plot(Emid_Eg,gsf_fit,"o")
    [gsf_ext_high_plt] = ax.plot(gsf_ext_high[:,0],gsf_ext_high[:,1],"r--", label="ext. high")
    [gsf_ext_low_plt] = ax.plot(gsf_ext_low[:,0],gsf_ext_low[:,1],"b--", label="ext. high")

    ax.set_yscale('log')
    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$gsf \, \mathrm{(MeV**(-3)}$')

    plt.show()
