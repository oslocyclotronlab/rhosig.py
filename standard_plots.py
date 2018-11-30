import numpy as np
import matplotlib.pyplot as plt
import rhosig as rsg

def rsg_plots(rho_fit, T_fit, P_in, Emid, Emid_rho, Emid_Ex, Exmin, Egmin,nld_ext=None, rho_true=None, **kwargs):
    Nbins_Ex, Nbins_Eg= np.shape(P_in) # after
    bin_width = Emid[1]- Emid[0]
    Eup_max = Exmin + Nbins_Ex * bin_width # upper bound of last bin
    pltbins_Ex = np.linspace(Exmin,Eup_max,Nbins_Ex+1) # array of (start-bin?) values used for plotting
    Eup_max = Egmin + Nbins_Eg * bin_width # upper bound of last bin
    pltbins_Eg = np.linspace(Egmin,Eup_max,Nbins_Eg+1) # array of (start-bin?) values used for plotting

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
