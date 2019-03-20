"""
Spin-sum factor which is potentially necessay to add
in the Oslo Method decomposition of nld and gSF

Copyright (C) 2019 Fabio Zeiser and Jørgen Eriksson Midtbø
University of Oslo
fabio.zeiser [0] fys.uio.no, jorgenem [0] gmail.com
"""

from spinfunctions import SpinFunctions
import numpy as np
import matplotlib.pyplot as plt

# choose a spincut model and give it's parameters
spincutModel = "EB05"
spincutPars = {"mass": 240, "NLDa": 25.16, "Eshift": 0.12}  # some dummy values


def spin_dist(Ex, J):
    return SpinFunctions(Ex=Ex, J=J,
                         model=spincutModel,
                         pars=spincutPars).distibution()


# Some more settings
Jmax = 20
Js = np.linspace(0, Jmax, Jmax+1)
Es = np.linspace(0, 10)


def z(Ex, Eg):
    """ Spin sum

    Depends on population cross-section (spin-parity distribution) and
    intrinsic spin-parity distribution
    """
    z = 0
    # sum over initial spins
    for ji in Js:
        # Jfs = possible final spins
        if ji == 0:
            Jfs = [1]
        else:
            Jfs = [ji-1, ji, ji+1]
        # assume g_pop propto g_int
        # TODO: should the 1/2 be there?
        g_pop = spin_dist(Ex, ji)
        # sum over final spins
        inner_sum = 0
        for jf in Jfs:
            # TODO: should the 1/2 be there?
            inner_sum += 1/2 * spin_dist(Ex-Eg, jf)
        z += g_pop * inner_sum
    return np.tril(z)


Ex, Eg = np.meshgrid(Es, Es)
z_array = z(Ex, Eg)

f, ax = plt.subplots(1, 1)
ax.set_title("z-Factor (spin sums)")
cbar = plt.imshow(z_array, origin='lower',
                  extent=[Es[0], Es[-1], Es[0], Es[-1]],
                  vmin=min(z_array[z_array!=0]), cmap='Greys')
f.colorbar(cbar, ax=ax)
ax.set_xlabel("Eg")
ax.set_ylabel("Ex")

f, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title("Fix Ex")
Ex_select = 2
idE = (np.abs(Es-Ex_select)).argmin()
ax1.plot(Es,z(Ex,Eg)[idE,:],label="Ex={:.1f}".format(Ex_select))
Ex_select = 3
idE = (np.abs(Es-Ex_select)).argmin()
ax1.plot(Es,z(Ex,Eg)[idE,:],label="Ex={:.1f}".format(Ex_select))
Ex_select = 4
idE = (np.abs(Es-Ex_select)).argmin()
ax1.plot(Es,z(Ex,Eg)[idE,:],label="Ex={:.1f}".format(Ex_select))
ax1.legend(loc="best")
ax1.set_xlabel("Ex")

ax2.set_title("Diagonal: \"impact on nld\"")
xarr = Es[::-1]-Es
diag = np.rot90(z_array).diagonal()[::-1]
ax2.plot(xarr, diag)
ax2.set_xlim(xmin=0)

f.tight_layout()
plt.show()
