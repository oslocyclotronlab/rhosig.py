import numpy as np

def rebin_and_shift(array, E_range, N_final, rebin_axis=0):
    """ *Smart* rebin of M-dimensional array either to larger or smaller binsize.

    Parameters:
    -----------
    array : ndarray
        (2D?) Array that should be rebinned
    E_range : ndarray
        Array of mid bin energies
    N_final : int
        Number of bins in the resulting matrix
    rebin_axis : int, optional
        Axis that will be rebinned

    Returns:
    --------
    array_rebinned : ndarray
        Rebinned array
    E_range_shifted_and_scaled : ndarray
        Bin center energies of the rebinned axis

    Notes:
    ------

    Written by J{\o}rgen E. Midtb{\o}, University of Oslo, 2018
    Modified: Fabio Zeiser, 2018
    # Changes:
        20181030 return mid bin energy
    GNU General Public License v3.0 or later

    Implementation notes:
    Rebinning is done with simple proportionality. E.g. for down-scaling rebinning (N_final < N_initial):
    If a bin in the original spacing ends up between two bins in the reduced spacing, then the counts of that bin are split proportionally between adjacent bins in the rebinned array. Upward binning (N_final > N_initial) is done in the same way, dividing the content of bins equally among adjacent bins.

    Technically it's done by repeating each element of array N_final times and dividing by N_final to preserve total number of counts, then reshaping the array from M dimensions to M+1 before summing along the new dimension of length N_initial, resulting in an array of the desired dimensionality.

    This version (called rebin_and_shift rather than just rebin) takes in also the energy range array (lower bin edge) corresponding to the counts array, in order to be able to change the calibration. What it does is transform the coordinates such that the starting value of the rebinned axis is zero energy. This is done by shifting allbins, so we are discarding some of the eventual counts in the highest energy bins. However, there is usually a margin.
    """

    if isinstance(array, tuple): # Check if input array is actually a tuple, which may happen if rebin_and_shift() is called several times nested for different axes.
        array = array[0]


    N_initial = array.shape[rebin_axis] # Initial number of counts along rebin axis

    # TODO: Loop this part over chunks of the Ex axis to avoid running out of memory.
    # Just take the loop from main program in here. Have some test on the dimensionality
    # to judge whether chunking is necessary?

    # Repeat each bin of array Nfinal times and scale to preserve counts
    array_rebinned = array.repeat(N_final, axis=rebin_axis)/N_final

    # Convert mid bin to lower bin edge
    E_range = np.copy(E_range)
    bin_width_org = E_range[1]-E_range[0]
    E_range -= bin_width_org/2.
    if np.isclose(E_range[0],0): E_range[0] = 0 # avoid small rounding error

    if E_range[0] < 0 or E_range[1] < E_range[0]:
        raise Exception("Error in function rebin_and_shift(): Negative zero energy is not supported. (But it should be relatively easy to implement.)")
    else:
        # Calculate number of extra slices in Nf*Ni sized array required to get down to zero energy
        n_extra = int(np.ceil(N_final * (E_range[0]/(E_range[1]-E_range[0]))))
        # Append this matrix of zero counts in front of the array
        indices_append = np.array(array_rebinned.shape)
        indices_append[rebin_axis] = n_extra
        array_rebinned = np.append(np.zeros(indices_append), array_rebinned, axis=rebin_axis)
        array_rebinned = np.split(array_rebinned, [0, N_initial*N_final], axis=rebin_axis)[1]
        indices = np.insert(array.shape, rebin_axis, N_final) # Indices to reshape to
        array_rebinned = array_rebinned.reshape(indices).sum(axis=(rebin_axis+1))
        E_range_shifted_and_scaled = np.linspace(0, E_range[-1]-E_range[0], N_final)
        bin_width_new = E_range_shifted_and_scaled[1] - E_range_shifted_and_scaled[0]
        # E_range_shifted_and_scaled += bin_width_new/2.
    return array_rebinned, E_range_shifted_and_scaled


def read_mama_2D(filename):
    """ Reads a MAMA matrix file and returns the matrix as a numpy array

    Parameters:
    -----------
    fname : string
        Path to input `mama` file

    Returns:
    --------

    matrix : ndarray
        `mama` matrix values (counts) as ndarray
    cal : dict of str : float
        Contains calibration coefficients `a0x`, `a1x`, `a2x`, `a0y`, `a1y`, `a2y`
    y_array : ndarray
        Mid bin energies of 1(!) axis (ie. Ex), in MeV
    x_array : ndarray
        Mid bin energies of 2(!) axis (ie. Eg), in MeV

    Notes:
    ------
    Written by J{\o}rgen E. Midtb{\o}, University of Oslo, 2018
    Modified: Fabio Zeiser, 2018
    GNU General Public License v3.0 or later

    # Changes:
        20181030 return mid bin energy & MeV instead of keV
    """
    matrix = np.genfromtxt(filename, skip_header=10, skip_footer=1)
    cal = {}
    with open(filename, 'r') as datafile:
        calibration_line = datafile.readlines()[6].split(",")
        cal = {"a0x":float(calibration_line[1]), "a1x":float(calibration_line[2]), "a2x":float(calibration_line[3]),
             "a0y":float(calibration_line[4]), "a1y":float(calibration_line[5]), "a2y":float(calibration_line[6])}

    # convert keV to MeV
    cal = {key: val / 1e3 for key, val in cal.items()}

    Ny, Nx = matrix.shape
    y_array = np.linspace(0, Ny-1, Ny)
    y_array = cal["a0y"] + cal["a1y"]*y_array + cal["a2y"]*y_array**2
    y_array += cal["a1y"]/2. # convert to bin centers
    x_array = np.linspace(0, Nx-1, Nx)
    x_array = cal["a0x"] + cal["a1x"]*x_array + cal["a2x"]*x_array**2
    x_array += cal["a1x"]/2. # convert to bin centers

    # Returning y (Ex) first as this is axis 0 in matrix language
    return matrix, cal, y_array, x_array
