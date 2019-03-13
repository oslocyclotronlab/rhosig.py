import json
import sys
import numpy
import math
import scipy.stats
import pymultinest
import os

import normalization as omnorm

def run_nld_2regions(popt, chi2_args):
    """
    Run multinest for the nld normalization on two regions:
    Discrete levels at low energy and nld(Sn) (or model) at high energies

    Parameters:
    -----------
    popt: (OrderedDict of str: float)
        Parameter names and corresponding best fit. They will be used to create the priors
    chi2_args: (tuple)
        Additional arguments for the Chi2 minimization

    Returns:
    --------
    multinest-files:
        dumps multinest files to disk
    """

    assert list(popt.keys()) == ["A", "alpha", "T"], \
        "check if parameters really differ, if so, need to adjust priors!"

    A = popt["A"]
    alpha = popt["alpha"]
    alpha_exponent = math.log(alpha, 10)
    T= popt["T"]
    T_exponent = math.log(T, 10)

    def prior(cube, ndim, nparams):
        # TODO: You may want to adjust this for your case!
        # log-normal prior
        cube[0] = scipy.stats.norm.ppf(cube[0], loc=A,scale=4*A)
        # log-uniform prior
        # # if alpha = 1e2, it's between 1e1 and 1e3
        cube[1] = 10**(cube[1]*(alpha_exponent+2) - (1-alpha_exponent))
        # # log-uniform prior
        # # if T = 1e2, it's between 1e1 and 1e3
        cube[2] = 10**(cube[2]*(T_exponent+2) - (1-T_exponent))


    def loglike(cube, ndim, nparams):
        chi2 = omnorm.NormNLD.chi2_disc_ext(cube, *chi2_args)
        loglikelihood = -0.5 * chi2
        return loglikelihood

    # number of dimensions our problem has
    n_params = len(popt)

    folder = 'multinest'
    if not os.path.exists(folder):
        os.makedirs(folder)
    datafile = os.path.join(os.getcwd(), *(folder,"nld_norm"))

    # run MultiNest
    pymultinest.run(loglike, prior, n_params,
                    outputfiles_basename=datafile + '_1_',
                    resume = False, verbose = True)
    # save parameter names
    json.dump(list(popt.keys()), open(datafile + '_1_params.json', 'w'))
