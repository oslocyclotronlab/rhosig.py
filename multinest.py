import json
import sys
import numpy
import scipy, scipy.stats
import pymultinest
import os

import normalization as onorm
# import matplotlib.pyplot as plt

def run_nld_2regions(parameters, args):

    # a more elaborate prior
    # parameters are pos1, width, height1, [height2]
    def prior(cube, ndim, nparams):
        # cube[0] = cube[0]            # uniform prior between 0:1
        # cube[0] = norm.ppf(cube[0], loc=0,scale=2) # log-normal prior
        # cube[0] = np.exp(cube[0])
        # cube[0] = cube[0]*20-10 # uniform prior between -10 and 10
        # cube[0] = cube[0]*20-10 # uniform prior between -10 and 10

        #TODO: important to adjust for each case!!
        cube[0]=scipy.stats.norm.ppf(cube[0], loc=6,scale=20)
        cube[1] = 10**(cube[1]*2 - 1) # log-uniform prior between 10^-1 and 10^1
        cube[2] = 10**(cube[2]*2 - 2) # log-uniform prior between 10^-2 and 10^1


    def loglike(cube, ndim, nparams):
        chi2 = onorm.NormNLD.chi2_disc_ext(cube, *args)
        loglikelihood = -0.5 * chi2
        return loglikelihood

    # number of dimensions our problem has
    # parameters = ["pos1", "width", "height1"]
    n_params = len(parameters)

    folder = 'multinest'
    if not os.path.exists(folder):
        os.makedirs(folder)
    datafile = os.path.join(os.getcwd(), "norm")

    # run MultiNest
    pymultinest.run(loglike, prior, n_params,
                    outputfiles_basename=datafile + '_3_',
                    resume = False, verbose = True)
    json.dump(parameters, open(datafile + '_3_params.json', 'w')) # save parameter names
