#!/usr/bin/env python3

"""
Created on Thu Dec  2 15:09:18 2021

@author: Nerine
"""

import numpy as np
from scipy.stats import norm, beta, bernoulli
from statsmodels.base.model import GenericLikelihoodModel
import scipy as sct
import scipy.stats as st

# variables
h_max = 16e3 # m , maximum cloud top height TODO check

# functions

def AIC(k, L):
    ### L maximized log likelihood
    return 2 * k - 2 * L

def BIC(k, L, n):
    ### L maximized log likelihood
    return k * np.log(n) - 2 * L


def KS(x, cdf, args = ()):
    return st.kstest(x, cdf, args = args)

def CM(x, cdf, args = ()):
    return st.cramervonmises(x, cdf, args = args)

def AD(x, F_cdf, *args):
    N = len(x)
    y = np.sort(x)
    cdf = F_cdf(y, *args)
    logcdf = np.log(cdf)
    logsf = np.log(1 - cdf)

    i = np.arange(1, N + 1)
    A2 = -N - np.sum((2*i - 1.0) / N * (logcdf + logsf[::-1]), axis=0)

    return A2