# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:43:47 2022

@author: Nerine

"""


import numpy as np
import xarray as xr
import itertools

import sys, os
sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib/')
import Utilities as util


def dropna(array1):
    nan_array = np.isnan(array1)
    not_nan_array = ~ nan_array
    array2 = array1[not_nan_array]
    return array2

def cloudFraction(image):
    x = dropna(image.astype(int).where(image > 0).data.flatten()).astype(int)
    return np.bincount(x, minlength = 11) / len(x)#, np.unique(x)

def sliceImage(image, ibound, jbound):
    
    return  image.sel( i  = slice(image.i[ibound[0]], image.i[ibound[1]]),
                      j = slice(image.j[jbound[0]], image.j[jbound[1]]))


def cloudFractionSD(image, N):
    """
    blocks of NXN
    """

    I = np.arange(len(image.i) // N)
    J = np.arange(len(image.j) // N)

    cfs = np.zeros((len(I), len(J), 11))

    for i, j in itertools.product(I, J):
        ibound = [i * N, (i + 1) * N - 1]
        jbound = [j * N, (j + 1) * N - 1]
        sub_image = sliceImage(image, ibound, jbound )
        
        cfs[i,j,:] = cloudFraction(sub_image)
    return cfs.std(axis = (0,1))
    
if __name__ == "__main__":
    
    
    x = xr.open_dataset('../mod/model2/sim/simulation2_main_beta_T30_N20')
    x['ct'][:] = util.classISCCP(np.exp(x.d), x.h)
    x['ct'] = x.ct.where(~x.z, 1)
    
    X = x.ct
    cf = np.array([cloudFraction(X.sel(t = t)) for t in x.t ])
    sigma_cf = np.array([cloudFractionSD(X.sel(t = t), 32) for t in x.t ])
    
    x['cf'] = (['t', 'classes'], cf)
    x['sigma_cf'] = (['t', 'classes'], sigma_cf)
    
    x.to_netcdf('../mod/model2/sim/simulation2_main_beta_T30_N20_cf')

