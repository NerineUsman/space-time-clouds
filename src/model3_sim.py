#!/usr/bin/env python3
# coding: utf-8

"""
Created on Tue Mar  8 11:35:55 2022

@author: Nerine
"""

# import modules
import numpy as np
import pandas as pd
import xarray as xr
import itertools
import pickle
from datetime import datetime 
from scipy import stats

import sys, os
from scipy.stats import beta, bernoulli, norm, uniform

sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib/')
sys.path.insert(0, '../src')


import ml_estimation as ml
import ml_estimation2 as ml2
import model2_sim as sim2
import Utilities as util



# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_sim2.txt'
# input_file = './space-time-clouds/src/input_model1.txt'

# functions





def w(ct):
    h = xr.where(ct == 1, 0, 
             xr.where(ct == 2, 1,
                xr.where(ct == 3, 1,
                 xr.where(ct == 4, 7, 
                 xr.where(ct == 5, 2, 
                      xr.where(ct == 6, 2, 
                           xr.where(ct == 7, 6, 
                                xr.where(ct == 8, 3, 
                                     xr.where(ct == 9, 4, 
                                          xr.where(ct == 10, 5, np.nan))))))))))

    return h


def w_compl(ct):
    h = xr.where(ct == 1, np.exp(-1 *  1j * np.pi), 
             xr.where(ct == 2, np.exp(-3/4 *  1j * np.pi),
                xr.where(ct == 3,  np.exp(-3/4 *  1j * np.pi),
                 xr.where(ct == 4, np.exp(2/3 *  1j * np.pi) , 
                 xr.where(ct == 5,  np.exp(-2/3 *  1j * np.pi) , 
                      xr.where(ct == 6, np.exp(-2/3 *  1j * np.pi), 
                           xr.where(ct == 7, np.exp(1/3 *  1j * np.pi), 
                                xr.where(ct == 8, np.exp(-1/3 *  1j * np.pi), 
                                     xr.where(ct == 9,  np.exp(-1/4 *  1j * np.pi), 
                                          xr.where(ct == 10, np.exp(0 *  1j * np.pi), np.nan))))))))))
    

    return h
    
    
    


def step_image(image, M, method = 'standard'):
    """Makes one step.
    Args:
        image has variables ct, (i, j)
    Returns:
        image with variables ct
        
    """
    
    if method not in ('standard', 'compl'):
        raise NotImplementedError("%s is unsupported: Use standard or compl " % method)
    
    
    image = image.copy(deep = True)
    orig_ct = image.ct.copy(deep =True)
    # calculate expl variables per pixel 
    # ct, g
    
    if method == 'standard':
        weights = w(image.ct)
    elif method == 'compl':
        weights = w_compl(image.ct)
        
        
    g = weights.rolling(i=3, j = 3, center=True, min_periods = 1).sum() - weights
    
    if method == 'compl':
        g = np.round( ((np.angle(g) / ( 2* np.pi) ) ) * 51 - .5).flatten()
    else:
        g = g.data.flatten()                

        
        
    # having all values, the order doesn't matter anymore for drawing the next 
    # time step
    ct = image.ct.data.flatten()

    # probability on cloudtpyes
    p = sim2.param_pixel(M, ['frm', 'g'], [ct, g])
    

    # draw next step from uniform
    u = uniform.rvs(size = len(ct))
    
    # determine next class
    invalid = (1 - (u < p).max(dim = 'to')).astype(bool)
    n_invalid = invalid.sum()
    ct_next =  (u < p ).argmax(dim = 'to') + 1
    
    ct_next = np.where(invalid, ct, ct_next)


    image.ct[:] = ct_next.reshape(*image.ct.shape)    
    
    
    return image.copy(deep = True).where(orig_ct > 0)   , int(n_invalid.data)






def sim_model3(x0, steps, M,
                       **kwargs):
    """Generate the state of an elementary cellular automaton after a pre-determined
    number of steps starting from some random state.
    Args:
        rule_number (int): the number of the update rule to use
        size (int): number of cells in the row
        steps (int): number of steps to evolve the automaton
        init_cond (str): either `random` or `impulse`. If `random` every cell
        in the row is activated with prob. 0.5. If `impulse` only one cell
        is activated.
        impulse_pos (str): if `init_cond` is `impulse`, activate the
        left-most, central or right-most cell.
    Returns:
        np.array: final state of the automaton
    """
    
    x = x0.copy(deep = True)
    x['t'] = 0
    
    for i in range(steps):
        print(i)
        if i > 0:
            x_prev = x.sel(t = i).copy(deep = True)
        else:
            x_prev = x.copy(deep = True)
            print('first step')
        x_next, n_invalid = step_image(x_prev, M, **kwargs)
        x_next['t'] = i + 1
        print(i, n_invalid)
        x_next['n_invalid'] = (['n'], [n_invalid])
        x = xr.concat((x, x_next), dim = 't' )
    print('finished')
    return x

# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f if (len(line) > 1) & (line[0] != '#') ])
    
    loc_model = input['loc_model3']
    
    loc_sim = input['loc_sim3']

    N = int(input['N'])
    T = int(input['T'])
    method = input['method']
    method = 'standard'
    
    if method == 'compl':
        loc_model = loc_model + 'compl/'

    
    M = pd.read_csv(loc_model + 'transition_ctypes.csv')
    M = M.iloc[:-1].astype({'g': 'float', 'from' : 'float'}).rename({'from': 'frm'}, axis=1)
    M = M.set_index(['g','frm']).to_xarray()
    M = xr.concat([M['0.0'],
                M['1.0'],
                M['2.0'],
                M['3.0'],
                M['4.0'],
                M['5.0'],
                M['6.0'],
                M['7.0'],
                M['8.0'],
                M['9.0'],
                M['10.0'],
                ], pd.Index(np.arange(11), name="to", dtype = 'float'))
    M = M.sel(to = slice(1,10), frm = slice(1, 10))
    M = M / M.sum(dim = 'to')
    mask = ''

    # M = xr.where((M < .1) & (M.frm > 1), 0 , M )
    # M = xr.where((M < .08) & (M.frm == 1), 0 , M )
    
    # M = xr.where((M < .1), 0 , M )
    # mask = 'mask_low_prob'
    
    # print(M.shape)
    # ## add extra weight to diagonal
    # # M = M +  xr.where(M.frm == M.to, 5 , 0)
    
    M = M / M.sum(dim = 'to')
    
    print(M.shape)
    
    M = M.cumsum(dim = 'to')

# =============================================================================
#     Generate X0
# =============================================================================
    
    image = xr.open_dataset('../data/start_image.nc')
    N = len(image.i)


    start_image = 'scene'
    
    # image.h[:] = 2000
    # image.d[:] = 0
    # image.h[:] = -1
    # image.ct[:] = 1
    # start_image = 'constant'
    
    T = 40

    
# # =============================================================================
# #     Simulation
# # =============================================================================
    
    # N = 20
    X0 = image[['ct']]#.isel(i = np.arange(N), j = np.arange(N))
    X0['t'] = 0
    X = sim_model3(X0, T, M, method = method)
    
    # update z and ct
    X['z'] = (X.ct == 1)
        
    filename = loc_sim + f'na_simulation3_T{T}_{method}_startimage_{start_image}_{mask}'#'_N{N}'
    X.to_netcdf(filename)
    
    

    N =  40    
    plot = X.ct.isel(
        # i = np.arange(N), j = np.arange(N),
                t = [0, 1, 2, 3, 4, 10 , 20, 30, T-1]
              ).plot(col = 't', col_wrap = 5)
    plot.fig.subplots_adjust(top=0.9, right = 0.8) # adjust the Figure in rp
    plot.fig.suptitle(f'simulation3_T{T}_{method}_startimage_{start_image}_mask')
    print(T, method, start_image, mask)

#     X.d.plot(col = 't', col_wrap = 5)
    
    import matplotlib.pyplot as plt

#     plt.figure()
#     plt.hist(X.h.where(X.h > 0).sel(t = 1).data.flatten(), bins = 50, density = True)
    
