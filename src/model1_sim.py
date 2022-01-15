# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:12:42 2022

@author: Nerine
"""

import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.insert(0, '../lib')
sys.path.insert(0, '../src')
from scipy.stats import beta, bernoulli


import ml_estimation as ml
import Utilities as util


# variables

# functions

def createFilename(prop):
    return 'sim_' + '_'.join([f'{x}={prop[x]}' for x in prop])


def toInt(x, interval):
    a, b = interval
    if np.isinf(b):
        above = 0
    else:
        above = (x > b) * b
        
    if np.isinf(a):
        below = 0
    else:
        below = (x < a) * a
    below = (x < a) * a
    within = ((a <= x) & ( x<= b)) * x
    return above + below + within

def param_c(h, d, ds, method = 'linear'):
    return ds.interp(dict(mu_h = h, mu_d=  d), method = method, kwargs={"fill_value": "extrapolate"})

def theta_c_to_cs(h, d, ds, **kwargs):
    p = param_c(h, d, ds, **kwargs).p_cs
    p = toInt(p, [0, 1])
    return p.data

def theta_c_to_c_cth(h, d, ds, **kwargs):
    theta = param_c(h, d, ds, **kwargs)
    alpha1 = toInt(theta.alpha1.data, [0, np.inf])
    beta1 = toInt(theta.beta1.data, [0, np.inf])
    alpha2 = toInt(theta.alpha2.data, [0, np.inf])
    beta2 = toInt(theta.beta2.data, [0, np.inf])
    p = toInt(theta.p.data, [0, 1])
    return (alpha1, beta1, alpha2, beta2, p)

def theta_c_to_c_cod(h, d, ds, **kwargs):
    theta = param_c(h, d, ds, **kwargs)
    mu = theta.mu.data
    sigma = toInt(theta.sigma.data, [0, np.inf])
    return (mu, sigma)

# =============================================================================
#  Simulation
# =============================================================================

def step(x, ds_cs, ds_c):
    """Makes one step.
    Args:
        x (np.array): current state of the pixel (h, d)
    Returns:
        np.array: updated state of the pixel
    """

    x_is_cs = np.isnan(x).any()
    if x_is_cs:
        p_cs = ds_cs.theta1.data
    else:
        p_cs = theta_c_to_cs(*x, ds_c , method = 'nearest')
    
#     print('x and p_cs', x, p_cs)
    
    # to cloud or clear sky
    x_next_is_cs = bernoulli.rvs(p_cs)
    
    if x_next_is_cs:
        h2, d2 = np.nan, np.nan
    else:               
        if x_is_cs: ## x current is cs
            cod_param = ds_cs.theta3[7:9].data
            cth_param = ds_cs.theta3[2:7].data

        else: ## x is cloud:
            cth_param = theta_c_to_c_cth(*x, ds_c, method = 'nearest')
            cod_param = theta_c_to_c_cod(*x, ds_c, method = 'nearest')
#         print(cod_param , cth_param)
        mu, sigma = cod_param
        alpha1, beta1, alpha2, beta2, p = cth_param
        d2 = np.random.randn(1) * sigma + mu   

        x = np.random.rand(1)
        y1 = beta.rvs(alpha1, beta1)
        if p == 1:
            h2 = y1
        else:
            y2 = beta.rvs(alpha2, beta2)
            u = (x < p)
        #     print('u, y1, y2', u, y1 ,y2)
            h2 = ml.UnitInttoCTH(u * y1 + ~u * y2)
    
    
    return h2, d2

def sim_model1(steps, ds_cs, ds_c,
                       init_cond='random', impulse_pos='center'):
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

    x = np.zeros((steps, 2))
    x[0,:] = np.array([1e3, 1])
    for i in range(steps - 1):
        print(i, end = '\r')
        x[i+1,:] = step(x[i,:], ds_cs, ds_c)
    print('finished')
    return x


if __name__ == "__main__":
    
    loc_mod = './space-time-clouds/mod/model1/'
    loc_sim_data = '/net/labdata/nerine/space-time-clouds/data/sim/model1'

    ds_c = xr.open_dataset(loc_mod + 'expl_local_param.nc')
    ds_cs = xr.open_dataset(loc_mod + 'glob_theta.nc')[['theta1', 'theta3']]
    
    
    
    n = 10000
    
    mu_h = np.linspace(0, ml.h_max, 60)
    mu_d = np.linspace(-3, 6, 50)
    ds = param_c(mu_h, mu_d, ds_c, method = 'nearest')
    x = sim_model1(n, ds_cs, ds)
    

    df = pd.DataFrame(x, columns = ['h_t', 'd_t'])
    df['cloud'] = df.apply(lambda x : 'cloud' if ~np.isnan(x.h_t) else 'clear sky', axis = 1)
    dqf = df.cloud.apply(lambda x : 6 if x == 'clear sky' else 0)
    df['ISCCP'] = util.classifyISCCP(df.d_t, df.h_t, dqf , bound = [np.log(3.6), np.log(23), 2e3, 8e3]).astype(int)
    
    df = df.append({'h_t' : np.nan, 'd_t' : np.nan}, ignore_index = True)
    df['h_t_next'] = np.roll(df.h_t, -1)
    df['d_t_next'] = np.roll(df.d_t, -1)
    df['cloud_next'] = np.roll(df.cloud, -1)
    df['ISCCP_next'] =  np.roll(df.ISCCP, -1)
    df.to_csv(loc_sim_data + f'sim_n={len(df) - 1}.csv')
    df

# ds_theta#.theta1, ds_theta.theta3
# ds_localda