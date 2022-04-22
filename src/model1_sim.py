#!/usr/bin/env python3
"""
Created on Sat Jan 15 14:12:42 2022

@author: Nerine
"""

import numpy as np
import pandas as pd
import xarray as xr

import sys, os
sys.path.insert(0, '../lib')
sys.path.insert(0, './space-time-clouds/lib')
from scipy.stats import beta, bernoulli, norm


import ml_estimation as ml
import Utilities as util

# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model.txt'

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
    within = ((a <= x) & ( x <= b)) * x
    return above + below + within

def param_c(h, d, ds, method = 'linear'):
    ds = ds.interpolate_na(dim = 'mu_h', 
                           method = method, 
                           fill_value = 'extrapolate')
    return ds.interp(dict(mu_h = h, mu_d=  d), 
                     method = method, 
                     kwargs={"fill_value": "extrapolate"})

def param_pixel(h, d, da):
    """
    

    Parameters
    ----------
    h : TYPE
        list of h.
    d : TYPE
        list of d.
    theta : Data array

    Returns
    -------
    None.

    """
    return da.sel(mu_h=xr.DataArray(h, dims="pixel"), mu_d=xr.DataArray(d, dims="pixel"), method = 'nearest')    


def theta_c_to_cs(h, d, ds, **kwargs):
    p = ds.p_cs.sel(mu_h=xr.DataArray(h, dims="pixel"), mu_d=xr.DataArray(d, dims="pixel"), method = 'nearest')    
    p = toInt(p, [0, 1])
    return p

def theta_c_to_c_cth(h, d, ds, **kwargs):
    # theta = param_c(h, d, ds, **kwargs)
    alpha1 = param_pixel(h, d, ds.alpha1)
    alpha1 = toInt(alpha1, [0, np.inf])

    beta1 = param_pixel(h, d, ds.beta1)
    beta1 = toInt(beta1, [0, np.inf])
    
    alpha2 = param_pixel(h, d, ds.alpha2)    
    alpha2 = toInt(alpha2, [0, np.inf])

    beta2 = param_pixel(h, d, ds.beta2)
    beta2 = toInt(beta2, [0, np.inf])

    p = param_pixel(h, d, ds.p)
    p = toInt(p, [0, 1])
    
    return (alpha1, beta1, alpha2, beta2, p)

def theta_c_to_c_cod(h, d, ds, **kwargs):
    # theta = param_c(h, d, ds, **kwargs)
    
    mu = param_pixel(h, d, ds.mu)
    sigma = param_pixel(h, d, ds.sigma)
    sigma = toInt(sigma, [0, np.inf])
    return (mu, sigma)

def param_model1(x, ds_cs, ds_c):
    x_is_cs = np.isnan(x).any()
    if x_is_cs:
        p_cs = ds_cs.theta1.data[0]
    else:
        p_cs = theta_c_to_cs(*x, ds = ds_c, method = 'nearest')
    
#     print('x and p_cs', x, p_cs)
    
    # to cloud or clear sky
    if x_is_cs: ## x current is cs
        cod_param = ds_cs.theta3[7:9].data
        cth_param = ds_cs.theta3[2:7].data

    else: ## x is cloud:
        cth_param = theta_c_to_c_cth(*x, ds = ds_c, method = 'nearest')
        cod_param = theta_c_to_c_cod(*x, ds = ds_c, method = 'nearest')
#         print(cod_param , cth_param)
    
    return p_cs, cod_param, cth_param

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

        print(f'x = {x}, cth param {cth_param}')
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

def step2d(image, ds_cs, ds_c):
    """Makes one step.
    Args:
        image.h (np.array): n x m array
        image.d (np.array): n x m array
    Returns:
        np.array: updated state of the domain n x m
    """

    # image['z'] = (image.h == -1) * (image.d == 0)

        
    image = image#.sel(i = 0)
    # print(image)
    h = image.h.data.flatten()
    d = image.d.data.flatten()
    
    z = (image.h == -1) * (image.d == 0)
    z = z.data.flatten()
    
    
    
    # probability on cs
    p_cs = theta_c_to_cs(h, d, ds_c, method = 'nearest')
    p_cs[z] = ds_cs.theta1.data

    # cloud distribution parameters
    # cs
    cod_param_cs = ds_cs.theta3[7:9].data
    cth_param_cs = ds_cs.theta3[2:7].data

    # c
    cth_param = theta_c_to_c_cth(h, d, ds_c, method = 'nearest')
    cth_param = xr.concat(cth_param, pd.Index( ['alpha1', 'beta1', 'alpha2', 'beta2', 'p'], name = 'variable'))
    cod_param = theta_c_to_c_cod(h, d, ds_c, method = 'nearest')
    cod_param = xr.concat(cod_param, pd.Index( ['mu', 'sigma'], name = 'variable'))
    
    # replace values for pixels with cs
    n_cs = sum(z)
    n = len(z)
    cth_param[dict(pixel = z)] = np.tile(cth_param_cs,(n_cs,1)).T
    cod_param[dict(pixel = z)] = np.tile(cod_param_cs,(n_cs,1)).T
    
    # draw next step    
    z_next = bernoulli.rvs(p_cs).astype(bool)
    
    cod_param['d_next'] =(['pixel'], norm.rvs(loc = cod_param.sel(variable = 'mu'), scale = cod_param.sel(variable = 'sigma')))
    
    cth_param['mix'] = (['pixel'], bernoulli.rvs(cth_param.sel(variable = 'p')))
    cth_param['b1'] = (['pixel'], beta.rvs(cth_param.sel(variable = 'alpha1'), cth_param.sel(variable = 'beta1')))
    cth_param['b2'] = (['pixel'], beta.rvs(cth_param.sel(variable = 'alpha2'), cth_param.sel(variable = 'beta2')))
    h_next = ml.UnitInttoCTH(cth_param.b1.where(cth_param.mix, cth_param.b2))
    
    # combine resutls of z, h, and d next and replace states for cs
    h_next = h_next.where(~z_next, -1)
    d_next = cod_param.d_next.where(~z_next, 0)
    
    image = image.copy(deep = True)
    image.h[:] = h_next.data.reshape(*image.h.shape)    
    image.d[:] = d_next.data.reshape(*image.h.shape)    
    
    return image.copy(deep = True)

def sim_model1(x0, steps, ds_cs, ds_c,
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
    
    x = x0.copy(deep = True)
    x['t'] = 0
    
    for i in range(steps):
        print(i)
        if i > 0:
            x_prev = x.sel(t = i)
        else:
            x_prev = x
        x_next = step2d(x_prev, ds_cs, ds_c)
        x_next['t'] = i + 1
        x = xr.concat((x, x_next), dim = 't' )
    print('finished')
    return x


if __name__ == "__main__":
    
    
    with open(input_file) as f:
        input = dict([line.split() for line in f if (len(line) > 1) & (line[0] != '#')])
        
    loc_mod = input['loc_model1']
    loc_mod = '../mod/model1/'
    # loc_mod = './space-time-clouds/mod/model1/'
    
    loc_sim_data = '/net/labdata/nerine/space-time-clouds/data/sim/model1/'
    loc_sim_data = '../data/simulation/model1/'


    ds_c = xr.open_dataset(loc_mod + 'expl_local_param.nc')
    ds_cs = xr.open_dataset(loc_mod + 'glob_theta.nc')[['theta1', 'theta3']]
    
    
    n = 6 * 6
    
    mu_h = np.linspace(0, ml.h_max, 60)
    mu_d = np.linspace(-3, 6, 50)
    ds = param_c(mu_h, mu_d, ds_c, method = 'nearest').sel(est = 'coef')
    
    
    
    image = xr.Dataset(
    data_vars=dict(
        d=(["i", "j"], np.array([[3,2,0],[3,0,1]], dtype = float)),
        h=(["i", "j"], np.array([[3000,200,-1],[3,-1,1]])),
    ),
    coords=dict(
    ))
    
    image = xr.open_dataset('../data/start_image0.nc')
    
    step2d(image, ds_cs, ds)
    
    x = sim_model1(image, n, ds_cs, ds)
    
    x['z'] = (x.h < 0)
    
    
    x.to_netcdf(loc_sim_data + f'sim_n={x.dims["t"]}_{x.dims["i"]}x{x.dims["j"]}.nc')
    
    
    
    x.h.where(x.h > 0).plot(x = 'i', y = 'j', col = 't', col_wrap = 4)
    

    # df = pd.DataFrame(x, columns = ['h_t', 'd_t'])
    # df['cloud'] = df.apply(lambda x : 'cloud' if ~np.isnan(x.h_t) else 'clear sky', axis = 1)
    # dqf = df.cloud.apply(lambda x : 6 if x == 'clear sky' else 0)
    # df['ct'] = util.classifyISCCP(df.d_t, df.h_t, dqf , bound = [np.log(3.6), np.log(23), 2e3, 8e3]).astype(int)
    
    # df = df.append({'h_t' : np.nan, 'd_t' : np.nan}, ignore_index = True)
    # df['h_t_next'] = np.roll(df.h_t, -1)
    # df['d_t_next'] = np.roll(df.d_t, -1)
    # df['cloud_next'] = np.roll(df.cloud, -1)
    # df['ct_next'] =  np.roll(df.ct, -1)
    # df.to_csv(loc_sim_data + f'sim_n={len(df) - 1}.csv')
    # df

# ds_theta#.theta1, ds_theta.theta3
# ds_localda
