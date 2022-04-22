#!/usr/bin/env python3
# coding: utf-8

"""
Created on Tue Mar  8 11:35:55 2022

@author: Nerine
"""

# import modules
import numpy as np
import xarray as xr
import itertools
import pickle
from datetime import datetime 

import sys, os
from scipy.stats import beta, bernoulli

sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib/')
sys.path.insert(0, '../src')


import ml_estimation as ml
import ml_estimation2 as ml2
import Utilities as util



# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_sim2.txt'
# input_file = './space-time-clouds/src/input_model1.txt'

# functions
def interpolateNN(var, dim):
    
    for i in range(len(var[dim])): 
        val = var[dim][i]
        temp = var.loc[{dim : val}]
        for j in range(len(var[dim])):
    #         print(j)
    #         print(val.data)
            if val > 0:
                j = -j
            if j > len(var[dim])/2:
                continue
            val1 = var[dim][i + j]
            if (val1 == 0):
                continue
            temp1 = var.loc[{dim : val1}]
            temp = temp.where(~np.isnan(temp), temp1) 

    #     temp.loc[np.isnan(temp)] = temp1.where(np.isnan(temp))
        var.loc[{dim : val}] = temp
    return var

def get_param(ds, loc, method = 'nearest'): # think more about interpolation method once we have the fll data
#     ds_c.interpolate_na(dim = 'mu_d', method = method, fill_value = 'extrapolate')
    return ds.interp(loc, method = method, kwargs={"fill_value": "extrapolate"})


def step_pixel(z, h, d, csf, h_bar, d_bar, ds, method = 'standard'):
    """Makes one step.
    Args:
        x (np.array(6,)):                
            current full state Y of the pixel (z, h, d, csf, h_bar, d_bar)
        ds xarray with parameter estimates
    Returns:
        np.array (3,): updated state of the pixel (z, h, d)
    """
    
    if method not in ('standard', 'main_beta'):
        raise NotImplementedError("%s is unsupported: Use standard or main_beta " % method)
    
    ## p_cs
    if z: 
        loc = dict(mu_csf = csf)
        p_cs = get_param(ds.cs_p_cs, loc)
    else: 
        loc = dict(mu_h = h, mu_d = d, mu_csf = csf)
        p_cs = get_param(ds.p_cs, loc)
        h_bar = h_bar - h
        d_bar = d_bar - d
    
#     print('x and p_cs', x, p_cs)
    
    # to cloud or clear sky
    z_next = bernoulli.rvs(p_cs)
    
    if z_next: # if there is a cloud (h, d)next are not defined
        h_next, d_next = np.nan, np.nan
    else:     
        if z: 
            if csf: ## x current is a clear sky and is only surrounded by cs
            # then h_bar and d_bar ar NaN
                cod_param = ds.cs_csf_param_cod.data
                cth_param = ds.cs_csf_param_cth_bm.sel(est = 'coef').data
            else:

                loc = dict(cs_mu_dN = d_bar)
                cod_param = get_param(ds.cs_param_cod, loc).data
                loc = dict(cs_mu_hN = h_bar)
                cth_param = get_param(ds.cs_param_cth_bm.loc[dict(est = 'coef')], loc).data
#             pdb.set_trace()
        else: ## x is cloud:
            loc = dict(mu_h = h, mu_d = d, mu_dN = d_bar)
            cod_param = get_param(ds.param_cod, loc).data
            loc = dict(mu_h = h, mu_d = d, mu_hN = h_bar)
            cth_param = get_param(ds.param_cth_bm#.loc[dict(est = 'coef')]
                                  , loc).data
            
#         print(cod_param.data, cth_param.data)
        mu, sigma = cod_param
        alpha1, beta1, alpha2, beta2, p = ml2.mixmnToab(*cth_param.data)
        
#         print(mu, sigma)
        d_next = (np.random.randn(1) * sigma + mu) 
        
#         print(cth_param)
        x = np.random.rand(1)
        y1 = beta.rvs(alpha1, beta1)
        
        if method == 'main_beta':
            p = 1
        
        if p == 1:
            h_next = ml.UnitInttoCTH(y1)
        else:
            y2 = beta.rvs(alpha2, beta2)
            u = (x < p)
        #     print('u, y1, y2', u, y1 ,y2)
            h_next = ml.UnitInttoCTH(u * y1 + ~u * y2)
    
    
    return np.hstack([z_next, h_next, d_next])

def param_pixel(da, list_of_coords_names, list_of_coords_values):
    """
    

    Parameters
    ----------

    Returns
    -------
    None.

    """
    
    d = {k:xr.DataArray(v, dims="pixel") for k, v in zip(list_of_coords_names, list_of_coords_values) }
       
    return da.sel(d, method = 'nearest')    



def step_image(image, ds, method = 'standard'):
    """Makes one step.
    Args:
        image has variables  h, d, (i, j)
    Returns:
        image with variables h, d
    """
    
    
    # calculate expl variables per pixel 
    # z, csf, h_bar, d_bar,
    z = (image.h == -1) * (image.d == 0)    
    h_bar = image.h.where(image.h>0).rolling(i=3, j = 3, center=True, min_periods = 1).mean()
    d_bar = image.d.where(image.h>0).rolling(i=3, j = 3, center=True, min_periods = 1).mean()
    csf = z.rolling(i = 3, j = 3, center=True, min_periods = 1).mean()
    
                  
    # having all values, the order doesn't matter anymore for drawing the next 
    # time step
    h = image.h.data.flatten()
    d = image.d.data.flatten()                
    z = z.data.flatten()
    h_bar = h_bar.data.flatten()
    d_bar = d_bar.data.flatten()
    csf = csf.data.flatten()
    
    # probability on cs
    param_pixel(ds.p_cs, ['mu_h', 'mu_d', 'mu_csf'], [h, d, csf])
    p_cs = theta_c_to_cs(h, d, h_bar, d_bar, csf, ds_c, method = 'nearest')
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
    if method not in ('standard', 'main_beta'):
        raise NotImplementedError("%s is unsupported: Use standard or main_beta " % method)
    
    ## p_cs
    if z: 
        loc = dict(mu_csf = csf)
        p_cs = get_param(ds.cs_p_cs, loc)
    else: 
        loc = dict(mu_h = h, mu_d = d, mu_csf = csf)
        p_cs = get_param(ds.p_cs, loc)
        h_bar = h_bar - h
        d_bar = d_bar - d
    
#     print('x and p_cs', x, p_cs)
    
    # to cloud or clear sky
    z_next = bernoulli.rvs(p_cs)
    
    if z_next: # if there is a cloud (h, d)next are not defined
        h_next, d_next = np.nan, np.nan
    else:     
        if z: 
            if csf: ## x current is a clear sky and is only surrounded by cs
            # then h_bar and d_bar ar NaN
                cod_param = ds.cs_csf_param_cod.data
                cth_param = ds.cs_csf_param_cth_bm.sel(est = 'coef').data
            else:

                loc = dict(cs_mu_dN = d_bar)
                cod_param = get_param(ds.cs_param_cod, loc).data
                loc = dict(cs_mu_hN = h_bar)
                cth_param = get_param(ds.cs_param_cth_bm.loc[dict(est = 'coef')], loc).data
#             pdb.set_trace()
        else: ## x is cloud:
            loc = dict(mu_h = h, mu_d = d, mu_dN = d_bar)
            cod_param = get_param(ds.param_cod, loc).data
            loc = dict(mu_h = h, mu_d = d, mu_hN = h_bar)
            cth_param = get_param(ds.param_cth_bm#.loc[dict(est = 'coef')]
                                  , loc).data
            
#         print(cod_param.data, cth_param.data)
        mu, sigma = cod_param
        alpha1, beta1, alpha2, beta2, p = ml2.mixmnToab(*cth_param.data)
        
#         print(mu, sigma)
        d_next = (np.random.randn(1) * sigma + mu) 
        
#         print(cth_param)
        x = np.random.rand(1)
        y1 = beta.rvs(alpha1, beta1)
        
        if method == 'main_beta':
            p = 1
        
        if p == 1:
            h_next = ml.UnitInttoCTH(y1)
        else:
            y2 = beta.rvs(alpha2, beta2)
            u = (x < p)
        #     print('u, y1, y2', u, y1 ,y2)
            h_next = ml.UnitInttoCTH(u * y1 + ~u * y2)
    
    
    return np.hstack([z_next, h_next, d_next])

def neighborhood(ds, i, j,
                 n = 1 # neighbor degree
                 ):

    xlow = max(0, i - n)
    xupper = min(ds.dims['i'], i + n + 1)
    
    ylow = max(0, j - n)
    yupper = min(ds.dims['j'], j + n + 1)
    
    N = (slice(xlow, xupper, 1), slice(ylow, yupper, 1))
    
    return N

def simulation(T, X0, ds, **kwargs):
    X = X0.copy(deep = True)
    t = 0
    for t in range(T):
        print(f't = {t}')
        if t == 0 :
            Xt = X0.copy(deep = True)
        else:
            Xt = X.sel(t = t)
        X_next = X0.copy(deep = True)
        
        I, J = np.arange(*X0.i.shape), np.arange(*X0.j.shape)
        
        for i, j in itertools.product(I, J):
            N = neighborhood(Xt, i, j)
            pix = dict(i = i, j = j)
            y = [Xt.z[pix], Xt.h[pix], Xt.d[pix], 
                 Xt.z[N].mean(), Xt.h[N].mean(), Xt.d[N].mean()]
        #     print(pix)
        #     print(y)
            y = np.array([yi.data for yi in y])
        #     print('y = ', y)
    #         print('x = ', y[:3])
            if np.isnan(y[0]):
                continue
            x_next = step_pixel(*y, ds, **kwargs)
    #         print('x_n =',x_next)
            X_next.z[pix] = x_next[0]
            X_next.h[pix] = x_next[1]
            X_next.d[pix] = x_next[2]
        #     X_next.loc[pix]
        X_next['t'] = t+1
        X = xr.concat([X, X_next], dim = 't')
        X_next = None
    return X

def sim_model2(x0, steps, ds,
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
        x_next = step_image(x_prev, ds)
        x_next['t'] = i + 1
        x = xr.concat((x, x_next), dim = 't' )
    print('finished')
    return x

# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f if (len(line) > 1) & (line[0] != '#') ])
    
    loc_model = input['loc_model2']
    loc_sim = input['loc_sim']
    loc_clean_data = input['loc_clean_data'] #'../data/clean_data/'

    N = int(input['N'])
    T = int(input['T'])
    method = input['method']
    
    ds = xr.open_dataset(loc_model + 'dH=500_dD=0_5_N=5000local_param.nc')

# =============================================================================
#   adjust ds such that nan cases get a fill
# =============================================================================
    ## from cloud 

    # p_cs
    n_loc = dict(n_or_val = 'n')
    n = ds.p_cs.loc[n_loc]
    
    
    dummy = ds.p_cs.loc[dict(n_or_val = 'val')].where(n > 100).interpolate_na(
        dim = 'mu_csf', method = 'zero', fill_value = 'extrapolate')
    ds.p_cs.loc[dict(n_or_val = 'val')] = dummy
    
    
    # cod (mu, sigma)
    n_loc = dict(var_cod = 'mu', n_or_val = 'n')
    n = ds.param_cod.loc[n_loc]
    ds.param_cod.loc[dict(n_or_val = 'val')] = ds.param_cod.loc[
            dict(n_or_val = 'val')].where(n > 100)
    
    ds.param_cod.loc[dict(n_or_val = 'val')]= interpolateNN(ds.param_cod.loc[dict(n_or_val = 'val')], 'mu_dN')
    
    
    # cth ()
    n_loc = dict(var_cth_bm = 'mu1', n_or_val = 'n', est = 'coef')
    n = ds.param_cth_bm.loc[n_loc]
    ds.param_cth_bm.loc[dict(est = 'coef', n_or_val = 'val')] = ds.param_cth_bm.loc[dict(est = 'coef', n_or_val = 'val')].where(
        n > 1000)
    
    ds.param_cth_bm.loc[dict(est = 'coef', n_or_val = 'val')] = interpolateNN(ds.param_cth_bm.loc[dict(est = 'coef', n_or_val = 'val')], 'mu_hN')
    
    
    # clear sky contains nans as well for cth
    n = ds.cs_n.sum(dim = ('cs_mu_dN', 'mu_csf'))
    ds.cs_param_cth_bm.loc[dict(est = 'coef')] = ds.cs_param_cth_bm.sel(est = 'coef').where(n > 1000)
    
    ds.cs_param_cth_bm.loc[dict(est = 'coef')] = interpolateNN(ds.cs_param_cth_bm.loc[dict(est = 'coef')], 'cs_mu_hN')
    

    ds = ds.loc[dict(n_or_val = 'val')]

    ds_temp_d = ds.param_cth_bm.sel(mu_h = ds.mu_h[ds.mu_h < 14700], est = 'coef').interpolate_na(dim = 'mu_d', method = 'nearest', fill_value="extrapolate")
    ds['param_cth_bm'] = ds_temp_d
    ds = ds.interpolate_na(dim = 'mu_h', method = 'nearest', fill_value="extrapolate")
    
    
# =============================================================================
#     Generate X0
# =============================================================================
    
    image = xr.open_dataset('../data/start_image0.nc')
    N = 10
    
# =============================================================================
#     Simulation
# =============================================================================

    X0 = image.isel(i = np.arange(N), j = np.arange(N))
    X0['t'] = 0
    X = sim_model2(X0, 5, ds)
    
    X.to_netcdf(loc_sim + f'simulation2_{method}_T{T}_N{N}')
    
    
    
    
    
    