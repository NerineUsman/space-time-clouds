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

def neighborhood(ds, i, j,
                 n = 1 # neighbor degree
                 ):

    xlow = max(0, i - n)
    xupper = min(ds.dims['i'], i + n + 1)
    
    ylow = max(0, j - n)
    yupper = min(ds.dims['j'], j + n + 1)
    
    N = (slice(xlow, xupper, 1), slice(ylow, yupper, 1))
    
    return N

def simulation(T, X0, **kwargs):
    X = X0.copy(deep = True)
    t = 0
    for t in range(T):
        print(f't = {t}')
        if t == 0 :
            Xt = X0.copy(deep = True)
        else:
            Xt = X.sel(t = t)
        X_next = X0.copy(deep = True)
        for i, j in itertools.product(Xt.i.data, Xt.j.data):
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
            X_next.z.loc[pix] = x_next[0]
            X_next.h.loc[pix] = x_next[1]
            X_next.d.loc[pix] = x_next[2]
        #     X_next.loc[pix]
        X_next['t'] = t+1
        X = xr.concat([X, X_next], dim = 't')
        X_next = None
    return X


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
    
    with open(loc_clean_data + 'clean_dates.pickle', 'rb') as f:
        dates = pickle.load(f)
    
    # print(dates)
    date = dates.date
    
    start_date = datetime.strptime('10-12-2020', '%d-%m-%Y')
    end_date = datetime.strptime('11-12-2020', '%d-%m-%Y')
    idx = (start_date < date)  & (date < end_date) 
    
    dates = dates[idx].reset_index(drop = True)
    
    file = dates.file_name.loc[0]
    images = xr.open_dataset(file)
    images
    
    # Only keep points which are in the right area and do contain a wind speed
    dss = []
    for file in dates.file_name:
        images = xr.open_dataset(file)
        dss.append(images)
    
    images = xr.concat(dss, 't')
    
    invalid =np.sum(images.ct == 0, axis = (1,2))
    images = images.where(invalid < 10000, drop = True)
    
    images = images.rename(dict(x = 'i', y = 'j'))
    images = images.rename(dict(cth = 'h', cod = 'd'))
    
    images['i'] = range(len(images.i))
    images['j'] = range(len(images.j))
    images['d'] = np.log(images.d)
    images['z'] = (['t', 'i', 'j'], np.select([images.ct == 1, images.ct > 1], [1, 0], np.nan))
    images['z'] = images.z.where(~np.isnan(images.d) & ~np.isnan(images.h) )
    images = images.where(images.z == 0)
    
    
# =============================================================================
#     Simulation
# =============================================================================

    X0 = images.sel(t = images.t[0], i = images.i[:N], j = images.j[:N])
    X0['t'] = 0
    X = simulation(T, X0, method = method)
    
    X.to_netcdf(loc_sim + f'simulation2_{method}_T{T}_N{N}_{file.rsplit("/")[-1]}')