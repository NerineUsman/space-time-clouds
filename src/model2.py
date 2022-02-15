#!/usr/bin/env python3
# coding: utf-8

"""
Created on Mon Nov 29 11:05:38 2021

@author: Nerine
"""

# import modules
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import itertools

sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib/')
import ml_estimation as ml
import ml_estimation2 as ml2
import test_stat as ts
import model1_explore as me


# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model.txt'
# input_file = './space-time-clouds/src/input_model1.txt'

hlim = [0, 16] #km
dlim = [-1.5, 5.1] #log (d)

# functions
def fitMixBetaCTH(y):
    h_ = ml.CTHtoUnitInt(y)
    
    if len(h_) > 1e4:
        h_ = h_.sample(int(1e4))
        print('decreased sample size')
    # print(y.min(), y.max())
    # print(h_.min(), h_.max())
    
    mu1 = h_.mean()
    nu1 = 20
    mu2 = 1 - mu1
    nu2 = 20
    start_params = [mu1, nu1, mu2, nu2, .8 ]
    
    ml_manual = ml2.MyMixBetaML(h_, h_).fit(disp = 0,
                start_params = start_params)
    
    return ml_manual #params, conv


# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f if (len(line) > 1) & (line[0] != '#') ])
    
    loc_model_data = input['loc_model2_data']
    loc_model = input['loc_model2']

    dh = 200
    dd = .2
       
    dH = 500 
    dD = .5
    dHN = 500
    dDN = .5
    dCSF = .1
    
    N = 5000
    
    n_bins = 50

    
    prop = {'dH' : dH, 'dD' : dD, 'N' : N }
    
    prop = '_'.join([f'{x}={prop[x]}' for x in prop]).replace('.' ,'_')
    loc_model = loc_model + prop
    
    
    # combine df's from all days in model 1 data
    files = [loc_model_data + f for f in os.listdir(loc_model_data) if (os.path.isfile(os.path.join(loc_model_data, f)))]
    files
    
    dfs =[]
    for file in files:
        df = pd.read_csv(file)   
        dfs.append(df)
    df = pd.concat(dfs)
    
    # delete rows to or from invalid point
    df = df[df.z_t.notna()]
    df = df[df['z_t_next'].notna()]
    
# =============================================================================
#   Clouds
# =============================================================================
    print(f'h_max in data is {df[["h_t", "h_t_next"]].max().max():.2f} m')
    
    df.loc[df.h_t > ml.h_max , 'h_t'] = ml.h_max
    df.loc[df.h_t_next > ml.h_max , 'h_t_next'] = ml.h_max
    
    print(f'h_max is {df[["h_t", "h_t_next"]].max().max():.2f} m')


# =============================================================================
#   Model1 Explorative
# =============================================================================


    bin_h = np.arange(0 + dh/2, ml.h_max - dh/2, dh) # m
    bin_d = np.arange(-1, 4.5, dd)
    bin_hN = np.arange(-1500, 1500 + dHN/2, dHN / 2 )
    bin_dN = np.arange(-.8, .8 + dDN/2, dDN / 2 )
    bin_csf = np.linspace(0, 1, 10)
    
    cs_bin_hN = np.arange(0 + dHN/2, ml.h_max, dHN / 2 )
    cs_bin_dN = np.arange(-1, 4.5, dDN)  
    
    
    n_h = len(bin_h)
    n_d = len(bin_d)
    n_hN = len(bin_hN)
    n_dN = len(bin_dN)
    n_csf = len(bin_csf)
    n_cs_hN = len(cs_bin_hN)
    n_cs_dN = len(cs_bin_dN)
        
    bin_coord_dim = (n_h, n_d, n_hN, n_dN, n_csf)    
    cs_bin_coord_dim = (n_cs_hN, n_cs_dN, n_csf)
    
    
    
# =============================================================================
#     Create ds with all estimates
# =============================================================================


    ds = xr.Dataset(
    data_vars=dict(
        p_cs = (['mu_h', 'mu_d', 'mu_csf', 'n_or_val'], np.empty((n_h, n_d, n_csf, 2)) * np.nan),         
        param_cod = (['mu_h', 'mu_d', 'mu_dN', 'var_cod', 'n_or_val'], np.empty((n_h, n_d, n_dN, 2, 2)) * np.nan),
        param_cth_bm = (['mu_h', 'mu_d', 'mu_hN', 'est', 'var_cth_bm', 'n_or_val'], np.empty((n_h, n_d, n_hN, 2, 5, 2)) * np.nan),
        cs_n = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf'], np.empty(cs_bin_coord_dim) * np.nan), 
        cs_p_cs = (['mu_csf'], np.empty(n_csf) * np.nan),         
        cs_param_cod = (['cs_mu_dN', 'var_cod'], np.empty((n_cs_dN, 2)) * np.nan),
        cs_param_cth_bm = (['cs_mu_hN', 'est', 'var_cth_bm'], np.empty((n_cs_hN, 2, 5)) * np.nan),
    ),
    coords=dict(
        mu_h=bin_h,
        mu_d=bin_d,
        mu_hN = bin_hN,
        mu_dN = bin_dN,
        cs_mu_hN = cs_bin_hN,
        cs_mu_dN = cs_bin_dN,
        mu_csf = bin_csf, 
        n_or_val = ['val', 'n'],
        method = ['ML', 'MoM'],
        est = ['coef', 'bse'],
        var_cth_bm = ['mu1', 'nu1', 'mu2', 'nu2', 'p'],
        var_cod = ['mu', 'sigma'],
        var_cth_b = ['alpha', 'beta']
    ),
    # attrs=dict(dh = dh, dd = dd
    # ),
    )

    param_names_b = ['alpha', 'beta']
    param_names_bmix = ['mu1',
                    'nu1', 
                    'mu2', 
                    'nu2', 
                    'p'
                  ]

# =============================================================================
#   From clear sky
# =============================================================================
    print('From Clear sky\n')
    bins_iter = itertools.product(cs_bin_hN, cs_bin_dN, bin_csf)
    n_b = np.product([cs_bin_coord_dim])
    
    ###
    # n
    ###
    for i, (hN, dN, csf) in zip(range(n_b), bins_iter):   
        
        dic = dict(cs_mu_hN = hN,
                         cs_mu_dN = dN, 
                         mu_csf = csf
                         )        
        
        df_temp = df.loc[(df.z_t == 1) & 
                         (hN - dHN <= df.h_bar_t) & (df.h_bar_t <= hN  + dHN) & 
                         (dN - dDN <= df.d_bar_t) & (df.d_bar_t <= dN  + dDN) & 
                         (csf - dCSF <= df.csf_t) & (df.csf_t <= csf  + dCSF)  
                         ].copy()
        
        n  = len(df_temp)
        # to clear sky
        
        ds.cs_n.loc[dic] = n
            

                
    print('\tp_cs')

    
    ####
    # p_cs
    ####
    
    for i, csf in zip(range(n_csf), bin_csf):   
        
        dic = dict(mu_csf = csf
                         )        
        
        df_temp = df.loc[(df.z_t == 1) & 
                         (csf - dCSF <= df.csf_t) & (df.csf_t <= csf  + dCSF)  
                         ].copy()
        
        n  = len(df_temp)
        # to clear sky
                
        if n == 0 : 
            continue
        p_cs = len(df_temp.loc[df_temp.z_t_next == 1])/ len(df_temp)

        ds.cs_p_cs.loc[dic] = p_cs

    ####
    # COD
    ####
    print('\tCOD')

                
    for i, dN in zip(range(n_cs_dN), cs_bin_dN):   
        
        dic = dict(cs_mu_dN = dN, 
                         )        
        
        df_temp = df.loc[(df.z_t == 1) & (df.z_t_next == 0) &
                         (dN - dDN <= df.d_bar_t) & (df.d_bar_t <= dN + dDN) 
                         ].copy()
        
        
        if len(df_temp) < 2:
            continue
        

        # fit
        #   cod
        d_next = df_temp.d_t_next
        mu = d_next.mean()
        sigma = np.sqrt(n /  ( n - 1) * d_next.var())
        
        ds.cs_param_cod.loc[dic] = [mu, sigma]
        

        
    ####
    # CTH
    ####
    print('\tCTH')
                
    for i, hN in zip(range(n_cs_hN), cs_bin_hN):   
        
        dic = dict(cs_mu_hN = hN
                         )        
        
        df_temp = df.loc[(df.z_t == 1) & (df.z_t_next == 0) &
                         (hN - dHN <= df.h_bar_t) & (df.h_bar_t <= hN  + dHN) 
                         ].copy()
        
        #   cth
        if len(df_temp) < 9:
            continue
        
        mix_beta_fit = fitMixBetaCTH(df_temp.h_t_next)
        params, bse = me.fitMixBetaCTHtoParams(mix_beta_fit)  ## fix such that p >.5
        x = ml.CTHtoUnitInt(df_temp.h_t_next)
        
        ds.cs_param_cth_bm.loc[dic] =  [params, bse]
        
        # beta_fit = me.fitBetaCTH(df_temp.h_t_next)
        # ## Beta MoM
        # param_mom = ml.MoM_sb(x)
        
        # ds.cs_param_cth_b.loc[dic] = [[[beta_fit.params[0], beta_fit.params[1]],
        #                           [param_mom[0], param_mom[1]]],
        #                          [[beta_fit.bse[0], beta_fit.bse[1]],
        #                           [np.nan, np.nan]]]
        
        
        
    # special case csf = 1 
    
    df_temp = df.loc[(df.z_t == 1) & (df.z_t_next == 0) &
                     (df.csf_t == 1)
                     ].copy()

    
    if len(df_temp) > 2:
        
        # fit
        #   cod
        d_next = df_temp.d_t_next
        mu = d_next.mean()
        sigma = np.sqrt(n /  ( n - 1) * d_next.var())
        
        ds['cs_csf_param_cod'] = (['var_cod'], [mu, sigma])
    if len(df_temp > 9):
        
        mix_beta_fit = fitMixBetaCTH(df_temp.h_t_next)
        params, bse = me.fitMixBetaCTHtoParams(mix_beta_fit)  ## fix such that p >.5
        x = ml.CTHtoUnitInt(df_temp.h_t_next)
        
        ds['cs_csf_param_cth_bm'] =  (['est','var_cth_bm'],[params, bse])
        
    # n, freq_hd, hedges, dedges, p_cs, param_cod, param_b, param_bm

# =============================================================================
#   From Cloud
# =============================================================================
    
    bins_iter = itertools.product(bin_h, bin_d, bin_hN, bin_dN, bin_csf)
    n_b = np.product([bin_coord_dim])
    
    print('\nFrom Cloud \n')
    
   
    ####
    # p_cs
    ####
    print('\tp_cs')
    bins_iter = itertools.product(bin_h, bin_d, bin_csf)
    n_b = np.product([bin_coord_dim])
    
    
    for i, (h, d, csf) in zip(range(n_b), bins_iter):   
        dic = dict(mu_h = h,
                         mu_d = d, 
                         mu_csf = csf
                         )
        
        df_temp = df.loc[(df.z_t == 0) & 
                         (h - dH <= df.h_t) & (df.h_t <= h  + dH) & 
                         (d - dD <= df.d_t) & (df.d_t <= d  + dD) & 
                         (csf - dCSF <= df.csf_t) & (df.csf_t <= csf  + dCSF)  
                         ].copy()
        
        n  = len(df_temp)
        
        dic['n_or_val'] = 'n'
        ds.p_cs.loc[dic] = n
        
        if n == 0:
            continue
          
        
        # # to clear sky
        p_cs = len(df_temp.loc[df_temp.z_t_next == 1])/ len(df_temp)
        
        dic['n_or_val'] = 'val'
        ds.p_cs.loc[dic] = p_cs

    ####
    # COD
    ####          
    print('\tCOD')
    bins_iter = itertools.product(bin_h, bin_d, bin_dN)
    n_b = np.product([bin_coord_dim])
    
    for i, (h, d, dN) in zip(range(n_b), bins_iter):   
        dic = dict(mu_h = h,
                         mu_d = d, 
                         mu_dN = dN, 
                         )
        
        df_temp = df.loc[(df.z_t == 0) & (df.z_t_next == 0) &
                         (h - dH <= df.h_t) & (df.h_t <= h  + dH) & 
                         (d - dD <= df.d_t) & (df.d_t <= d  + dD) & 
                         (dN - dDN <= (df.d_bar_t - df.d_t)) & ((df.d_bar_t - df.d_t) <= dN  + dDN)
                         ].copy()
        
        n  = len(df_temp)
        
        dic['n_or_val'] = 'n'
        ds.param_cod.loc[dic] = n

        if len(df_temp) < 2:
            continue        

  
        #   cod
        d_next = df_temp.d_t_next
        mu = d_next.mean()
        sigma = np.sqrt(n /  ( n - 1) * d_next.var())
        
        dic['n_or_val'] = 'val'
        ds.param_cod.loc[dic] =  [mu, sigma]
        
    
    ####
    # CTH
    ####      
    print('\tCTH')    
    bins_iter = itertools.product(bin_h, bin_d,  bin_hN)
    n_b = np.product([bin_coord_dim])
    
    for i, (h, d, hN) in zip(range(n_b), bins_iter):   
        dic = dict(mu_h = h,
                         mu_d = d, 
                         mu_hN = hN,
                         )
        
        df_temp = df.loc[(df.z_t == 0) & (df.z_t_next == 0) &
                         (h - dH <= df.h_t) & (df.h_t <= h  + dH) & 
                         (d - dD <= df.d_t) & (df.d_t <= d  + dD) & 
                         (hN - dHN <= (df.h_bar_t - df.h_t)) & ((df.h_bar_t - df.h_t) <= hN  + dHN) 
                         ].copy()
        
        n = len(df_temp)
        dic['n_or_val'] = 'n'
        ds.param_cth_bm.loc[dic] = n
        
        # #   cth
        if len(df_temp) < 9:
            continue
        
        mix_beta_fit = fitMixBetaCTH(df_temp.h_t_next)
        params, bse = me.fitMixBetaCTHtoParams(mix_beta_fit)  ## fix such that p >.5
        x = ml.CTHtoUnitInt(df_temp.h_t_next)
        
        dic['n_or_val'] = 'val'
        ds.param_cth_bm.loc[dic] = [params, bse]
        
        # beta_fit = me.fitBetaCTH(df_temp.h_t_next)
        # ## Beta MoM
        # param_mom = ml.MoM_sb(x)
        
        # ds.param_cth_b.loc[dic] = [[[beta_fit.params[0], beta_fit.params[1]],
        #                           [param_mom[0], param_mom[1]]],
        #                           [[beta_fit.bse[0], beta_fit.bse[1]],
        #                           [np.nan, np.nan]]]
                

    ds.to_netcdf(loc_model + 'local_param.nc')


## maybe also add test statistics

        
