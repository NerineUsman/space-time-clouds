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
import matplotlib.pyplot as plt
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
    
       
    dH = 500 
    dD = .5
    dHN = 500
    dDN = .5
    dCSF = .4
    
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
#   from clouds
# =============================================================================
    df_c = df.loc[(df.z_t == 0)]
    df_c = df_c.copy()
# =============================================================================
#   Cloud to cloud    
# =============================================================================
    df_cc = df.loc[(df.z_t == 0) & (df.z_t_next == 0) ]
    df_cc.insert(len(df_cc.columns), 'dh', df_cc.apply(lambda row: row.h_t_next - row.h_t, axis = 1))
    df_cc.insert(len(df_cc.columns), 'dd', df_cc.apply(lambda row: row.d_t_next - row.d_t, axis = 1))
    df_cc = df_cc.copy()
        
# =============================================================================
#   Clear sky to cloud
# =============================================================================

    df_sc = df.loc[(df.z_t == 1) & (df.z_t_next == 0) ]
    df_sc = df_sc.copy()
    
# =============================================================================
#   To clear sky
# =============================================================================

    df_s = df.loc[(df.z_t_next == 1)]
    df_s = df_s.copy()

# =============================================================================
#   Model1 Explorative
# =============================================================================

    bin_h = cs_bin_hN = [1000, 7000, 13000] # m

    bin_d = cs_bin_dN = [ -.5, 1.5, 3.5]
    bin_hN = [-1500, -750, 0 , 750, 1500]
    bin_dN = [-.8, -.4, 0, .4, .8]
    bin_csf = [.2, .5, .8]
    
    cs_bin_hN = [1000, 4000, 7000, 10000, 13000]
    cs_bin_dN = [ -.5, .5, 1.5, 2.5, 3.5]    
    
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
#  0. Overall
# =============================================================================

    # contains histogram for cloud distribution
    # and transition cross matrix for cloud / clear sky  transitions
    

    freq, xedges, yedges, __ = plt.hist2d(df_c.d_t, 
                                              df_c.h_t, 
                                              bins = [n_bins, n_bins])
 
    ds_hist = xr.Dataset(
    data_vars=dict(
    ),
    coords = dict(
        dedges = xedges,
        hedges = yedges
        )
    )
    
    ds_hist['freq'] = (['h', 'd'], freq)
    ds_hist.to_netcdf(loc_model + 'expl_hist_clouds.nc')
    plt.show()
    plt.figure(2)
    freq, xedges, yedges, __ = plt.hist2d(df_c.d_bar_t, 
                                              df_c.h_bar_t, 
                                              bins = [n_bins, n_bins])
 
    ds_hist = xr.Dataset(
    data_vars=dict(
    ),
    coords = dict(
        dedges = xedges,
        hedges = yedges
        )
    )
    plt.show()
    
    
    ds_hist['freq'] = (['h', 'd'], freq)
    ds_hist.to_netcdf(loc_model + 'expl_hist_cloudN.nc')

    T = T_total = pd.crosstab(df.ct, df.ct_next, rownames=['from'], colnames=[ 'to'], normalize = 'index', margins = True)
    T.to_csv(loc_model + 'expl_transition_ctypes.csv')
    
    
    
# =============================================================================
#     Create ds with all estimates
# =============================================================================


    ds = xr.Dataset(
    data_vars=dict(
        n = (['mu_h', 'mu_d', 'mu_hN', 'mu_dN', 'mu_csf'], np.empty(bin_coord_dim) * np.nan), 
        p_cs = (['mu_h', 'mu_d', 'mu_hN', 'mu_dN', 'mu_csf'], np.empty(bin_coord_dim) * np.nan),         
        freq = (['mu_h', 'mu_d', 'mu_hN', 'mu_dN', 'mu_csf', 'h_next', 'd_next'], np.empty((*bin_coord_dim, n_bins, n_bins)) * np.nan),
        hedges = (['mu_h', 'mu_d', 'mu_hN', 'mu_dN', 'mu_csf', 'bin_edges'], np.empty((*bin_coord_dim, n_bins + 1)) * np.nan),
        dedges = (['mu_h', 'mu_d', 'mu_hN', 'mu_dN', 'mu_csf', 'bin_edges'], np.empty((*bin_coord_dim, n_bins + 1)) * np.nan),
        param_cod = (['mu_h', 'mu_d', 'mu_hN', 'mu_dN', 'mu_csf', 'var_cod'], np.empty((*bin_coord_dim, 2)) * np.nan),
        param_cth_bm = (['mu_h', 'mu_d', 'mu_hN', 'mu_dN', 'mu_csf', 'est', 'var_cth_bm'], np.empty((*bin_coord_dim, 2, 5)) * np.nan),
        param_cth_b = (['mu_h', 'mu_d', 'mu_hN', 'mu_dN', 'mu_csf', 'est', 'method', 'var_cth_b'], np.empty((*bin_coord_dim, 2, 2, 2)) * np.nan),
        cs_n = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf'], np.empty(cs_bin_coord_dim) * np.nan), 
        cs_p_cs = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf'], np.empty(cs_bin_coord_dim) * np.nan),         
        cs_freq = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf', 'h_next', 'd_next'], np.empty((*cs_bin_coord_dim, n_bins, n_bins)) * np.nan),
        cs_hedges = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf', 'bin_edges'], np.empty((*cs_bin_coord_dim, n_bins + 1)) * np.nan),
        cs_dedges = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf', 'bin_edges'], np.empty((*cs_bin_coord_dim, n_bins + 1)) * np.nan),
        cs_param_cod = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf', 'var_cod'], np.empty((*cs_bin_coord_dim, 2)) * np.nan),
        cs_param_cth_bm = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf', 'est', 'var_cth_bm'], np.empty((*cs_bin_coord_dim, 2, 5)) * np.nan),
        cs_param_cth_b = (['cs_mu_hN', 'cs_mu_dN', 'mu_csf', 'est', 'method', 'var_cth_b'], np.empty((*cs_bin_coord_dim, 2, 2, 2)) * np.nan),
    ),
    coords=dict(
        mu_h=bin_h,
        mu_d=bin_d,
        mu_hN = bin_hN,
        mu_dN = bin_dN,
        cs_mu_hN = cs_bin_hN,
        cs_mu_dN = cs_bin_dN,
        mu_csf = bin_csf, 
        method = ['ML', 'MoM'],
        est = ['coef', 'bse'],
        var_cth_bm = ['mu1', 'nu1', 'mu2', 'mu2', 'p'],
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
    
    bins_iter = itertools.product(cs_bin_hN, cs_bin_dN, bin_csf)
    n_b = np.product([cs_bin_coord_dim])
    
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
        
        if n == 0 : 
            continue
        p_cs = len(df_temp.loc[df_temp.z_t_next == 1])/ len(df_temp)

        ds.cs_p_cs.loc[dic] = p_cs
        
        # to cloud
        df_temp = df_temp.loc[df_temp.z_t_next == 0]
        
        if len(df_temp) < 2:
            continue
        

        # contains histogram of cth, cod and joint from clear sky
    
        freq_hd, xedges, yedges, __ = plt.hist2d(df_temp.d_t_next, 
                                                  df_temp.h_t_next, 
                                                  bins = [n_bins, n_bins])
     
        ds_hist = xr.Dataset(
        data_vars=dict(
        ),
        coords = dict(
            dedges = xedges,
            hedges = yedges
            )
        )
        
        # save values    
        ds.cs_freq.loc[dic] = freq_hd
        ds.cs_dedges.loc[dic] = xedges
        ds.cs_hedges.loc[dic] = yedges
        
        # fit
        #   cod
        

        d_next = df_temp.d_t_next
        mu = d_next.mean()
        sigma = np.sqrt(n /  ( n - 1) * d_next.var())
        
        ds.cs_param_cod.loc[dic] = [mu, sigma]
    
        
        #   cth
        if len(df_temp) < 9:
            continue
        mix_beta_fit = fitMixBetaCTH(df_temp.h_t_next)
        params, bse = me.fitMixBetaCTHtoParams(mix_beta_fit)  ## fix such that p >.5
        x = ml.CTHtoUnitInt(df_temp.h_t_next)
        
        ds.cs_param_cth_bm.loc[dic] =  [params, bse]
        
        beta_fit = me.fitBetaCTH(df_temp.h_t_next)
        ## Beta MoM
        param_mom = ml.MoM_sb(x)
        
        ds.cs_param_cth_b.loc[dic] = [[[beta_fit.params[0], beta_fit.params[1]],
                                  [param_mom[0], param_mom[1]]],
                                 [[beta_fit.bse[0], beta_fit.bse[1]],
                                  [np.nan, np.nan]]]
        
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
    
    for i, (h, d, hN, dN, csf) in zip(range(n_b), bins_iter):   
        dic = dict(mu_h = h,
                         mu_d = d, 
                         mu_hN = hN,
                         mu_dN = dN, 
                         mu_csf = csf
                         )
        
        df_temp = df.loc[(df.z_t == 0) & 
                         (h - dH <= df.h_t) & (df.h_t <= h  + dH) & 
                         (d - dD <= df.d_t) & (df.d_t <= d  + dD) & 
                         (hN - dHN <= (df.h_bar_t - df.h_t)) & ((df.h_bar_t - df.h_t) <= hN  + dHN) & 
                         (dN - dDN <= (df.d_bar_t - df.d_t)) & ((df.d_bar_t - df.d_t) <= dN  + dDN) & 
                         (csf - dCSF <= df.csf_t) & (df.csf_t <= csf  + dCSF)  
                         ].copy()
        
        n  = len(df_temp)
        ds.n.loc[dic] = n
        if n == 0:
            continue
          
        
        # # to clear sky
        p_cs = len(df_temp.loc[df_temp.z_t_next == 1])/ len(df_temp)
        ds.p_cs.loc[dic] = p_cs
        
        # # to cloud
        df_temp = df_temp.loc[df_temp.z_t_next == 0]
        
        if len(df_temp) < 2:
            continue        # # contains histogram of cth, cod and joint from clear sky
        
        freq_hd, xedges, yedges, __ = plt.hist2d(df_temp.d_t_next, 
                                                  df_temp.h_t_next, 
                                                  bins = [n_bins, n_bins])
         
        
        # # save values    
        ds.freq.loc[dic] = freq_hd
        ds.dedges.loc[dic] = xedges
        ds.hedges.loc[dic] = yedges
        
        # fit
  
        #   cod


        d_next = df_temp.d_t_next
        mu = d_next.mean()
        sigma = np.sqrt(n /  ( n - 1) * d_next.var())
        
        ds.param_cod.loc[dic] =  [mu, sigma]
        
        
        # #   cth
        if len(df_temp) < 9:
            continue
        mix_beta_fit = fitMixBetaCTH(df_temp.h_t_next)
        params, bse = me.fitMixBetaCTHtoParams(mix_beta_fit)  ## fix such that p >.5
        x = ml.CTHtoUnitInt(df_temp.h_t_next)
        
        ds.param_cth_bm.loc[dic] = [params, bse]
        
        beta_fit = me.fitBetaCTH(df_temp.h_t_next)
        ## Beta MoM
        param_mom = ml.MoM_sb(x)
        
        ds.param_cth_b.loc[dic] = [[[beta_fit.params[0], beta_fit.params[1]],
                                  [param_mom[0], param_mom[1]]],
                                  [[beta_fit.bse[0], beta_fit.bse[1]],
                                  [np.nan, np.nan]]]
                

    ds.to_netcdf(loc_model + 'expl_local_param.nc')


## maybe also add test statistics
  
    
# # # =============================================================================
# # #  4. fit for multiple bins
# # # =============================================================================

# ########
#         ## COD ##
# ########
#     print('cod local fits')

#     mu_hat = np.zeros((len(mu_h) * len(mu_d)))
#     sigma_hat = np.zeros((len(mu_h) * len(mu_d)))
#     n_clouds = np.zeros((len(mu_h) *len(mu_d)))
    
    
#     for i, b, (d, h) in zip(range(len(bins)), bins, itertools.product(mu_d, mu_h)):
#         # filter cloud to cloud on from within bin
#         df_temp = me.df_temp_at(df_cc, h, d, dH, dD, N = N)
                
#         df_temp = df_temp.copy()
#         n = len(df_temp)
#         n_clouds[i] = n
        
#         if n <= 1:
#             mu_hat[i] = np.nan
#             sigma_hat[i] = np.nan
#             continue
        
#         mu_hat[i] = df_temp.d_t_next.mean()
#         sigma_hat[i] = np.sqrt(n / (n-1) * df_temp.d_t_next.var())
        
        
#     ds['mu'] = (['mu_h', 'mu_d'], mu_hat.reshape(n_d, n_h).T)
#     ds['sigma'] = (['mu_h', 'mu_d'], sigma_hat.reshape(n_d, n_h).T)
#     # ds['n_cc'] = (['mu_h', 'mu_d'], n_clouds.reshape(n_d, n_h).T)
    
#     ds.mu.attrs['method'] = 'ML'
#     ds.sigma.attrs['method'] = 'ML'
#     ds.mu.attrs['distr'] = 'norm'
#     ds.sigma.attrs['distr'] = 'norm'
#     print('cth local fits')

    
#     cth_beta_params = np.zeros((n_h * n_d , 7))
#     cth_conv_flag = np.zeros((n_h * n_d, 2))
    
    # param_names_bmix = ['alpha1',
    #                 'beta1', 
    #                 'alpha2', 
    #                 'beta2', 
    #                 'p'
    #               ]
    
    
#     for i, param in zip(range(len(param_names_bmix)), param_names_bmix):
#         ds[param] = (['mu_h', 'mu_d', 'est'], np.empty((n_h, n_d, 2)) * np.nan)
    
#     ds['n_cc'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
#     ds['AIC_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
#     ds['BIC_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
#     ds['KS_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
#     ds['CM_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
#     ds['AD_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
#     ds['conv_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
    
#     param_names_b = ['alpha', 'beta']

#     for i, param in zip(range(len(param_names_b)), param_names_b):
#         ds[param] = (['mu_h', 'mu_d', 'method', 'est'], np.empty((n_h, n_d, 2, 2)) * np.nan)
    
#     ds['AIC_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
#     ds['BIC_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
#     ds['KS_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
#     ds['CM_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
#     ds['AD_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
#     ds['conv_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)

#     j = 0
    
#     # for i, b in zip(range(len(bins)), bins):
#     for h, d in itertools.product(mu_h, mu_d):
#         # filter cloud to cloud on from within bin
#         df_temp = me.df_temp_at(df_cc, h, d, dH, dD, N = N)

#         df_temp = df_temp.copy()
#         n = len(df_temp)
        
#         ds['n_cc'].loc[dict(mu_h = h, mu_d = d)] = n

        
#         if n <= 9:
#             # cth_beta_params[j] = np.nan
#             # cth_conv_flag[j] = np.nan
#             continue
        
#         # print('try_beta_fit')
        
#         ## Mix Beta ML
#         mix_beta_fit = me.fitMixBetaCTH(df_temp.h_t_next)
#         params, bse = me.fitMixBetaCTHtoParams(mix_beta_fit)  ## fix such that p >.5
#         x = ml.CTHtoUnitInt(df_temp.h_t_next)
        
#         for i, param in zip(range(5), param_names_bmix):
#             ds[param].loc[dict(mu_h = h, mu_d = d, est = 'coef')] = params[i]
#             ds[param].loc[dict(mu_h = h, mu_d = d, est = 'bse')] = bse[i]
            
#         ds['conv_bm'].loc[dict(mu_h = h, mu_d = d)] = mix_beta_fit.mle_retvals['converged']
        
#         ds['AIC_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.AIC(5, mix_beta_fit.llf)
#         ds['BIC_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.BIC(5, mix_beta_fit.llf, n)
#         ds['KS_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.KS(x, ml.cdf_bmix, args = (mix_beta_fit.params[:])).statistic
#         ds['CM_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.CM(x, ml.cdf_bmix, args = (mix_beta_fit.params[:])).statistic
#         ds['AD_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.AD(x, ml.cdf_bmix, *mix_beta_fit.params)


#         ## Beta ML
#         beta_fit = me.fitBetaCTH(df_temp.h_t_next)
#         ## Beta MoM
#         param_mom = ml.MoM_sb(x)
        
#         for i, param in zip(range(2), param_names_b):
#             ds[param].loc[dict(mu_h = h, mu_d = d, method = 'ML', est = 'coef')] = beta_fit.params[i]
#             ds[param].loc[dict(mu_h = h, mu_d = d, method = 'ML', est = 'bse')] = beta_fit.bse[i]  

#             ds[param].loc[dict(mu_h = h, mu_d = d, method = 'MoM')] = param_mom[i]  
            
#         ds['conv_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = beta_fit.mle_retvals['converged']
            
#         ds['AIC_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.AIC(2, beta_fit.llf)
#         ds['BIC_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.BIC(2, beta_fit.llf, n)
#         ds['KS_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.KS(x, ml.cdf_b, args = (beta_fit.params[:])).statistic
#         ds['CM_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.CM(x, ml.cdf_b, args = (beta_fit.params[:])).statistic
#         ds['AD_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.AD(x, ml.cdf_b, *beta_fit.params)

#         ds['KS_b'].loc[dict(mu_h = h, mu_d = d, method = 'MoM')] = ts.KS(x, ml.cdf_b, args = (param_mom[:])).statistic
#         ds['CM_b'].loc[dict(mu_h = h, mu_d = d, method = 'MoM')] = ts.CM(x, ml.cdf_b, args = (param_mom[:])).statistic
#         ds['AD_b'].loc[dict(mu_h = h, mu_d = d, method = 'MoM')] = ts.AD(x, ml.cdf_b, *param_mom)

        
#     ds.to_netcdf(loc_model + 'expl_local_param.nc')
        
