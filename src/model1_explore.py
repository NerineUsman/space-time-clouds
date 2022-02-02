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
import test_stat as ts




# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model.txt'
# input_file = './space-time-clouds/src/input_model1.txt'

hlim = [0, 16] #km
dlim = [-1.5, 5.1] #log (d)

# functions
def state_bins(mu_h, mu_d, delta_h = 300, delta_d =.1):
    """
    Parameters
    ----------
    mu_h : list
        cth coordinates. in km
    mu_d : list
        cod coordinates. log(cod)
    delta_h : float, optional
        half the binwidth in cth. The default is 300.
    delta_d : TYPE, optional
        half the binwidth for cod. The default is .1.

    Returns
    -------
    bins : list
        list of bins ([h_min, hmax],[dmin, dmax]).

    """
    mu_h, mu_d = np.meshgrid(mu_h, mu_d)
    mu_h, mu_d = [x.flatten() for x in [mu_h, mu_d]]
    
    bincenter = (mu_h, mu_d)
    bins = [[[h - delta_h, h + delta_h], [d - delta_d, d + delta_d]] for h,d in zip(mu_h, mu_d)]
    return bins, bincenter


# =============================================================================
#  1. clear sky to clear sky
# =============================================================================

def p_cs_to_cs(df):
    return  len(df.loc[(df.cloud == 'clear sky') & (df.cloud_next == 'clear sky')]) / \
                len (df.loc[df.cloud == 'clear sky'])

def fitNormal(y):
    n = len(y)
    mu = y.mean()
    sigma= np.sqrt(n / (n-1) * y.var())
    return mu, sigma

def fitBetaCTH(y):
    h_ = ml.CTHtoUnitInt(y)
    
    if len(h_) > 1e4:
        h_ = h_.sample(int(1e4))
        
    ml_manual = ml.MyBetaML(h_, h_).fit(disp = 0,
                start_params = [1, 1])
    return ml_manual #.params, conv

def paramsFitBetaCTH(y):
    fit = fitBetaCTH(y)
       
    conv = fit.mle_retvals['converged']
    return fit.params, conv
    

def fixInvalidP(params):
    alpha1, beta1, alpha2, beta2, p = params
    if p > 1: 
        p = 1
        alpha2, beta2 = np.nan, np.nan
    elif p < 0:
        p = 0
        alpha1, beta1 = np.nan, np.nan
    return (alpha1, beta1, alpha2, beta2, p)


def switchBeta(params):
    alpha1, beta1, alpha2, beta2, p = params
    if p < .5:
        alpha1, beta1, alpha2, beta2, p = alpha2, beta2, alpha1, beta1, 1-p
    return np.array([alpha1, beta1, alpha2, beta2, p])


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
    alpha1, beta1, alpha2, beta2 = mu1 * nu1 , nu1 - mu1 * nu1,  mu2 * nu2 , nu2 - mu2 * nu2
    start_params = [alpha1, beta1, alpha2, beta2, .8 ]
    
    ml_manual = ml.MyMixBetaML(h_, h_).fit(disp = 0,
                start_params = start_params)
    

    return ml_manual #params, conv

def paramsFitMixBetaCTH(y):
    fit = fitMixBetaCTH(y)
    params = fixInvalidP(fit.params)
    params = switchBeta(params)
    conv = fit.mle_retvals['converged']
    return params, conv

def fitMixBetaCTHtoParams(fit):
    params = fixInvalidP(fit.params)
    p = params[-1]
    params = switchBeta(params)
    bse = fit.bse
    bse[:-1] = switchBeta(np.hstack([fit.bse[:-1], p]))[:-1]
    return params, bse
    
    

def meanBeta(alpha, beta):
    return alpha / (alpha + beta)


def df_temp_at(df, h, d, delta_h, delta_d, N = 500, **kwargs):
    (b,), b_c = state_bins(h, d, delta_h = delta_h, delta_d = delta_d)
    
    df_t = df.loc[(df.h_t > b[0][0]) & (df.h_t < b[0][1])
                        & (df.d_t > b[1][0]) & (df.d_t < b[1][1])] 
    df_t= df_t.copy()    
    df_t.attrs['dH'] = delta_h
    df_t.attrs['dD'] = delta_d

    if N is None:
        # print('Nis None')
        df_t = df_t
    elif len(df_t) < N:
        # print(len(df_t), delta_h, delta_d)
        df_t = df_temp_at(df, h, d, delta_h + 200, delta_d + .1, N = N, **kwargs)
    return df_t



# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f])
    
    loc_model1_data = input['loc_model1_data']
    loc_model1 = input['loc_model1']
    
    
    dh = 200
    dd = .2
    
    dH = 400 
    dD = .4
    
    N = 5000
    
    prop = {'dH' : dH, 'dD' : dD, 'N' : N }
    
    prop = '_'.join([f'{x}={prop[x]}' for x in prop]).replace('.' ,'_')
    loc_model1 = loc_model1 + prop
    
    
    # combine df's from all days in model 1 data
    files = [loc_model1_data + f for f in os.listdir(loc_model1_data) if (os.path.isfile(os.path.join(loc_model1_data, f)))]
    files
    
    dfs =[]
    for file in files:
        df = pd.read_csv(file)   
        dfs.append(df)
    df = pd.concat(dfs)
    
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
    df_c = df.loc[(df.cloud == 'cloud')]
    df_c = df_c.copy()
# =============================================================================
#   Cloud to cloud    
# =============================================================================
    df_cc = df.loc[(df.cloud == 'cloud') & (df.cloud_next == 'cloud') ]
    df_cc.insert(len(df_cc.columns), 'dh', df_cc.apply(lambda row: row.h_t_next - row.h_t, axis = 1))
    df_cc.insert(len(df_cc.columns), 'dd', df_cc.apply(lambda row: row.d_t_next - row.d_t, axis = 1))
    df_cc = df_cc.copy()
        
# =============================================================================
#   Clear sky to cloud
# =============================================================================

    df_sc = df.loc[(df.cloud == 'clear sky') & (df.cloud_next == 'cloud') ]
    df_sc = df_sc.copy()
    
# =============================================================================
#   To clear sky
# =============================================================================

    df_s = df.loc[((df.cloud == 'clear sky') | (df.cloud == 'cloud')) & (df.cloud_next == 'clear sky')  ]
    df_s = df_s.copy()

# =============================================================================
#   Model1 Explorative
# =============================================================================

    mu_h = np.arange(0 + dh/2, ml.h_max - dh/2, dh) # m
    mu_d = np.arange(-1, 4.5, dd)
    n_h = len(mu_h)
    n_d = len(mu_d)

    
    bins, bin_center = state_bins(mu_h, mu_d, delta_h = dH/2, delta_d = dD/2)
    
    
# =============================================================================
#  0. Overall
# =============================================================================

    # contains histogram for cloud distribution
    # and transition cross matrix for cloud / clear sky  transitions
    
    nbins = 50

    freq, xedges, yedges, __ = plt.hist2d(df_c.d_t, 
                                             df_c.h_t, 
                                             bins = [nbins, nbins])
 
    ds_hist = xr.Dataset(
    data_vars=dict(
    ),
    coords = dict(
        dedges = xedges,
        hedges = yedges
        )
    )
    
    ds_hist['freq'] = (['h', 'd'], freq)
    ds_hist.to_netcdf(loc_model1 + 'expl_hist_clouds.nc')


    T = T_total = pd.crosstab(df.ct, df.ct_next, rownames=['from'], colnames=[ 'to'], normalize = 'index', margins = True)
    T.to_csv(loc_model1 + 'expl_transition_ctypes.csv')
    
# =============================================================================
#  1. Clear sky to clear sky
# =============================================================================

# =============================================================================
#  2. Cloud to Clear sky
# =============================================================================
    # contains local estimates for probability to cs: p_(h,d)
    
    print('clear sky probability')

    p_cs = np.zeros((len(mu_h) * len(mu_d)))
    n_clouds = np.zeros((len(mu_h) *len(mu_d)))
    dH_i = np.zeros((len(mu_h) *len(mu_d)))
    dD_i = np.zeros((len(mu_h) *len(mu_d)))
    
    
    for i, b, (d, h) in zip(range(len(bins)), bins, itertools.product(mu_d, mu_h)):
        # filter cloud to cloud  from within bin
        df_temp = df_temp_at(df_cc, h, d, dH, dD, N = N)
        
        df_s_temp = df_temp_at(df_s, h, d, df_temp.attrs['dH'], df_temp.attrs['dD'], N = None)
     
        
        # df_temp = df_cc.loc[(df_cc.h_t > b[0][0]) & (df_cc.h_t < b[0][1])
        #                     & (df_cc.d_t > b[1][0]) & (df_cc.d_t < b[1][1])] 
        
        # # filter cloud to clear sky  from within bin
        # df_s_temp = df_s.loc[(df_s.h_t > b[0][0]) & (df_s.h_t < b[0][1])
        #                     & (df_s.d_t > b[1][0]) & (df_s.d_t < b[1][1])]
        
        n = len(df_temp)
        n_cs = len(df_s_temp)
        
        n_clouds[i] = n + n_cs
        dH_i[i] = df_temp.attrs['dH']
        dD_i[i] = df_temp.attrs['dD']

        if n + n_cs > 0:
            p_cs[i] = n_cs / (n + n_cs) 
        else:
            p_cs[i] = np.nan
        

    p_cscs = p_cs_to_cs(df)
    
    ds = xr.Dataset(
    data_vars=dict(
    ),
    coords=dict(
        mu_h=mu_h,
        mu_d=mu_d,
        # distr = ['Norm', 'Beta','MixBeta'],
        method = ['ML', 'MoM'],
        est = ['coef', 'bse']
    ),
    attrs=dict(dh = dh, dd = dd,  p_cscs = p_cscs
    ),
    )
    
    ds['p_cs'] = (['mu_h', 'mu_d'], p_cs.reshape(n_d, n_h).T)
    ds['n_c'] = (['mu_h', 'mu_d'], n_clouds.reshape(n_d, n_h).T)
    ds['dH'] = (['mu_h', 'mu_d'], dH_i.reshape(n_d, n_h).T)
    ds['dD'] = (['mu_h', 'mu_d'], dD_i.reshape(n_d, n_h).T)



# =============================================================================
#  3. clear sky to cloud
# =============================================================================

    # contains histogram of cth, cod and joint from clear sky
    nbins = 50

    freq_hd, xedges, yedges, __ = plt.hist2d(df_sc.d_t_next, 
                                             df_sc.h_t_next, 
                                             bins = [nbins, nbins])
 
    ds_hist = xr.Dataset(
    data_vars=dict(
    ),
    coords = dict(
        dedges = xedges,
        hedges = yedges
        )
    )
    
    ds_hist['freq'] = (['h', 'd'], freq_hd)
    ds_hist.to_netcdf(loc_model1 + 'expl_hist_cs_to_c.nc')

    
    # # to visualize:
    # X, Y = np.meshgrid(ds_hist.dedges[:-1], ds_hist.hedges[:-1])
    # hist2 = plt.hist2d(X.flatten(), Y.flatten(), bins = [ds_hist.dedges, ds_hist.hedges],
    #                    weights = ds_hist.freq.data.T.flatten())
    # plt.show()
    
# =============================================================================
#   4. cloud to cloud distribution from a few bins
# =============================================================================

    mu_h_e = [  500.,  4500.,  8500., 12500.] #[1e3, 6e3, 9e3, 12e3] #
    mu_d_e = [0, 1, 2, 3]
    

    bins_e, (bincenter_h, bincenter_d) = state_bins(mu_h_e, mu_d_e, delta_h = dh/2, delta_d = dd/2)
    
    df_example_bins = pd.DataFrame(columns = np.append(df_cc.columns.values,
                                                        ['bincenter_h',
                                                        'bincenter_d']))
    
    df_example_bins_fit = pd.DataFrame(columns = ['bincenter_h',
                                                  'bincenter_d',
                                                  'alpha',
                                                  'beta',
                                                  'conv_b',
                                                  'alpha1',
                                                  'beta1',
                                                  'alpha2',
                                                  'beta2',
                                                  'p', 
                                                  'conv_mb',
                                                  'mu',
                                                  'sigma',
                                                  'p_cs'])
    
    for i, b, (d, h) in zip(range(len(bins_e)), bins_e, itertools.product(mu_d_e, mu_h_e)):       
        print(b)
        # filter cloud to cloud on from within bin
        
        df_temp = df_temp_at(df_cc, h, d, dH, dD, N = N)
        
        df_s_temp = df_temp_at(df_s, h, d, df_temp.attrs['dH'], df_temp.attrs['dD'], N = None)
        
        # df_temp = df_cc.loc[(df_cc.h_t > b[0][0]) & (df_cc.h_t < b[0][1])
        #                     & (df_cc.d_t > b[1][0]) & (df_cc.d_t < b[1][1])]
        # df_temp = df_temp.copy()
        # df_s_temp = df_s.loc[(df_s.h_t > b[0][0]) & (df_s.h_t < b[0][1])
        #                     & (df_s.d_t > b[1][0]) & (df_s.d_t < b[1][1])]
        
        n_c = len(df_temp)
        n_cs = len(df_s_temp)
        
        if n_c + n_cs > 0:
            p_cs = n_cs / (n_c + n_cs) 
        else:
            p_cs = np.nan
            
        if len(df_temp) <= 1:
            print('not enough data')
            continue
        
        df_temp['bincenter_h'] = bincenter_h[i]
        df_temp['bincenter_d'] = bincenter_d[i]
        
        df_example_bins = df_example_bins.append(df_temp)
        
        mu, sigma = fitNormal(df_temp.d_t_next)
        (alpha, beta), conv_b = paramsFitBetaCTH(df_temp.h_t_next)
        (alpha1, beta1, alpha2, beta2, p), conv_mb = paramsFitMixBetaCTH(df_temp.h_t_next)
        
        dic = {'bincenter_h': bincenter_h[i], 
                'bincenter_d': bincenter_d[i],
                'alpha' : alpha,
                'beta' : beta,
                'conv_b' : conv_b,
                'alpha1' : alpha1,
                'beta1' : beta1,
                'alpha2' : alpha2,
                'beta2': beta2,
                'p' : p,
                'conv_mb' : conv_mb,
                'mu' : mu,
                'sigma' : sigma,
                'p_cs' : p_cs
              }
        
        df_example_bins_fit = df_example_bins_fit.append(dic, ignore_index=True)
        
    df_example_bins.to_csv(loc_model1 + 'expl_example_bin_data.csv')
    df_example_bins_fit.to_csv(loc_model1 + 'expl_examble_bin_fit.csv')
    
    
# # =============================================================================
# #  4. fit for multiple bins
# # =============================================================================

########
        ## COD ##
########
    print('cod local fits')

    mu_hat = np.zeros((len(mu_h) * len(mu_d)))
    sigma_hat = np.zeros((len(mu_h) * len(mu_d)))
    n_clouds = np.zeros((len(mu_h) *len(mu_d)))
    
    
    for i, b, (d, h) in zip(range(len(bins)), bins, itertools.product(mu_d, mu_h)):
        # filter cloud to cloud on from within bin
        df_temp = df_temp_at(df_cc, h, d, dH, dD, N = N)
                
        df_temp = df_temp.copy()
        n = len(df_temp)
        n_clouds[i] = n
        
        if n <= 1:
            mu_hat[i] = np.nan
            sigma_hat[i] = np.nan
            continue
        
        mu_hat[i] = df_temp.d_t_next.mean()
        sigma_hat[i] = np.sqrt(n / (n-1) * df_temp.d_t_next.var())
        
        
    ds['mu'] = (['mu_h', 'mu_d'], mu_hat.reshape(n_d, n_h).T)
    ds['sigma'] = (['mu_h', 'mu_d'], sigma_hat.reshape(n_d, n_h).T)
    # ds['n_cc'] = (['mu_h', 'mu_d'], n_clouds.reshape(n_d, n_h).T)
    
    ds.mu.attrs['method'] = 'ML'
    ds.sigma.attrs['method'] = 'ML'
    ds.mu.attrs['distr'] = 'norm'
    ds.sigma.attrs['distr'] = 'norm'
    print('cth local fits')

    
    cth_beta_params = np.zeros((n_h * n_d , 7))
    cth_conv_flag = np.zeros((n_h * n_d, 2))
    
    param_names_bmix = ['alpha1',
                    'beta1', 
                    'alpha2', 
                    'beta2', 
                    'p'
                  ]
    
    
    for i, param in zip(range(len(param_names_bmix)), param_names_bmix):
        ds[param] = (['mu_h', 'mu_d', 'est'], np.empty((n_h, n_d, 2)) * np.nan)
    
    ds['n_cc'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
    ds['AIC_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
    ds['BIC_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
    ds['KS_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
    ds['CM_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
    ds['AD_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
    ds['conv_bm'] = (['mu_h', 'mu_d'], np.empty((n_h, n_d)) * np.nan)
    
    param_names_b = ['alpha', 'beta']

    for i, param in zip(range(len(param_names_b)), param_names_b):
        ds[param] = (['mu_h', 'mu_d', 'method', 'est'], np.empty((n_h, n_d, 2, 2)) * np.nan)
    
    ds['AIC_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
    ds['BIC_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
    ds['KS_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
    ds['CM_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
    ds['AD_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)
    ds['conv_b'] = (['mu_h', 'mu_d', 'method'], np.empty((n_h, n_d, 2)) * np.nan)

    j = 0
    
    # for i, b in zip(range(len(bins)), bins):
    for h, d in itertools.product(mu_h, mu_d):
        # filter cloud to cloud on from within bin
        df_temp = df_temp_at(df_cc, h, d, dH, dD, N = N)

        df_temp = df_temp.copy()
        n = len(df_temp)
        
        ds['n_cc'].loc[dict(mu_h = h, mu_d = d)] = n

        
        if n <= 9:
            # cth_beta_params[j] = np.nan
            # cth_conv_flag[j] = np.nan
            continue
        
        # print('try_beta_fit')
        
        ## Mix Beta ML
        mix_beta_fit = fitMixBetaCTH(df_temp.h_t_next)
        params, bse = fitMixBetaCTHtoParams(mix_beta_fit)  ## fix such that p >.5
        x = ml.CTHtoUnitInt(df_temp.h_t_next)
        
        for i, param in zip(range(5), param_names_bmix):
            ds[param].loc[dict(mu_h = h, mu_d = d, est = 'coef')] = params[i]
            ds[param].loc[dict(mu_h = h, mu_d = d, est = 'bse')] = bse[i]
            
        ds['conv_bm'].loc[dict(mu_h = h, mu_d = d)] = mix_beta_fit.mle_retvals['converged']
        
        ds['AIC_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.AIC(5, mix_beta_fit.llf)
        ds['BIC_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.BIC(5, mix_beta_fit.llf, n)
        ds['KS_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.KS(x, ml.cdf_bmix, args = (mix_beta_fit.params[:])).statistic
        ds['CM_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.CM(x, ml.cdf_bmix, args = (mix_beta_fit.params[:])).statistic
        ds['AD_bm'].loc[dict(mu_h = h, mu_d = d)] = ts.AD(x, ml.cdf_bmix, *mix_beta_fit.params)


        ## Beta ML
        beta_fit = fitBetaCTH(df_temp.h_t_next)
        ## Beta MoM
        param_mom = ml.MoM_sb(x)
        
        for i, param in zip(range(2), param_names_b):
            ds[param].loc[dict(mu_h = h, mu_d = d, method = 'ML', est = 'coef')] = beta_fit.params[i]
            ds[param].loc[dict(mu_h = h, mu_d = d, method = 'ML', est = 'bse')] = beta_fit.bse[i]  

            ds[param].loc[dict(mu_h = h, mu_d = d, method = 'MoM')] = param_mom[i]  
            
        ds['conv_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = beta_fit.mle_retvals['converged']
            
        ds['AIC_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.AIC(2, beta_fit.llf)
        ds['BIC_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.BIC(2, beta_fit.llf, n)
        ds['KS_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.KS(x, ml.cdf_b, args = (beta_fit.params[:])).statistic
        ds['CM_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.CM(x, ml.cdf_b, args = (beta_fit.params[:])).statistic
        ds['AD_b'].loc[dict(mu_h = h, mu_d = d, method = 'ML')] = ts.AD(x, ml.cdf_b, *beta_fit.params)

        ds['KS_b'].loc[dict(mu_h = h, mu_d = d, method = 'MoM')] = ts.KS(x, ml.cdf_b, args = (param_mom[:])).statistic
        ds['CM_b'].loc[dict(mu_h = h, mu_d = d, method = 'MoM')] = ts.CM(x, ml.cdf_b, args = (param_mom[:])).statistic
        ds['AD_b'].loc[dict(mu_h = h, mu_d = d, method = 'MoM')] = ts.AD(x, ml.cdf_b, *param_mom)

        
    ds.to_netcdf(loc_model1 + 'expl_local_param.nc')
        
