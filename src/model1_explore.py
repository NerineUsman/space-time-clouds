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

sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib')
import ml_estimation as ml




# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model1.txt'
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
    conv = ml_manual.mle_retvals['converged']
    return ml_manual.params, conv

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
    
    params = fixInvalidP(ml_manual.params)
    params = switchBeta(params)
    conv = ml_manual.mle_retvals['converged']
    return params, conv

# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f])
    
    loc_model1_data = input['loc_model1_data']
    loc_fig = input['loc_fig']
    loc_model1 = input['loc_model1']
    
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
    dh = 300
    dd = .3
    mu_h = np.arange(1e3, 14e3, dh) # m
    mu_d = np.arange(-1, 4, dd)
    n_h = len(mu_h)
    n_d = len(mu_d)

    
    bins, bin_center = state_bins(mu_h, mu_d, delta_h = dh/2, delta_d = dd/2)
    
    
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


    T = pd.crosstab(df.cloud, df.cloud_next, rownames=['from'], colnames=[ 'to'], margins = True)
    T.to_csv(loc_model1 + 'expl_transition_c_cs.csv')
    
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
    
    
    for i, b in zip(range(len(bins)), bins):
        # filter cloud to cloud  from within bin
        df_temp = df_cc.loc[(df_cc.h_t > b[0][0]) & (df_cc.h_t < b[0][1])
                            & (df_cc.d_t > b[1][0]) & (df_cc.d_t < b[1][1])] 
        n = len(df_temp)
        
        # filter cloud to clear sky  from within bin
        df_s_temp = df_s.loc[(df_s.h_t > b[0][0]) & (df_s.h_t < b[0][1])
                            & (df_s.d_t > b[1][0]) & (df_s.d_t < b[1][1])]
        
        n_cs = len(df_s_temp)
        
        n_clouds[i] = n + n_cs

        if n + n_cs > 0:
            p_cs[i] = n_cs / (n + n_cs) 
        else:
            p_cs[i] = np.nan
        

    p_cscs = len(df.loc[(df.cloud == 'clear sky') & (df.cloud_next == 'clear sky')]) / \
                len (df.loc[df.cloud == 'clear sky'])
    
    ds = xr.Dataset(
    data_vars=dict(
    ),
    coords=dict(
        mu_h=mu_h,
        mu_d=mu_d,
    ),
    attrs=dict(dh = dh, dd = dd,   p_cscs = p_cscs
    ),
    )
    
    ds['p_cs'] = (['mu_h', 'mu_d'], p_cs.reshape(n_d, n_h).T)
    ds['n_c'] = (['mu_h', 'mu_d'], n_clouds.reshape(n_d, n_h).T)


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

    mu_h_e = [1e3, 6e3, 9e3, 12e3] #
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
    
    for i, b in zip(range(len(bins_e)), bins_e):       
        print(b)
        # filter cloud to cloud on from within bin
        df_temp = df_cc.loc[(df_cc.h_t > b[0][0]) & (df_cc.h_t < b[0][1])
                            & (df_cc.d_t > b[1][0]) & (df_cc.d_t < b[1][1])]
        df_temp = df_temp.copy()
        df_s_temp = df_s.loc[(df_s.h_t > b[0][0]) & (df_s.h_t < b[0][1])
                            & (df_s.d_t > b[1][0]) & (df_s.d_t < b[1][1])]
        
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
        (alpha, beta), conv_b = fitBetaCTH(df_temp.h_t_next)
        (alpha1, beta1, alpha2, beta2, p), conv_mb = fitMixBetaCTH(df_temp.h_t_next)
        
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
    
    
    for i, b in zip(range(len(bins)), bins):
        # filter cloud to cloud on from within bin
        df_temp = df_cc.loc[(df_cc.h_t > b[0][0]) & (df_cc.h_t < b[0][1])
                            & (df_cc.d_t > b[1][0]) & (df_cc.d_t < b[1][1])] 
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
    ds['n_cc'] = (['mu_h', 'mu_d'], n_clouds.reshape(n_d, n_h).T)
    
    
    print('cth local fits')

    
    cth_beta_params = np.zeros((n_h * n_d , 7))
    cth_conv_flag = np.zeros((n_h * n_d, 2))
    
    for i, b in zip(range(len(bins)), bins):
        # filter cloud to cloud on from within bin
        df_temp = df_cc.loc[(df_cc.h_t > b[0][0]) & (df_cc.h_t < b[0][1])
                            & (df_cc.d_t > b[1][0]) & (df_cc.d_t < b[1][1])] 
        df_temp = df_temp.copy()
        n = len(df_temp)
        
        if n <= 9:
            cth_beta_params[i] = np.nan
            cth_conv_flag[i] = np.nan
            continue
        
        # print('try_beta_fit')
        
        cth_beta_params[i,:2] , cth_conv_flag[i, 0] = fitBetaCTH(df_temp.h_t_next)
        cth_beta_params[i, 2:7], cth_conv_flag[i, 1] = fitMixBetaCTH(df_temp.h_t_next)
        
    
    param_names = ['alpha', 'beta',
                'alpha1',
                  'beta1', 
                  'alpha2', 
                  'beta2', 
                  'p'
                  ]
    
    for i, param in zip(range(i), param_names):
        ds[param] = (['mu_h', 'mu_d'], cth_beta_params[:,i].reshape(n_d, n_h).T)
    
    
    ds['conv_b'] = (['mu_h', 'mu_d'], cth_conv_flag[:, 0].reshape(n_d, n_h).T)
    ds['conv_mb'] = (['mu_h', 'mu_d'], cth_conv_flag[:, 1].reshape(n_d, n_h).T)
    ds.to_netcdf(loc_model1 + 'expl_local_param.nc')
    
    
    
    # print('cod global fit')
    # ## ml estimation COD deep params
    # df_cc['constant'] = 1
    # df_cc['hd'] = df_cc.h_t * df_cc.d_t
    # model1_cod = ml.MyDepNormML(df_cc.d_t_next,df_cc[['constant','h_t', 'd_t', 'hd']])
    # sm_ml_cod = model1_cod.fit(
    #                     start_params = [1, .001, 0.9, 0, .7, .001])
    # df_cod = pd.DataFrame(sm_ml_cod._cache)
    # df_cod['coef'] = sm_ml_cod.params
    # df_cod['names'] = model1_cod.exog_names
    # df_cod.to_csv(loc_model1 + 'glob_c_to_c_cod.csv')
    
    # print(sm_ml_cod.summary())
    
    # print('cth global fit')
    # ## ml estimation COD deep params
    # model1_cth = ml.MyDepMixBetaML(df_cc.h_t_next,df_cc[['h_t', 'd_t']])
    # sm_ml_cth = model1_cth.fit()
    # df_cth = pd.DataFrame(sm_ml_cth._cache)
    # df_cth['coef'] = sm_ml_cth.params
    # df_cth['names'] = model1_cth.exog_names
    # df_cth.to_csv(loc_model1 + 'model1_cth.csv')
    
    # print(sm_ml_cth.summary())
    

# =============================================================================
#     Clear sky probability
# =============================================================================
# =============================================================================
#  3. Clear sky to cloud global
# =============================================================================
       

    # mu, sigma = fitNormal(df_sc.d_t_next)
    # (alpha, beta), conv_b = fitBetaCTH(df_sc.h_t_next)
    # (alpha1, beta1, alpha2, beta2, p), conv_mb = fitMixBetaCTH(df_sc.h_t_next)
    
    # dic = { 'alpha' : alpha,
    #         'beta' : beta,
    #         'conv_b' : conv_b,
    #         'alpha1' : alpha1,
    #         'beta1' : beta1,
    #         'alpha2' : alpha2,
    #         'beta2': beta2,
    #         'p' : p,
    #         'conv_mb' : conv_mb,
    #         'mu' : mu,
    #         'sigma' : sigma,
    #         }
    
    # df_param_cstc = pd.Series(dic)
    # df_param_cstc.to_csv(loc_model1 + 'cstc_param.csv')
    
    # ### prob to cs global
    
    # # =============================================================================
    # #   To clear sky
    # # =============================================================================

    # df_cs = df.loc[ (df.cloud == 'cloud') & ((df.cloud_next == 'clear sky') | (df.cloud_next == 'cloud')) ]
    # df_cs = df_cs.copy()
    # df_cs['to_clear_sky'] = (df_cs.cloud_next == 'clear sky')
    
    # print('p_cs global fit')
    # ## ml estimation COD deep params
    # model1_cod = ml.MyDepPcsML(df_cs.to_clear_sky,df_cs[['h_t', 'd_t']])
    # sm_ml_cod = model1_cod.fit()
    # df_cod = pd.DataFrame(sm_ml_cod._cache)
    # df_cod['coef'] = sm_ml_cod.params
    # df_cod['names'] = model1_cod.exog_names
    # df_cod.to_csv(loc_model1 + 'model1_p_cs.csv')
    
    # print(sm_ml_cod.summary())
        
