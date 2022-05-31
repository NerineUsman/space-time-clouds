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
        input = dict([line.split() for line in f if (len(line) > 1) & (line[0] != '#') ])
      
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
    df_cs = df.loc[ (df.cloud == 'cloud') & ((df.cloud_next == 'clear sky') | (df.cloud_next == 'cloud')) ]
    df_cs = df_cs.copy()

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

# =============================================================================
#  0. Overall
# =============================================================================

    ds = xr.Dataset(
    data_vars=dict(
    ),
    coords = dict(
    )
    )
    
    
# =============================================================================
#  1. Clear sky to clear sky
# =============================================================================
    p_cscs = len(df.loc[(df.cloud == 'clear sky') & (df.cloud_next == 'clear sky')]) / \
                len (df.loc[df.cloud == 'clear sky'])
              
    ds['theta1'] = (['cs_to_cs'], [p_cscs])
    ds.theta1.attrs['var_names'] = 'p_cs'

# =============================================================================
#  2. Cloud to Clear sky
# =============================================================================
    
    df_cs['to_clear_sky'] = (df_cs.cloud_next == 'clear sky')
    
    print('2. p_cs global fit')
    ## ml estimation COD deep params
    model1_p = ml.MyDepPcsML(df_cs.to_clear_sky,df_cs[['h_t', 'd_t']])
    sm_ml_p = model1_p.fit()
    df_p = pd.DataFrame(sm_ml_p._cache)
    df_p['coef'] = sm_ml_p.params
    df_p['names'] = model1_p.exog_names
    df_p.to_csv(loc_model1 + 'model1_p_cs.csv')
    
    ds['theta2'] = (['c_to_cs'], sm_ml_p.params)
    ds.theta2.attrs['var_names'] = model1_p.exog_names

    print(sm_ml_p.summary())
     
# =============================================================================
#  3. clear sky to cloud
# =============================================================================
    print('3. clear sky to cloud')
    mu, sigma = fitNormal(df_sc.d_t_next)
    (alpha, beta), conv_b = fitBetaCTH(df_sc.h_t_next)
    (alpha1, beta1, alpha2, beta2, p), conv_mb = fitMixBetaCTH(df_sc.h_t_next)
    
    dic = { 'alpha' : alpha,
            'beta' : beta,
            'alpha1' : alpha1,
            'beta1' : beta1,
            'alpha2' : alpha2,
            'beta2': beta2,
            'p' : p,
            'mu' : mu,
            'sigma' : sigma,
            'conv_b' : conv_b,
            'conv_mb' : conv_mb,
            }
    
    df_param_cstc = pd.Series(dic)
    df_param_cstc.to_csv(loc_model1 + 'cstc_param.csv')
    
    ds['theta3'] = (['cs_to_c'], df_param_cstc.values)
    ds.theta3.attrs['var_names'] = list(df_param_cstc.index.values)
    
# =============================================================================
#   4. cloud to cloud 
# =============================================================================

    print('4. cod global fit')
    ## ml estimation COD deep params
    df_cc['constant'] = 1
    df_cc['hd'] = df_cc.h_t * df_cc.d_t
    model1_cod = ml.MyDepNormML(df_cc.d_t_next,df_cc[['constant','h_t', 'd_t', 'hd']])
    sm_ml_cod = model1_cod.fit(
                        start_params = [1, .001, 0.9, 0, .7, .001])
    df_cod = pd.DataFrame(sm_ml_cod._cache)
    df_cod['coef'] = sm_ml_cod.params
    df_cod['names'] = model1_cod.exog_names
    df_cod.to_csv(loc_model1 + 'glob_c_to_c_cod.csv')
    
    print(sm_ml_cod.summary())
    
    print('4. cth global fit')
    # ## ml estimation COD deep params
    # model1_cth = ml.MyDepMixBetaML(df_cc.h_t_next,df_cc[['h_t', 'd_t']])
    # sm_ml_cth = model1_cth.fit()
    # df_cth = pd.DataFrame(sm_ml_cth._cache)
    # df_cth['coef'] = sm_ml_cth.params
    # df_cth['names'] = model1_cth.exog_names
    # df_cth.to_csv(loc_model1 + 'model1_cth.csv')
    
    # print(sm_ml_cth.summary())
    
    
    ## TODO change second ones to cth
    
    ds['theta4'] = (['c_to_c'], np.concatenate([sm_ml_cod.params, sm_ml_cod.params]))
    ds.theta4.attrs['var_names'] = model1_cod.exog_names + model1_cod.exog_names
        
    ds.to_netcdf(loc_model1 + 'glob_theta.nc')

   
