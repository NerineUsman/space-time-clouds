#!/usr/bin/env python3
# coding: utf-8

"""
Created on Tue Mar 15 12:20:15 2022

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

    bin_h = cs_bin_hN = [1000] # m

    bin_d = cs_bin_dN = [1.5]
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
#   From Cloud
# =============================================================================
    
    h = 1000
    d = 1.5
        
    df = df.loc[(df.z_t == 0) & 
                     (h - dH <= df.h_t) & (df.h_t <= h  + dH) & 
                     (d - dD <= df.d_t) & (df.d_t <= d  + dD)  
                     ].copy()
 
    df.to_csv(loc_model + 'expl_model2_small_data_set.csv')
