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
def all_combinations(any_list):
    return itertools.chain.from_iterable(
        itertools.combinations(any_list, i)
        for i in range(len(any_list) + 1))


def mutual_infdd(x, n = 50, binx = None, biny = None):
    X = x[0]
    Y = x[1:]
    
    N = len(X)
    
    if binx is None:
        binx = n
    
    if biny is None:
        biny = n
    
    px, xedges = np.histogramdd(X, bins = binx)
    py, yedges = np.histogramdd(Y, bins = biny)
    
    bins = [*xedges, *yedges]

    h, edges= np.histogramdd(x, bins = bins)
    plt.clf()
    
    
    shape = [n] * len(x)
    lists = [np.arange(s) for s in shape]

    px = px/N
    py = py/N
    h = h/N
    I = 0

    for i in itertools.product(*lists):
#         print(i)
#         print(h)
        x = i[0]
        y = i[1:]
        p_xy = h[i]
        p_x = px[x]
        p_y = py[y]
        if p_xy == 0:
            term = 0
        else:
            term = p_xy * np.log(p_xy / (p_x * p_y) )
        I += term
    return I


# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f if (len(line) > 1) & (line[0] != '#') ])
    
    loc_model_data = input['loc_model2_data']
    loc_model = input['loc_model2']
    
   
    n = 100
    
    
    prop = {'n' :n }
    
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
    
    
    # set clear sky points to 0, -1
    df.loc[df.z_t == 1, 'h_t'] = -1000    
    df.loc[df.z_t == 1, 'd_t'] = 0    
    df.loc[df.csf_t == 1, 'h_bar_t'] = -1000    
    df.loc[df.csf_t == 1, 'd_bar_t'] = 0    
    
    df.loc[df.z_t_next == 1, 'h_t_next'] = -1000    
    df.loc[df.z_t_next == 1, 'd_t_next'] = 0    


# =============================================================================
#   Clouds
# =============================================================================
    print(f'h_max in data is {df[["h_t", "h_t_next"]].max().max():.2f} m')
    
    df.loc[df.h_t > ml.h_max , 'h_t'] = ml.h_max
    df.loc[df.h_t_next > ml.h_max , 'h_t_next'] = ml.h_max
    
    print(f'h_max is {df[["h_t", "h_t_next"]].max().max():.2f} m')
    
    




# =============================================================================
#   Mutual information
# =============================================================================

    next_state_var = ['h_t_next', 'd_t_next', 'z_t_next']
    
    current_state_var = ['h_t', 'd_t']
    g_cand = ['h_bar_t', 'd_bar_t', 'csf_t']


    row_lists = []
    for var in next_state_var:
        if var != 'z_t_next':
            X = df.loc[df.z_t_next == 0, var]
        else: 
            X = df.loc[:, var]
            
        for cand in all_combinations(g_cand):
            expl_var = [*current_state_var, *cand]
            
            if var != 'z_t_next':
                Y = [df.loc[df.z_t_next == 0, column] for column in expl_var]
            else: 
                Y = [df[column] for column in expl_var]
        
            x =[X,*Y] 

            m = mutual_infdd(x, n = n)
            
            row = dict(X = var, Y = expl_var, I = m)
            row_lists.append(row)
            print(var, cand, '\n', m,'\n')
        
            
        m = mutual_infdd([X,X], n = n)
        
        row = dict(X = var, Y = var, I = m)
        row_lists.append(row)            
        print('max = ', m, '\n\n')

            
    df_mi= pd.DataFrame(row_lists)
    df_mi.to_csv(loc_model + 'mutualinf.csv')
            







        
