#!/usr/bin/env python3
# coding: utf-8

"""
Created on Mon May 16 11:54:24 2022

@author: Nerine
"""

# import modules
import os, sys
import pandas as pd


sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib/')





# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model.txt'


# functions



# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f if (len(line) > 1) & (line[0] != '#')])
    
    loc_model3_data = input['loc_model3_data']
    loc_model3 = input['loc_model3']
    
    
    dh = 200
    dd = .2
    
    dH = 400 
    dD = .4
    
    N = 5000
    
    prop = {'dH' : dH, 'dD' : dD, 'N' : N }
    
    prop = '_'.join([f'{x}={prop[x]}' for x in prop]).replace('.' ,'_')
    loc_model1 = loc_model3 + prop
    
    
    # combine df's from all days in model 1 data
    files = [loc_model3_data + f for f in os.listdir(loc_model3_data) if (os.path.isfile(os.path.join(loc_model3_data, f)))]
    files
    
    dfs =[]
    for file in files:
        df = pd.read_csv(file)   
        dfs.append(df)
    df = pd.concat(dfs)
    



    # transition cross matrix for cloud / clear sky  transitions, for all possible neighbourhood
    # states
    

    T = pd.crosstab([df.g, df.ct], df.ct_next, rownames=['g', 'from'], colnames=[ 'to'], normalize = 'index', margins = True)
    T.to_csv(loc_model3 + 'transition_ctypes.csv')
