#!/usr/bin/env python3
# coding: utf-8

"""
Created on Wed Dec 15 11:54:44 2021

@author: Nerine
"""

import pickle
import pandas as pd
import os, sys
sys.path.insert(0, './space-time-clouds/lib')
import ml_estimation as ml


src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model1.txt'

with open(input_file) as f:
    input = dict([line.split() for line in f])

loc_model1_data = input['loc_model1_data']
loc_fig = input['loc_fig']
loc_model1 = input['loc_model1']

with open(loc_model1 + 'model1_cod.pickle', 'rb') as f:
    x = pickle.load(f)

print(x.summary())
df_cod = pd.DataFrame(x._cache)
df_cod['coef'] = x.params
df_cod.to_csv(loc_model1 + 'model1_cod.csv')
    
with open(loc_model1 + 'model1_cth.pickle', 'rb') as f:
    x = pickle.load(f)

print(x.summary())
df_cth = pd.DataFrame(x._cache)
df_cth['coef'] = x.params
df_cth.to_csv(loc_model1 + 'model1_cth.csv')

