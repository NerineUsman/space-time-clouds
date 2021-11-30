#!/usr/bin/env python3
# coding: utf-8

"""
Created on Mon Nov 29 11:05:38 2021

@author: Nerine
"""

# import modules
import pickle
import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd


from datetime import datetime, timedelta


# variables
input_file = 'input_model1_local.txt'

# functions
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)



# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f])
    
    loc_model1_data = input['loc_model1_data']
    
    # combine df's from all days in model 1 data
    files = [loc_model1_data + f for f in os.listdir(loc_model1_data) if (os.path.isfile(os.path.join(loc_model1_data, f)))]
    files
    
    dfs =[]
    for file in files:
        df = pd.read_csv(file)   
        print(len(df))
        dfs.append(df)
    df = pd.concat(dfs)
    df
    
    
    

    

