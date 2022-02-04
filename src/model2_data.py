#!/usr/bin/env python3
# coding: utf-8

"""
Created on Wed Feb  2 15:25:38 2022

@author: Nerine
"""

# import modules
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime, timedelta



sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib/')

import model1_data as md

# 
# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model.txt'


def neighborhood(ds, i, x, y,
                 n = 1 # neighbor degree
                 ):

    xlow = max(0, x - n)
    xupper = min(ds.dims['x'], x + n + 1)
    
    ylow = max(0, y - n)
    yupper = min(ds.dims['y'], y + n + 1)
    
    N = (i, slice(xlow, xupper, 1), slice(ylow, yupper, 1))
    
    return N

# main
if __name__ == "__main__":


    with open(input_file) as f:
        input = dict([line.split() for line in f if (len(line) > 1) & (line[0] != '#') ])
    
    loc_clean_data = input['loc_clean_data']
    loc_model2_data = input['loc_model2_data']
    
    with open(loc_clean_data + 'clean_dates.pickle', 'rb') as f:
        dates = pickle.load(f)
    
    dx = int(input['dx']) # pixels
    dy = int(input['dy']) # pixels
        
    start_date = datetime.strptime(input['start_date'], '%d-%m-%Y')
    end_date = datetime.strptime(input['end_date'], '%d-%m-%Y')
    
    start_time = datetime.strptime(input['start_time'], '%H:%M').strftime('%H%M%S')
    end_time = datetime.strptime(input['end_time'], '%H:%M').strftime('%H%M%S')

# =============================================================================
#     for each day compute df and save
# =============================================================================
   
    for single_date in md.daterange(start_date, end_date):
       
# =============================================================================
#     select days from clean data
# ============================================================================+
       
        print(single_date, end = '\t')
        date = dates.date
        idx = (single_date < date)  & (date < single_date + timedelta(1)) 
        dates_day = dates[idx].sort_values(by = ['date'])
        
        print(len(dates_day))
        
        if len(dates_day) == 0:
            continue
        
        # combine the files of one day in one dataset
        dss = []
        for file in dates_day.file_name:
            ds = xr.open_dataset(file)
            dss.append(ds)
        
        ds = xr.concat(dss, 't')
        print(ds.t.data)
        # invalid =np.sum(ds.ct == 0, axis = (1,2))
        # ds = ds.where(invalid < 10000, dr/op = True)

        # determine starting pixels      
        if single_date == start_date:
            i_indx = np.arange(int(dx/2), ds.dims['x'], dx)
            j_indx = np.arange(int(dy/2), ds.dims['y'], dy)
            
            i, j = [grid.flatten() for grid in np.meshgrid(i_indx, j_indx)]
            i_start, j_start =  [grid.flatten() for grid in np.meshgrid(i_indx, j_indx)]
            
            # check if pixel is in the domain
            pixels = []
            var = ['u', 'v']
            for p in range(len(i)):
                check = np.sum([ds.isel(x = i[p], y= j[p], t = 0)[v] for v in var])
                if np.isnan(check):
            #         print(ds.isel(x = i[p], y= j[p], t = 0))
                    continue
                else:
                    pixels.append(p)

            i, j = i[pixels], j[pixels]
            i_start, j_start = np.copy(i),np.copy(j)
            n_p = len(i)
        else:
            i, j = np.copy(i_start), np.copy(j_start)
                    
        # make n_t x n_p - matrix (n_t number of timesteps, n_p number of pixels) with x-
        # and y-  coordinates and indexes.        
        n_t = ds.t.size
        xloc = np.empty((n_t, n_p))
        yloc = np.empty((n_t, n_p))
        xloc[:] = np.nan
        yloc[:] = np.nan
        xloc[0, :] = i
        yloc[0, :] = j
        
# =============================================================================
#         Pixel path
# =============================================================================
        
        for p in range(n_p):
            for k in range(0, n_t-1):
        #         print(k ,i[p], j[p], p , '           ', end = '\r')
                try:
                    i[p], j[p] = md.nextLoc(ds, k, i[p], j[p])
                except ValueError:
                    break
                xloc[k+1, p] = i[p]
                yloc[k+1, p] = j[p]
            # print(f'\rpixel {p + 1:3.0f} of {n_p}     ', flush = True, end = '')


# =============================================================================
#         Values for pixel path
# =============================================================================
        
        # add variable with cloud/clear sky/not defined
        ds['z'] = (['t', 'x', 'y'], np.select(
                condlist=[ds.ct > 1, ds.ct == 1], 
                choicelist=[0, 1], 
                default=np.nan))

        # clear clouds with insuficient data
        ds.cod.data = ds.cod.where(~((ds.z == 0) & (np.isnan(ds.cth))))
        ds.z.data = ds.z.where(~((ds.z == 0) & (np.isnan(ds.cth))))

        ds.cth.data = ds.cth.where(~((ds.z == 0) & (np.isnan(ds.cod))))
        ds.z.data = ds.z.where(~((ds.z == 0) & (np.isnan(ds.cod))))
        
        
        # clear data of clear sky or non classified
        ds.cth.data = ds.cth.where(ds.z == 0)
        ds.cod.data = ds.cod.where(ds.z == 0)
        
        ds.cod.data = np.log(ds.cod.where(ds.cod != 0)) #-> should add this in the data processing ?
        
        # Create matrix with corresponding values

        n_nanrows = 1 # number of nan rows between different pixel paths. 
        X = np.zeros(((n_t + n_nanrows) * n_p , 7))
        
        for p in range(n_p):
            for i in range(n_t + n_nanrows ):
        #     print(i, xloc[i,p], yloc[i,p])
        
                if i < n_t:
        #             X[(ds.t.size + hoi) * p + i, 2] = p
                    if np.isnan(xloc[i, p]):
                        X[(n_t + n_nanrows) * p + i, 6] = -10
                    else:
                        
                        x = int(xloc[i, p])
                        y = int(yloc[i, p])                        
                        
                        N = neighborhood(ds, i, x, y)
                        
                        
                        X[(n_t + n_nanrows) * p + i, 0] = ds.cth[i, x, y ]
                        X[(n_t + n_nanrows) * p + i, 1] = ds.cod[i, x, y ]
                        X[(n_t + n_nanrows) * p + i, 2] = ds.z[i, x, y ]
                        X[(n_t + n_nanrows) * p + i, 3] = ds.z[N].mean()
                        X[(n_t + n_nanrows) * p + i, 4] = ds.cth[N].mean()
                        X[(n_t + n_nanrows) * p + i, 5] = ds.cod[N].mean()
                        X[(n_t + n_nanrows) * p + i, 6] = ds.ct[i, x, y ]
                else: 
                    X[(n_t + n_nanrows) * p + i, :] = np.nan # add a nan row to seperate different days/pixels
        
        df = pd.DataFrame(X,
                           columns=['h_t', 'd_t', 'z_t', 'csf_t', 'h_bar_t', 'd_bar_t', 'ct']
                         )
                
        df = df.drop(df[df.ct == -10].index) # drop rows for pixels that went out of the frame

        # combine current state and next state in one row
        s_t = df.iloc[:-1].reset_index(drop = True)
        s_t1 = df.iloc[1:].reset_index(drop = True).add_suffix('_next')
        df = pd.concat([s_t, s_t1], axis = 1)

# =============================================================================
#         Save model 1 data
# =============================================================================

        md.saveModelData(df, loc_model2_data, single_date)
