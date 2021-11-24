#!/usr/bin/env python3
# coding: utf-8

"""
Created on Mon Nov  8 15:46:40 2021

@author: Nerine
"""

#import modules
import sys, os
import pickle
import numpy as np
import scipy as sc
import xarray as xr
import pandas as pd
from datetime import datetime
import time as tm

sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib')
import data_clean as dc

# variables
input_file = 'input_cleandata_local.txt'
# input_file = './space-time-clouds/src/input_cleandata.txt'
centerpoint = [5, -40] # deg lat, deg lon
scale_lat = 111.32e3 #m
scale_lon = 40075e3 * np.cos( centerpoint[0] * np.pi/ 180 ) / 360 #m
print_process_time = False

   
# functions
def datetime64_to_time_of_day(datetime64_array):
    """
    Return a new array. For every element in datetime64_array return the time of day (since midnight).
    >>> datetime64_to_time_of_day(np.array(['2012-01-02T01:01:01.001Z'],dtype='datetime64[ms]'))
    array([3661001], dtype='timedelta64[ms]')
    >>> datetime64_to_time_of_day(np.datetime64('2012-01-02T01:01:01.001Z','[ms]'))
    numpy.timedelta64(3661001,'ms')
    """
    day = datetime64_array.astype('datetime64[D]').astype(datetime64_array.dtype)
    time_of_day = datetime64_array - day
    return time_of_day

def latlon_to_meters(lat, lon, 
                     centerpoint = centerpoint
                    ):

    return (lon - centerpoint[1]) * scale_lon , (lat - centerpoint[0]) * scale_lat

def meters_to_latlon(x, y, 
                     centerpoint = centerpoint
                    ):
    return (y / scale_lat) + centerpoint[0], (x / scale_lon) + centerpoint[1]

def saveDs(ds, output_loc):
    date = ds.t.data
    ts = pd.to_datetime(str(date)) 
    d = ts.strftime('%Y%m%d-%Hh%M%S') # maybe change day/month to julian days
    file_name = f'{output_loc}/image_{d}.nc'
    ds.to_netcdf(file_name)
    return file_name
     
def makeCleanDatesFile(path):
    files = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and 'image' in f)]
    acq_dates = dc.acquisitionDates(files)
    dates = pd.DataFrame({'file_name': files, 'date': acq_dates } )
    
    with open(path + 'clean_dates.pickle', 'wb') as outfile:
        pickle.dump(dates, outfile)

# main #
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f])
    
    loc_data = input['loc_data']
    loc_clean_data = input['loc_clean_data']
    
    with open(loc_data + 'dates.txt', "rb") as f:
        dates = pickle.load(f)
    
# =============================================================================
#     determine for which dates we need to do the cleaning
# =============================================================================
    
    # skip files which are not in the correct time range
    start_date = datetime.strptime(input['start_date'], '%d-%m-%Y')
    end_date = datetime.strptime(input['end_date'], '%d-%m-%Y')
    start_time = pd.to_timedelta(input['start_time'] + ':00')
    end_time = pd.to_timedelta(input['end_time'] + ':00')
    
    date = dates['cth'].date 
    
    # check whether t  is within start and end date and time
    time = datetime64_to_time_of_day(date) 
    idx = (start_date < date)  & (date < end_date) & (start_time < time)  & (time < end_time)
    dates['cth'] = dates['cth'][idx]
    
    print(dates['cth'])
    
    makeCleanDatesFile(loc_clean_data)

    # skip files which are already cleaned
    with open(loc_clean_data + 'clean_dates.pickle', 'rb') as f:
        clean_dates = pickle.load(f)
    

    idx = [idx for idx, date in dates['cth'].iterrows() if ~(date.date == clean_dates.date).any()]
    dates['cth'] = dates['cth'].loc[idx]

    print(dates['cth'].date)    
    
    

    for idx_cth, date in dates['cth'].iterrows(): # loop over all available files for cth
        
        # time the loop for each file
        if print_process_time:
            start_time = tm.time()
# =============================================================================
#       Find the corresponding files from cod and dm    
# =============================================================================

        t = dates['cth'].loc[idx_cth].date
        
        # cod
        dt = abs(dates['cod'].date - t)
        idx_cod = dt.argmin()
        if dt.loc[idx_cod] > np.timedelta64(10, 'm'):
            print(f'!Error: No cod found within 10 minutes of {t}')
            continue
            # TODO: download the right file or skip
        
        # dm
        dt = abs(dates['dm'].date - t)
        idx_dm = dt.nsmallest(6).index
        if dt.loc[idx_dm[0]] > np.timedelta64(1, 'h'):
            print(f'!Error: No cod found within 1 hour of {t}')
            continue
        
        print(t)
        file_cth = dates['cth'].loc[idx_cth].file_name
        file_cod = dates['cod'].loc[idx_cod].file_name
        files_dm = dates['dm'].loc[idx_dm].file_name.values
        
        
# =============================================================================
#       Interpolated cth to cod points and domain to right square
# =============================================================================
        cod_cth = dc.combineCOD_CTH(file_cod, file_cth) 

        # Classify the clouds
        ct = dc.classifyISCCP(cod_cth.cod, cod_cth.cth, cod_cth.dqf_cod)
        da = cod_cth.assign(ct = (["x", "y"], ct))
        da = da.where(da.lat - dc.coastLine(da.lon) > 0)
        
        # projection to meters
        xx, yy = latlon_to_meters(da.lat, da.lon)
        da_m = da.assign_coords({"X" : xx, "Y": yy})
        
# =============================================================================
#        Interpolation to equidistant grid
# =============================================================================

        # create equidistant grid within the boundaries of available data.
        x_min = xx.min(axis = 0).max().values
        x_max = xx.max(axis = 0).min().values
        y_min = yy.min(axis = 1).max().values
        y_max = yy.max(axis = 1).min().values
        x_min, x_max, y_min, y_max
        dx = 5e3 # m
        nx = (x_max - x_min ) / dx * 1j
        ny = (y_max - y_min ) / dx * 1j
        grid_x, grid_y = np.mgrid[x_min:x_max:nx, y_min:y_max:ny] # maybe find extent bit more precise
        
        # points where we have the values of the data
        points = np.empty((da_m.lat.size, 2))
        points[:, 0] = da_m.X.values.flatten()
        points[:, 1] = da_m.Y.values.flatten()
        
        # select a single timestep in which to interpolate the data
        cth_old = da_m.cth.values.flatten()
        cod_old = da_m.cod.values.flatten()
        ct_old = da_m.ct.values.flatten()
        
        cth = sc.interpolate.griddata(points, cth_old, (grid_x, grid_y), method='nearest')
        cod = sc.interpolate.griddata(points, cod_old, (grid_x, grid_y), method='nearest')
        ct = sc.interpolate.griddata(points, ct_old, (grid_x, grid_y), method='nearest')
        
        
        ## make new xarray with new coordinates in meters
            #put cod, cth and ct data in xarray
        d = xr.Dataset(
                data_vars=dict(
                    cth=(["x", "y"], cth),
                    cod=(["x", "y"], cod), 
                    ct=(["x", "y"], ct), 
                ),
                coords=dict(
                    x = (["x"], grid_x[:, 0]),
                    y = (["y"], grid_y[0, :]),
                    t = da_m.time.data,
                ),
                attrs = dict(extent = da_m.extent)
            )
        
        d.cth.attrs["units"] = "m"
        d.x.attrs["units"] = "m"
        d.y.attrs["units"] = "m"
        
        invalid =np.sum(d.ct == 0)
        if invalid > 10000:
            if print_process_time:
                end_time = tm.time()
                elapsed_time = end_time - start_time 
                print(f'File with timestamp: ({t}) processed in {elapsed_time} s')
            continue

# =============================================================================
#         Advection
# =============================================================================
        # Only keep points which are in the right area and do contain a wind speed
        dss = []
        for file in files_dm:
            ds = xr.open_dataset(file)
        
            nMeasures = ds.nMeasures[(ds.lon< -30) &
                                     (ds.lon> -50) &
                                     (ds.lat > -5.0) &
                                     (ds.lat < 15.0 ) &
                                     (ds.lat - dc.coastLine(ds.lon) > 0) &  # Delete over land points
                                     ~(np.isnan(ds.wind_speed))             # Delete points which do not contain a wind speed
        
                                    ]
            ds_small = ds[dict(nMeasures=nMeasures)]
            dss.append(ds_small)
        
        ds_wind = xr.concat(dss, 'nMeasures')
        
        # Determine coordinates in X-Y plane
        xx, yy = latlon_to_meters(ds_wind.lat,ds_wind.lon)
        ds_wind = ds_wind.assign_coords({"X" : xx, "Y": yy})

        # points where we have the values of the data
        points = np.empty((ds_wind.X.size, 2))
        points[:, 0] = ds_wind.X
        points[:, 1] = ds_wind.Y
        
        z = ds_wind.wind_speed * np.exp(1j * ds_wind.wind_direction / 180 * np.pi)

        # interpolate
        wind = sc.interpolate.griddata(points, z, (grid_x, grid_y), method='nearest')
    
        # add wind field to dataset
        ws = np.abs(wind)
        wd = np.angle(wind)
        
        
        
        d = d.assign({"u" : (("x", "y"), np.real(wind)),
                      "v" : (("x", "y"), np.imag(wind))}) 
        
        d.u.attrs["units"] = "m s-1"
        d.v.attrs["units"] = "m s-1"
        
        # delete values above land
        lat, lon = meters_to_latlon(d.x, d.y)
        d = d.where(lat - dc.coastLine(lon) > 0)

# =============================================================================
#         save new dataset
# =============================================================================
        file_name = saveDs(d, loc_clean_data)
        
        if print_process_time:
            end_time = tm.time()
            elapsed_time = end_time - start_time 
            print(f'File with timestamp: ({t}) processed in {elapsed_time} s')
# =============================================================================
#     Update file with cleaned dates
# =============================================================================

    path = loc_clean_data
    makeCleanDatesFile(path)

