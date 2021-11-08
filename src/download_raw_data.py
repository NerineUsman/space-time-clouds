#!/usr/bin/env python3
# coding: utf-8

"""
Created on Thu Nov  4 11:35:33 2021

@author: Nerine
"""

# import modules
import GOES as GOES
import sys, os
import pickle
import xarray as xr
import numpy as np
from datetime import datetime, timedelta

# variables
input_file = 'input_download.txt'

# functions
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def acquisitionDates(flist):
    acq_dates = []
    for idx, file in enumerate(flist):
        ds = xr.open_dataset(file)
        acq_dates.append(ds.t.values)
    return np.array(acq_dates)

# main #
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f])
    
    loc_data = input['loc_data']
    
    start_date = datetime.strptime(input['start_date'], '%d-%m-%Y')
    end_date = datetime.strptime(input['end_date'], '%d-%m-%Y')
    
    start_time = datetime.strptime(input['start_time'], '%H:%M').strftime('%H%M%S')
    end_time = datetime.strptime(input['end_time'], '%H:%M').strftime('%H%M%S')    

    for single_date in daterange(start_date, end_date):
        print(single_date.strftime('%Y-%m-%d'))
        DateTimeIni = single_date.strftime("%Y%m%d") + "-" + start_time
        DateTimeFin = single_date.strftime("%Y%m%d") + "-" + end_time
        
        # download the data
        # Cloud top height
        GOES.download('goes16', 'ABI-L2-ACHAF', # see https://www.noaa.gov/organization/information-technology/list-of-big-data-program-datasets for product names
                              DateTimeIni = DateTimeIni, DateTimeFin = DateTimeFin, 
                              path_out= loc_data + 'cth/', show_download_progress=False)
        # Cloud optical depth
        GOES.download('goes16', 'ABI-L2-CODF', # see https://www.noaa.gov/organization/information-technology/list-of-big-data-program-datasets for product names
                              DateTimeIni = DateTimeIni, DateTimeFin = DateTimeFin, 
                              path_out= loc_data + 'cod/', show_download_progress=False)
        # Derived Motions
        GOES.download('goes16', 'ABI-L2-DMWF', # see https://www.noaa.gov/organization/information-technology/list-of-big-data-program-datasets for product names
                              DateTimeIni = DateTimeIni, DateTimeFin = DateTimeFin, 
                              path_out= loc_data + 'dm/', show_download_progress=False)
    
    ## create dictionairy with all available dates
    dates = {}
    
    path = loc_data + 'cth/'
    files = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and 'ACHAF' in f)]
    acq_dates = acquisitionDates(files)
    dates['cth'] = acq_dates
    
    path = loc_data + 'cod/'
    files = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and 'CODF' in f)]
    acq_dates = acquisitionDates(files)
    dates['cod'] = acq_dates
    
    path = loc_data + 'dm/'
    files = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and 'DMWF' in f)]
    acq_dates = []
    
    for idx, file in enumerate(files):
        ds = xr.open_dataset(file)
        t = datetime.strptime(ds.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')
        t = np.datetime64(t)
        acq_dates.append(t)
    dates['dm'] = np.array(acq_dates)

    with open(loc_data + 'dates.txt', 'wb') as outfile:
        pickle.dump(dates, outfile)
    
    # with open(loc_data + 'dates.txt', 'rb') as f:
    #     x = pickle.load(f)
    
