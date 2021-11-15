#!/usr/bin/env python3
# coding: utf-8

"""
Created on Thu Nov  4 11:35:33 2021

@author: Nerine
"""

# import modules
import sys, os
import pickle
import GOES as GOES
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, '../lib')
import data_clean as dc


# variables
input_file = 'input_download.txt'

# functions
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

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
        print('download cth', end = '\r')
        GOES.download('goes16', 'ABI-L2-ACHAF', # see https://www.noaa.gov/organization/information-technology/list-of-big-data-program-datasets for product names
                              DateTimeIni = DateTimeIni, DateTimeFin = DateTimeFin, 
                              path_out= loc_data + 'cth/', show_download_progress=True)
        # Cloud optical depth
        print('download cod', end = '\r')
        GOES.download('goes16', 'ABI-L2-CODF', # see https://www.noaa.gov/organization/information-technology/list-of-big-data-program-datasets for product names
                              DateTimeIni = DateTimeIni, DateTimeFin = DateTimeFin, 
                              path_out= loc_data + 'cod/', show_download_progress=False)
        # Derived Motions
        print('download dm', end = '\r')
        GOES.download('goes16', 'ABI-L2-DMWF', # see https://www.noaa.gov/organization/information-technology/list-of-big-data-program-datasets for product names
                              DateTimeIni = DateTimeIni, DateTimeFin = DateTimeFin, 
                              path_out= loc_data + 'dm/', show_download_progress=False)
    
    ## create dictionairy with all available dates
    dates = {}
    
    path = loc_data + 'cth/'
    files = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and 'ACHAF' in f)]
    acq_dates = dc.acquisitionDates(files)
    dates['cth'] = pd.DataFrame({'file_name': files, 'date': acq_dates } )
    
    path = loc_data + 'cod/'
    files = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and 'CODF' in f)]
    acq_dates = dc.acquisitionDates(files)
    dates['cod'] =  pd.DataFrame({'file_name': files, 'date': acq_dates } )
    
    path = loc_data + 'dm/'
    files = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and 'DMWF' in f)]
    acq_dates = []
    band_id = []
    
    for idx, file in enumerate(files):
        ds = xr.open_dataset(file)
        t = datetime.strptime(ds.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')
        t = np.datetime64(t)
        band_id.append(ds.band_id.values[0])
        acq_dates.append(t)
    dates['dm'] =  pd.DataFrame({'file_name': files, 'date': acq_dates , 'band_id': band_id} )

    with open(loc_data + 'dates.txt', 'wb') as outfile:
        pickle.dump(dates, outfile)
    
    # with open(loc_data + 'dates.txt', 'rb') as f:
    #     x = pickle.load(f)
    
