#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np
import GOES as GOES


# In[2]:


def classifyISCCP(cod, cth, dqf_cod, bound = [3.6, 23, 2e3, 8e3]):
    """
    Function to classify pixels based on cod and cth. 
    cod, cth, dqf_cod should all be of the same shape. 
    input: cod : (mxn) - array.  cloud optical depth
           cth : (mxn) - array.  cloud top height
           dqf_cod: (mxn) - array. data quality flags for cod data. 6 for clear sky, 0 for good quality. 
    
    bound = [lower split value cod, upper split value cod, 
                lower split value cth, upper split value cth ]
    output: ct : (mxn) - array. contains cloud classes indicated by
                0 - invalid pixel
                1 - clear sky
                2 - cumulus
                3 - altocumulus
                4 - cirrus
                5 - stratocumulus
                6 - altostratus
                7 - cirrostratus
                8 - stratus
                9 - nimbostratus
                10 - deep convection
    """
    b_cod = bound[:2]
    b_cth = bound[2:]
    ct = np.where(dqf_cod == 6, 1, # clear sky # should we also use dqf_cth = 4 here?
              np.where(dqf_cod != 0, 0, # non valid data
                       np.where(cod < b_cod[0] , 
                                np.where(cth<b_cth[0], 2,
                                        np.where(cth< b_cth[1], 3, 4)),
                       np.where(cod< b_cod[1] , 
                                np.where(cth<b_cth[0], 5,
                                        np.where(cth< b_cth[1], 6, 7)
                                        ), 
                                np.where(cth<b_cth[0], 8,
                                        np.where(cth< b_cth[1], 9, 10)
                                        )
                               )
                               )
                      )
             )
    return ct

def histClassifications(ct, ax = None):
    if ax is None:
        ax = plt.gca()
    bins = np.linspace(0.5, 10.5, 11)
    return ax.hist(ct.flatten(), bins = bins, density = True)

def makeXArrayFromNetCDFs(file_cod, file_cth,
                          domain = [-50.0,-30.0,-5.0,15.0] # [left, right, bottom, top]
                         ):
    """
    function which makes from the two seperate files for cod and cth one xarray which containts cod, cth and the cloud classification
    input: file_cod    NETCDF from NOAA which contains the COD data
           file_cth    NETCDF from NOAA which contanis the CTH data
           [domain]    array (4,) which contains the extent of the area of interest
                                [left, right, bottom, top]
    output: xarray     xarray : variables - cth, cod, ct (cloud types)
                                coordinates - lat, lon, x, y
                                attribute - timestamp
    """ 
    # reads the file using xarray
    ds_cod = xr.open_dataset(file_cod)
    ds_cth_coarse = xr.open_dataset(file_cth)

    # reads the file using GOES
    ds_cod_goes = GOES.open_dataset(file_cod)
    ds_cth_goes = GOES.open_dataset(file_cth)
    
    # TODO: check same time step
#     if ds_cod.t != ds_cth.t:
#         return
    

    # get image with the coordinates of corners of their pixels
    cod, LonCor, LatCor = ds_cod_goes.image('COD', lonlat='corner', domain=domain)
    dqf_cod, LonCor, LatCor = ds_cod_goes.image('DQF', lonlat='corner', domain=domain)

    #put cod data in xarray
    xr_cod = xr.Dataset(
        data_vars=dict(
            cod=(["x", "y"], cod.data), 
            dqf_cod = (["x", "y"], dqf_cod.data)
        ),
        coords=dict(
            lon=(["x", "y"], LonCor.data[:-1, :-1]),
            lat=(["x", "y"], LatCor.data[:-1, :-1]),
            x = (["x"], np.linspace(0, 1, cod.data.shape[0])),
            y = (["y"], np.linspace(0, 1, cod.data.shape[1])),
            time=ds_cod.t.data,
        ),
        attrs = dict(extent = domain)
    )

    cth, LonCor_cth, LatCor_cth = ds_cth_goes.image('HT', lonlat='corner', domain=domain)
    # dqf_cth, LonCor, LatCor = ds_cth.image('DQF', lonlat='corner', domain=domain)
    xr_cth = xr.Dataset(
        data_vars=dict(
            cth=(["x", "y"], cth.data)
        ),
        coords=dict(
            lon=(["x", "y"], LonCor_cth.data[:-1, :-1]),
            lat=(["x", "y"], LatCor_cth.data[:-1, :-1]),
            x = (["x"], np.linspace(0, 1, cth.data.shape[0])),
            y = (["y"], np.linspace(0, 1, cth.data.shape[1])),
            time=ds_cod.t.data,
        ),
        attrs=dict(description="cth"),
    )

    # Interpolate cth such that the values are evalauted at the same locations as for cod
    image = xr_cth.interp(x=xr_cod.x, y=xr_cod.y, method = 'nearest') # deal with nans, xr.interp gives to many nan values due to interp. 
                                                                      # nearest improves some but is still quite smoothing...
    # add cod to the same xarray
    image = xr_cod.assign(cth = (["x", "y"], image.cth.data))
    
    return image


def plotImage(image):
    domain = image.extent
    fig, ax = plt.subplots(1,3, figsize = (15, 4), sharex = True, sharey = True)
    fig_cod = ax[0].pcolormesh(image.lon.data, image.lat.data, image.cod)
    ax[0].set_xlim([domain[0], domain[1]])
    ax[0].set_ylim([ domain[2], domain[3]])
    ax[0].set_title('COD')
    plt.colorbar(fig_cod, ax = ax[0])

    fig_cth = ax[1].pcolormesh(image.lon.data, image.lat.data, image.cth)
    ax[1].set_title('CTH')
    plt.colorbar(fig_cth, ax =ax[1])

    fig_ct = ax[2].pcolormesh(image.lon.data, image.lat.data, image.ct)
    ax[2].set_title('Classification')
    fig.colorbar(fig_ct, ax =ax[2])
    return fig

def saveImage(image, output_loc = 'output'):
    date = image.time.data
    ts = pd.to_datetime(str(date)) 
    d = ts.strftime('%Y%m%d-%Hh%M%S') # maybe change day/month to julian days
    image.drop(['lon', 'lat']).to_netcdf(f'{output_loc}/image_{d}.nc')
    return
   

def rawDatatoClassification(flist_cth, flist_cod, **kwargs):
    for i in range(len(flist_cth)):
        file_cth = flist_cth[i]
        file_cod = flist_cod[i]
        image = makeXArrayFromNetCDFs(file_cod, file_cth)
        ct = classifyISCCP(image.cod, image.cth, image.dqf_cod)
        image = image.assign(ct = (["x", "y"], ct))
    #     print(image.time.data)
    #     plotImage(image)
    #     plt.show()
        saveImage(image, **kwargs)
    return image

