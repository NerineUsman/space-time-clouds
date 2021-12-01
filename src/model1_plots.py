#!/usr/bin/env python3
# coding: utf-8

"""
Created on Mon Nov 29 11:05:38 2021

@author: Nerine
"""

# import modules
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


# variables
input_file = 'input_model1_local.txt'
# input_file = './space-time-clouds/src/input_model1.txt'

hlim = [0, 16] #km
dlim = [-1.5, 5.1] #log (d)

# functions
def plot_distribution_next_cloud(df, title = None, nbins = 50, **kwargs):
    fig, ax = plt.subplots(1,3,figsize = (20, 4))
    # cth
    ax[0].hist(df.h_t_next * 1e-3, bins = nbins, **kwargs)
    ax[0].set_title('CTH (km)')
    ax[0].set_xlim(hlim)
    # cod
    ax[1].hist(df.d_t_next, bins = nbins, **kwargs)
    ax[1].set_title('COD (log($\cdot$)')
    ax[1].set_xlim(dlim)
    
    # joint density
    bins = [nbins, nbins]
    h = ax[2].hist2d(df.d_t_next, df.h_t_next*1e-3, bins=bins, 
                     cmap=plt.cm.Blues, norm=mpl.colors.LogNorm(),
                     **kwargs)
    ax[2].set_xlabel('COD (log($\cdot$)')
    ax[2].set_ylabel('CTH (km)')
    ax[2].set_xlim(dlim)
    ax[2].set_ylim(hlim)
    plt.colorbar(h[3], ax=ax[2])

    # ax[2].set_colorbar()
    
    if title != None:
        fig.suptitle(title)
    return fig, ax



# main
if __name__ == "__main__":
    with open(input_file) as f:
        input = dict([line.split() for line in f])
    
    loc_model1_data = input['loc_model1_data']
    loc_fig = input['loc_fig']
    
    # combine df's from all days in model 1 data
    files = [loc_model1_data + f for f in os.listdir(loc_model1_data) if (os.path.isfile(os.path.join(loc_model1_data, f)))]
    files
    
    dfs =[]
    for file in files:
        df = pd.read_csv(file)   
        print(len(df))
        dfs.append(df)
    df = pd.concat(dfs)
# =============================================================================
#   Clouds
# =============================================================================

# =============================================================================
#   Cloud to cloud    
# =============================================================================
    df_cc = df.loc[(df.cloud == 'cloud') & (df.cloud_next == 'cloud') ]
    df_cc.insert(len(df_cc.columns), 'dh', df_cc.apply(lambda row: row.h_t_next - row.h_t, axis = 1))
    df_cc.insert(len(df_cc.columns), 'dd', df_cc.apply(lambda row: row.d_t_next - row.d_t, axis = 1))
    df_cc.head()
        
# =============================================================================
#   Clear sky to cloud
# =============================================================================

    df_sc = df.loc[(df.cloud == 'clear sky') & (df.cloud_next == 'cloud') ]

# =============================================================================
#   To clear sky
# =============================================================================
    df_s = df.loc[((df.cloud == 'clear sky') | (df.cloud == 'cloud')) & (df.cloud_next == 'clear sky')  ]

    
# =============================================================================
#   plots
# =============================================================================

    # joint density
    title = f'Cloud distributions, n = {len(df_cc) + len(df_sc)}'
    fig, ax = plot_distribution_next_cloud(pd.concat([df_cc, df_sc]), title = title, density = True )
    fig.savefig(loc_fig + 'cloud_distr.png')

    
    # distribution for clouds after clear sky
    title = f'Clear sky to cloud distributions, n = {len(df_sc)}'
    fig, ax = plot_distribution_next_cloud(df_sc, title = title, density = True )
    fig.savefig(loc_fig + 'clear_sky_to_cloud_distr.png')

    
    # cloud to cloud overview
    
    fig, ax = plt.subplots(2,3,figsize = (15, 7))
    # cth
    ax[0,0].hist(df_cc.h_t * 1e-3, bins = 50, density = True)
    ax[0,0].set_title('CTH (km)')
    ax[0,1].hist(df_cc.dh * 1e-3, bins = 200, density = True)
    ax[0,1].set_title('$\Delta$CTH (km)')
    ax[0,2].hist(df_cc.dh * 1e-3, bins = 200, density = True)
    ax[0,2].set_title('$\Delta$CTH (km) zoomed')
    ax[0,2].set(xlim = [-2, 2])
    ax[1,0].hist(df_cc.d_t, bins = 50, density = True)
    ax[1,0].set_title('COD ($\cdot$)')
    ax[1,1].hist(df_cc.dd, bins = 100, density = True)
    ax[1,1].set_title('$\Delta$COD ($\cdot$)')
    ax[1,2].hist(df_cc.dd, bins = 100, density = True)
    ax[1,2].set_title('$\Delta$COD zoomed')
    ax[1,2].set(xlim = [-2, 2])
    fig.suptitle('Cloud to Cloud')
    fig.savefig(loc_fig + 'cloud_to_cloud_overview.png')


    
    # cloud to cloud distribution from a few bins
    
    mu_h = [1e3, 6e3, 9e3, 12e3] #
    mu_d = [0, 1, 2, 3]
    mu_h, mu_d = np.meshgrid(mu_h, mu_d)
    mu_h, mu_d = [x.flatten() for x in [mu_h, mu_d]]
    
    delta_h = 300 # m
    delta_d = .1
    
    bins = [[[h - delta_h, h + delta_h], [d - delta_d, d + delta_d]] for h,d in zip(mu_h, mu_d)]
    
    i = 0
    for b in bins:       
        print(b)
        # filter cloud to cloud on from within bin
        df_temp = df_cc.loc[(df_cc.h_t > b[0][0]) & (df_cc.h_t < b[0][1])
                            & (df_cc.d_t > b[1][0]) & (df_cc.d_t < b[1][1])]
        if len(df_temp) <= 1:
            continue
        title = f'Bin centre (h, d) = ({mu_h[i]*1e-3} km, {mu_d[i]}), n = {len(df_temp)}'
        fig, ax = plot_distribution_next_cloud(df_temp, title = title, density = True)
        ax[0].axvline(mu_h[i]*1e-3, color = 'r', label = 'bin center')
        ax[0].legend()
        ax[1].axvline(mu_d[i], color = 'r', label = 'bin center')
        ax[1].legend()
        ax[2].plot(mu_d[i], mu_h[i]*1e-3,'ro', label = 'bin center')
        ax[2].legend()
        fig.savefig(f'{loc_fig}cloud_to_cloud_(h_d)_({mu_h[i]*1e-3}_{mu_d[i]}).png')
        plt.show()
        i += 1
        
        
        
        
        
        
        
        
        
        
