#!/usr/bin/env python3
# coding: utf-8

"""
Created on Mon Nov 29 11:05:38 2021

@author: Nerine
"""

# import modules
import os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm

sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib')
import ml_estimation as ml




# variables
# input_file = 'input_model1_local.txt'
input_file = './space-time-clouds/src/input_model1.txt'

hlim = [0, 16] #km
dlim = [-1.5, 5.1] #log (d)

# functions
def state_bins(mu_h, mu_d, delta_h = 300, delta_d =.1):
    """
    Parameters
    ----------
    mu_h : list
        cth coordinates. in km
    mu_d : list
        cod coordinates. log(cod)
    delta_h : float, optional
        half the binwidth in cth. The default is 300.
    delta_d : TYPE, optional
        half the binwidth for cod. The default is .1.

    Returns
    -------
    bins : list
        list of bins ([h_min, hmax],[dmin, dmax]).

    """
    mu_h, mu_d = np.meshgrid(mu_h, mu_d)
    mu_h, mu_d = [x.flatten() for x in [mu_h, mu_d]]

    bins = [[[h - delta_h, h + delta_h], [d - delta_d, d + delta_d]] for h,d in zip(mu_h, mu_d)]
    return bins


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
                     cmap=plt.cm.Blues,
                       norm=mpl.colors.LogNorm(),
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
    

    bins = state_bins(mu_h, mu_d)
    
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
        
    # cod estimators normal distribution
    mu_h = np.arange(1e3, 14e3, 2e3) # m
    mu_d = np.arange(0, 4, .1)
    n_h = len(mu_h)
    
    bins = state_bins(mu_h, mu_d)
    mu_hat = np.zeros((len(mu_h) * len(mu_d)))
    sigma_hat = np.zeros((len(mu_h) * len(mu_d)))
    n_clouds = np.zeros((len(mu_h) *len(mu_d)))
    i = 0
    for b in bins:
        # filter cloud to cloud on from within bin
        df_temp = df_cc.loc[(df_cc.h_t > b[0][0]) & (df_cc.h_t < b[0][1])
                            & (df_cc.d_t > b[1][0]) & (df_cc.d_t < b[1][1])] 
        n = len(df_temp)
        n_clouds[i] = n

        if n <= 1:
            continue
        mu_hat[i] = df_temp.d_t_next.mean()
        sigma_hat[i] = np.sqrt(n / (n-1) * df_temp.d_t_next.var())
        i += 1
    
    mu_hat = mu_hat.reshape((len(mu_d), len(mu_h)))
    sigma_hat = sigma_hat.reshape((len(mu_d), len(mu_h)))
    n_clouds = n_clouds.reshape((len(mu_d), len(mu_h)))
    
    
    ## ml estimation
    df_cc['constant'] = 1
    df_cc['hd'] = df_cc.h_t * df_cc.d_t
    sm_ml_manual = ml.MyDepNormML(df_cc.d_t_next,df_cc[['constant','h_t', 'd_t', 'hd']]).fit(
                        start_params = [1, .001, 0.9, 0, .7, .001])
    print(sm_ml_manual.summary())
    beta = sm_ml_manual.params[:4]
    gamma = sm_ml_manual.params[-2:]
    
    h_labels = [f'cth = {h * 1e-3} km' for h in mu_h]
    
    color= cm.Blues(np.linspace(.2,1, len(mu_h)))
    color_ml= cm.Greens(np.linspace(.2,1, len(mu_h)))
    
    
# =============================================================================
#     Estimators for mu and sigma
# =============================================================================
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    ax[0].plot(mu_d, mu_d, label = 'bin center', c = 'r')
    
    for i, c, c_ml, label in zip(range(n_h), color, color_ml, h_labels):
        ax[0].plot(mu_d, mu_hat[:,i], label = label, c = c,
                   # marker = '.', ls = '--'
                   )
        # ax[0].plot(mu_d, )
        ax[1].plot(mu_d, sigma_hat[:,i], label = label, c = c,
                   # marker = '.', ls = '--'
                   )

    ax[0].legend()
    ax[0].set(xlabel = 'Current state COD (log($\cdot$)',
              ylabel = '$\hat{\mu_d}$',
              title = 'Mean')
    
    ax[1].legend()
    ax[1].set(xlabel = 'Current state COD (log($\cdot$)',
              ylabel = '$\hat{\sigma_d}$',
              title = 'Variance')
    fig.suptitle('Estimators of time distribution COD')
    fig.savefig(loc_fig + 'estimator_COD_local.png')
    
    
# =============================================================================
#     Maximum likelihood
# =============================================================================
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    ax[0].plot(mu_d, mu_d, label = 'bin center', c = 'r')
    
    for i, c, c_ml, label in zip(range(n_h), color, color_ml, h_labels):
        if i % 3 == 0:
            ax[0].plot(mu_d, mu_hat[:,i], label = label, c = c,
                       # marker = '.', ls = '--'
                       )
            # ax[0].plot(mu_d, )
            ax[1].plot(mu_d, sigma_hat[:,i], label = label, c = c,
                       # marker = '.', ls = '--'
                       )
            mu_ml, sigma_ml = np.array([ml.model1(mu_h[i], d, beta, gamma) for d in mu_d]).T
            ax[0].plot(mu_d, mu_ml, label = label + ' ml estimator',
                       c = c_ml)
            ax[1].plot(mu_d, sigma_ml, label = label + ' ml estimator', c = c_ml)

    


    ax[0].legend()
    ax[0].set(xlabel = 'Current state COD (log($\cdot$)',
              ylabel = '$\hat{\mu_d}$',
              title = 'Mean')
    
    ax[1].legend()
    ax[1].set(xlabel = 'Current state COD (log($\cdot$)',
              ylabel = '$\hat{\sigma_d}$',
              title = 'Variance')
    fig.suptitle('Estimators of time distribution COD')
    fig.savefig(loc_fig + 'estimator_COD_ML.png')

    
# =============================================================================
#     Number of points per bin 
# =============================================================================
    fig, ax = plt.subplots(1,1, figsize = (15, 5))
    for i, c, label in zip(range(n_h), color, h_labels):
        ax.plot(mu_d, n_clouds[:,i], label = label, c = c)

    ax.legend()
    ax.set(xlabel = 'Current state COD (log($\cdot$)',
              ylabel = 'N',
              title = 'number of data points per bin')
    
    fig.savefig(loc_fig + 'n_estimator.png')    
    
    
        

        
        
        
        
        
