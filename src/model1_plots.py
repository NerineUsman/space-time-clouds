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
from cmcrameri import cm as  cmc
import matplotlib.colors as mcolors


from scipy.stats import norm, beta



sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib')
import ml_estimation as ml
import model1_explore as me
import model1 as mod
import Utilities as util




# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model1.txt'
# input_file = './space-time-clouds/src/input_model1.txt'

hlim = [0, 16] #km
dlim = [-1.5, 5.1] #log (d)

# =============================================================================
# Cloud distribution Histograms
# =============================================================================

def hist2d_f(ax, xedges, yedges, freq, **kwargs):
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    h = ax.hist2d(X.flatten(), Y.flatten(), bins = [xedges, yedges], weights = freq.T.flatten(), **kwargs)
    return h

def hist_f(ax, edges, freq, **kwargs):
    hist = ax.hist(edges[:-1], bins = edges, weights = freq, **kwargs)
    return hist

def plotCloudHist_f(dedges, hedges, freq,
                                 title = None, 
                                 mixture = False,
                                 ML = True,
                                 cmap = cmc.batlow,
                                 figsize =  (20, 4),
                                 **kwargs):
    fig, ax = plt.subplots(1,3,figsize = figsize)
    # histograms
    hist_f(ax[0], hedges * 1e-3, freq.sum(axis = 0), color = cmap(0), **kwargs)
    hist_f(ax[1], dedges, freq.sum(axis = 1), color = cmap(0), **kwargs)
    h = hist2d_f(ax[2], dedges, hedges * 1e-3, freq, 
             cmap= cmap,
                       norm=mpl.colors.LogNorm(),
                     **kwargs)
    
    # titles etc.
    ax[0].set_xlabel('CTH (km)')
    dh = hedges[1] - hedges[0]
    max_h = (freq.sum(axis = 0) /  (freq.sum(axis = 0).sum() * dh * 1e-3)).max()
    ax[0].set_ylim([0, max_h + .1])
    ax[0].set_xlim(hlim)
    ax[1].set_xlabel('log COD ($\cdot$)')
    ax[1].set_xlim(dlim)
    
    # joint density
    ax[2].set_xlabel('log COD ($\cdot$)')
    ax[2].set_ylabel('CTH (km)')
    ax[2].set_xlim(dlim)
    ax[2].set_ylim(hlim)
    plt.colorbar(h[3], ax=ax[2])

    # ax[2].set_colorbar()
    
    if title != None:
        fig.suptitle(title)
    return fig, ax


# =============================================================================
#  Plot fit distributions
# =============================================================================

def getDisplayName(theta):
    if 'long_name' in theta.attrs.keys():
        ylabel = theta.attrs['long_name']
    else:
        ylabel = theta.name
    return ylabel
        

def plotCTHBeta(ax, a, b, n = 50, label = 'Beta', color = 'forestgreen'):
    H = np.linspace(0, ml.h_max, n)
    H_norm = ml.CTHtoUnitInt(H)
    h_beta_fit = beta(a, b).pdf(H_norm)

    line = ax.plot(H * 1e-3, h_beta_fit / 15, label = label, color = color )
    ax.legend()
    return line

def plotCTHBetaMix(ax, alpha1, beta1, alpha2, beta2, p, n = 50,
                   mainlabel = None, 
                   maincolor = 'darkorange',
                   plotSubBeta = True, 
                   param = 'standard',
                   **kwargs):
    
    if param == 'standard':
        alpha1
    elif param == 'mean_sum':
        alpha1, beta1 = alpha1 * beta1, beta1 - alpha1 * beta1
        alpha2, beta2 = alpha2 * beta2, beta2 - alpha2 * beta2
    else: 
        print('incorrect parameterization')
    
    if mainlabel == None:
        mainlabel = f'Beta mixture p = {p:.2f}'
        
    H = np.linspace(0, ml.h_max, n)
    H_norm = ml.CTHtoUnitInt(H)
    
    if plotSubBeta:
        ax.plot(H * 1e-3, ml.pdf_b(H_norm, alpha1, beta1) / 15, '--',
                   color = 'sandybrown', 
                   label = '$Beta_1$')
        ax.plot(H * 1e-3, ml.pdf_b(H_norm, alpha2, beta2) / 15, '-.',
                   color = 'sandybrown',
                   label = '$Beta_2$')
        
    ax.plot(H * 1e-3, ml.pdf_bmix(H_norm, alpha1, beta1, alpha2, beta2, p) / 15, 
               color = maincolor,
               label = mainlabel, 
               **kwargs)
    ax.legend()
    return ax
    
def plotCODNormal(ax, mu, sigma, n = 50, label = 'Normal', color = 'darkorange', **kwargs):
    D = np.linspace(-1.5, 5, n)
    d_fit = norm(mu, sigma).pdf(D)
    ax.plot(D, d_fit, label = label, color = color, **kwargs)
    ax.legend()
    return ax

# =============================================================================
# Local parameters
# =============================================================================

def plotLocalParamCth(ax, theta, 
                      logscale = False,
#                       n = 5,
                      cmap = cm.Blues,
                      ind = [2,6,10],#14],
                      **kwargs):
    """
    Function to 
    Parameters
    ----------
    ax : An axis-type object to put the figure.
    theta : xarray.DataArray(data, coords = (mu_h, mu_d))
        local parameters 
    logscale : TYPE, optional
        DESCRIPTION. The default is False.
    #                       n : TYPE, optional
        DESCRIPTION. The default is 5.
    cmap : TYPE, optional
        DESCRIPTION. The default is cm.Blues.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """
        
#     n_d = len(theta.mu_d) # TODO: implement number of lines to be plotted
#     ind = np.round(np.linspace(0, n_d, n, endpoint = False))
#     ind = (ind + np.floor((ind[1] - ind[0])/2)).astype(int)
#     ind = list(ind)
#     print(ind)
    mu_h = theta.mu_h
    mu_d = theta.mu_d[ind]
    color = cmap(np.linspace(.2,1, len(ind)))
    
    theta = theta.where(mu_d == mu_d, drop = True)
    for i, c in zip(range(len(ind)), color):
        ax.scatter(mu_h * 1e-3, theta[:,i],
               label = f'logCOD = {mu_d[i].data:.1f}',
               color = c,
               marker = '*',
               )
    ax.set_xlabel('CTH (km)')

    ax.set_ylabel(getDisplayName(theta))
    if logscale:
        ax.set_yscale('log')
    ax.legend()
    return ax

def plotLocalParamCod(ax, theta, 
                      logscale = False,
#                       n = 5,
                      cmap = cm.Blues,
                      ind = [0, 10], 
                      **kwargs):
    """
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    logscale : TYPE, optional
        DESCRIPTION. The default is False.
    #                       n : TYPE, optional
        DESCRIPTION. The default is 5.
    cmap : TYPE, optional
        DESCRIPTION. The default is cm.Blues.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    mu_h = theta.mu_h[ind]
    mu_d = theta.mu_d
    color = cmap(np.linspace(.2,1, len(ind)))
    
    theta = theta.where(mu_h == mu_h, drop = True)
    for i, c in zip(range(len(ind)), color):
        ax.scatter(mu_d , theta[i,:],
               label = f'CTH = {mu_h[i].data * 1e-3:.1f} km',
               color = c,
               marker = '*',
               **kwargs)
    if logscale:
        ax.set_yscale('log')
    ax.set_xlabel('log COD ($\cdot$)')
    ax.set_ylabel(getDisplayName(theta))
    ax.legend()
    return ax


def plotLocalParam2d(ax, theta,
                     cmap = cmc.batlow,
                     norm = None,
                     logscale = False,
                    **kwargs):
    """
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    cmap : TYPE, optional
        DESCRIPTION. The default is cm.Blues.
    norm : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.
    hist : TYPE
        DESCRIPTION.

    """
    theta.coords["mu_h"] = theta.coords["mu_h"] * 1e-3
    
    if logscale:
        theta = theta.where(theta > 0)
        norm = mpl.colors.LogNorm()
    
    hist = theta.plot(ax = ax, cmap = cmap, norm = norm, **kwargs)
    ax.set_xlabel('log COD ($\cdot$)')
    ax.set_ylabel('CTH (km)')
    return ax, hist
    

def plotLocalParam(theta,
                   logscale = False,
                   cmapsv = cmc.oslo_r, 
                   cod_kwargs = {}, 
                   cth_kwargs = {}, 
                   figsize = (20, 4),
                   **kwargs):
    """
    

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    logscale : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """

    fig, ax = plt.subplots(1,3, figsize = figsize)

    plotLocalParamCth(ax[0], theta, logscale = logscale, cmap = cmapsv, **cth_kwargs)
    plotLocalParamCod(ax[1], theta, logscale = logscale, cmap = cmapsv, **cod_kwargs)
    if logscale:
        plotLocalParam2d(ax[2], theta.where(theta >0), norm = mpl.colors.LogNorm(),
                         **kwargs
                    )
    else:
        plotLocalParam2d(ax[2], theta,
                     **kwargs
                    )
    return fig, ax




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
    
    bincenter = (mu_h, mu_d)
    bins = [[[h - delta_h, h + delta_h], [d - delta_d, d + delta_d]] for h,d in zip(mu_h, mu_d)]
    return bins, bincenter

def plot_distribution_next_cloud(df, 
                                 title = None, 
                                 nbins = 50, 
                                 mixture = False,
                                 ML = True,
                                 hist_kwargs = {}, 
                                 **kwargs):
    # histograms
    freq, dedges, hedges = np.histogram2d(df.d_t_next, df.h_t_next, bins = nbins, **kwargs)
    fig, ax = plotCloudHist_f(dedges, hedges, freq, density = True, **hist_kwargs)
    
    if ML:
        param_norm = me.fitNormal(df.d_t_next)
        # (alpha, beta), conv_b = fitBetaCTH(df_temp.h_t_next)
        param_bm, conv_mb = mod.fitMixBetaCTH(df.h_t_next)
        
        # plotCTHBeta(ax[0], *theta[0:2])
        plotCTHBetaMix(ax[0], *param_bm)
        plotCODNormal(ax[1], *param_norm)
        
    if title != None:
        fig.suptitle(title)
        
    return fig, ax


cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in np.linspace(0, 1, 11)]

# force the first color entry to be grey
cmaplist[0] = (.5, 0.5, .5, .8)
cmaplist[1] = (13/255, 0/255, 181/255, .8)

cmaplist[2] = (215/255, 73/255, 255/255, .8)
cmaplist[3] = (141/255, 96/255, 255/255, .8)
cmaplist[4] = (134/255, 162/255, 255/255, .8)

cmaplist[5] = (229/255, 138/255, 255/255, .8)
cmaplist[6] = (176/255, 146/255, 255/255, .8)
cmaplist[7] = (188/255, 204/255, 255/255, .8)

cmaplist[8] = (240/255, 188/255, 255/255, .8)
cmaplist[9] = (220/255, 206/255, 255/255, .8)
cmaplist[10] = (1, 1, 1, .8)

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist)

# define the bins and normalize
bounds = np.linspace(-.5, 10.5, 12)
cmap_norm = mpl.colors.BoundaryNorm(bounds, 11)

levels = np.arange(11)



def plotCT(image, ax = None,
                  xcoord = 'i', 
                  ycoord = 'j', 
                  cmap = cmap, norm = cmap_norm, 
                  xlim = None,
                  ylim = None,
                  figsize = (14,8), 
                  equal_axis = True, 
                  add_colorbar = True,
                  **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)


    im = image.astype(int).where(
            image >= 0, 0).plot(ax = ax, 
                                x = xcoord, y = ycoord,
                                cmap = cmap,
                                norm = norm,
                                add_colorbar = False,
                                xlim = xlim,
                                ylim  =ylim,
                                 **kwargs )   
    if add_colorbar:
        cbar = fig.colorbar(im, ticks = levels)
        cbar.ax.set_yticklabels(util.ISCCP_classes.values())      
        
    if equal_axis:
        plt.axis('equal')

    return fig, ax

def plotCloudFrac(X, t = None):
    if t.any() is None:
        t = X.t
        
    colors =  list(mcolors.TABLEAU_COLORS.values())
    names = list(util.ISCCP_classes.values())
    
    cf = X.cf.data
    sigma_cf = X.sigma_cf.data
    
    fig = plt.figure(figsize = (8, 5))
    for cloud_type in range(1, 11):
        plt.plot(t, cf[:, cloud_type], label = names[cloud_type], c = colors[cloud_type - 1])
        plt.plot(t, cf[:, cloud_type] + sigma_cf[:, cloud_type], '--', c = colors[cloud_type - 1], alpha = .5)
        plt.plot(t, cf[:, cloud_type] - sigma_cf[:, cloud_type], '--', c = colors[cloud_type - 1], alpha = .5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time (h)')
    plt.ylabel('Cloud fraction')
    plt.tight_layout()
    return fig


# main
if __name__ == "__main__":
    import xarray as xr
    x = xr.open_dataset("../data/simulation/model1/sim_n=19_441x322_cf.nc")
    plotCloudFrac(x)
    
    
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
    df_cc = df_cc.copy()
        
# =============================================================================
#   Clear sky to cloud
# =============================================================================

    df_sc = df.loc[(df.cloud == 'clear sky') & (df.cloud_next == 'cloud') ]
    df_sc = df_sc.copy()
# =============================================================================
#   To clear sky
# =============================================================================
    df_s = df.loc[((df.cloud == 'clear sky') | (df.cloud == 'cloud')) & (df.cloud_next == 'clear sky')  ]

    
# =============================================================================
#   plots
# =============================================================================

    # joint density
    title = f'Cloud distributions, n = {len(df_cc) + len(df_sc)}'
    fig, ax = plot_distribution_next_cloud(pd.concat([df_cc, df_sc]),
                                            title = title, 
                                            mixture = True,
                                            density = True )
    fig.savefig(loc_fig + 'cloud_distr.png')

    
    # distribution for clouds after clear sky
    title = f'Clear sky to cloud distributions, n = {len(df_sc)}'
    fig, ax = plot_distribution_next_cloud(df_sc, title = title,
                                            mixture = True,
                                            density = True )
    fig.savefig(loc_fig + 'clear_sky_to_cloud_distr.png')

        

        
        
        
        
        
