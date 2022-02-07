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
import itertools


from scipy.stats import norm, beta



sys.path.insert(0, './space-time-clouds/lib')
sys.path.insert(0, '../lib')
import ml_estimation as ml
import model1_explore as me
import model1 as mod




# variables
src_path = os.path.dirname(os.path.realpath(__file__))
input_file = src_path + '/input_model1.txt'
# input_file = './space-time-clouds/src/input_model1.txt'


# =============================================================================
#  from clear sky param view
# =============================================================================



def plotLocalParamfromCS(theta,
                         cmap = cmc.batlow,
                         **kwargs):
    vmin = theta.min()
    vmax = theta.max()
    fig, ax = plt.subplots(1,3, figsize = (20,5))
    prop = dict( cmap = cmap, vmin = vmin, vmax = vmax, )
    theta[:,:,0].plot(ax = ax[0], **prop, **kwargs)
    theta[:,:,1].plot(ax = ax[1], **prop, **kwargs)
    theta[:,:,2].plot(ax = ax[2], **prop, **kwargs)
    return fig, ax


def plotLocalParam(theta, 
                   cmap = cmc.batlow,
                   **kwargs):
    vmin = theta.min()
    vmax = theta.max()
    
    mu_d = theta.mu_d
    mu_csf = theta.mu_csf
    mu_h = theta.mu_h
    
    prop = dict( cmap = cmap, vmin = vmin, vmax = vmax, )

    
    fig, ax = plt.subplots(len(mu_d) * len(mu_h),len(mu_csf), figsize = (20,15))
    
    for h, d, csf in itertools.product(np.arange(len(mu_h)),
                                    np.arange(len(mu_d)),
                                    np.arange(len(mu_csf))):
        theta[h, d, :, :, csf].plot(ax = ax[3 * h +  d, csf], **prop, **kwargs)
        
    return fig, ax














