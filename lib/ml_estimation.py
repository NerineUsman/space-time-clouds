#!/usr/bin/env python3
# coding: utf-8

"""
Created on Thu Dec  2 15:09:18 2021

@author: Nerine
"""

import numpy as np
from scipy.stats import norm, beta
from statsmodels.base.model import GenericLikelihoodModel

# variables
h_max = 15e3 # m , maximum cloud top height TODO check

# functions
def _ll_ols(y, X, beta, gamma):
    mu = X.dot(beta)
    sigma = X[:,:2].dot(gamma)
    return norm(mu,sigma).logpdf(y).sum()    

def model1(h, d, beta, gamma):
    X = np.array([1, h, d, h*d])
    mu = X.dot(beta)
    sigma = X[:2].dot(gamma)
    return mu, sigma
    

class MyDepNormML(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(MyDepNormML, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        gamma = params[-2:]
        beta = params[:-2]
        ll = _ll_ols(self.endog, self.exog, beta, gamma)
        return -ll
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('gamm1')
        self.exog_names.append('gamm2')
        if start_params == None:
            # Reasonable starting values
            start_params = np.append(np.zeros(self.exog.shape[1]), [0.5,.5])
        return super(MyDepNormML, self).fit(start_params=start_params, 
                                  maxiter=maxiter, maxfun=maxfun, 
                                  **kwds)

def _ll_beta(y, X, alpha1, beta1, alpha2, beta2, p):
    B1 = beta(alpha1, beta1).pdf(y)
    B2 = beta(alpha2, beta2).pdf(y)
    H = p * B1 + (1 - p) * B2
    return np.log(H).sum()    

def pdf_bmix(y, alpha1, beta1, alpha2, beta2, p):
    B1 = beta(alpha1, beta1).pdf(y)
    B2 = beta(alpha2, beta2).pdf(y)
    H = p * B1 + (1 - p) * B2
    return H

def CTHtoUnitInt(h):
    return (h + 1) / (h_max + 1.1)

def UnitInttoCTH(h_):
    return h_ * (h_max + 1.1) - 1

class MyMixBetaML(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(MyMixBetaML, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        alpha1, beta1 = params[:2]
        alpha2, beta2 = params[2:4]
        p = params[4]
        ll = _ll_beta(self.endog, self.exog, alpha1, beta1, alpha2, beta2, p)
        return -ll
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.pop()
        self.exog_names.append('alpha1')
        self.exog_names.append('beta1')
        self.exog_names.append('alpha2')
        self.exog_names.append('beta2')
        self.exog_names.append('p')                                    
        if start_params == None:
            # Reasonable starting values
            start_params = np.append(np.zeros(self.exog.shape[1]), [0.5])
        return super(MyMixBetaML, self).fit(start_params=start_params, 
                                  maxiter=maxiter, maxfun=maxfun, 
                                  **kwds)
    
    