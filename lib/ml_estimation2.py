#!/usr/bin/env python3
# coding: utf-8

"""
Created on Thu Dec  2 15:09:18 2021

@author: Nerine
"""

import numpy as np
from scipy.stats import norm, beta, bernoulli
from statsmodels.base.model import GenericLikelihoodModel

# variables
h_max = 16e3 # m , maximum cloud top height TODO check

# functions
# =============================================================================
#  Conversion alpha, beta <-> mu, nu
# =============================================================================

def abToMn(alpha, beta):
    mu = alpha / (alpha + beta)
    nu = alpha + beta
    return mu, nu

def mnToAb(mu, nu):
    alpha = mu * nu
    beta = nu - alpha
    return alpha, beta

def mixmnToab(mu1, nu1, mu2, nu2, p):
    alpha1, beta1 = mnToAb(mu1, nu1)
    alpha2, beta2 = mnToAb(mu2, nu2)
    return alpha1, beta1, alpha2, beta2, p

# =============================================================================
# Beta mixture
# =============================================================================

def _ll_beta_mix(y, X, mu1, nu1, mu2, nu2, p):
    alpha1, beta1 = mnToAb(mu1, nu1)
    alpha2, beta2 = mnToAb(mu2, nu2)
    B1 = beta(alpha1, beta1).pdf(y)
    B2 = beta(alpha2, beta2).pdf(y)
    H = p * B1 + (1 - p) * B2
    return np.log(H).sum()    

def pdf_bmix(y, mu1, nu1, mu2, nu2, p):
    alpha1, beta1 = mnToAb(mu1, nu1)
    alpha2, beta2 = mnToAb(mu2, nu2)
    B1 = beta(alpha1, beta1).pdf(y)
    B2 = beta(alpha2, beta2).pdf(y)
    if p <= 0: 
        return B2
    elif p >= 1: 
        return B1
    H = p * B1 + (1 - p) * B2
    return H

def cdf_bmix(y, mu1, nu1, mu2, nu2, p):
    alpha1, beta1 = mnToAb(mu1, nu1)
    alpha2, beta2 = mnToAb(mu2, nu2)
    B1 = beta(alpha1, beta1).cdf(y)
    B2 = beta(alpha2, beta2).cdf(y)
    if p <= 0: 
        return B2
    elif p >= 1: 
        return B1
    H = p * B1 + (1 - p) * B2
    return H

class MyMixBetaML(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(MyMixBetaML, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        mu1, nu1 = params[:2]
        mu2, nu2 = params[2:4]
        p = params[4]
        if p < 0: 
            p = 0
        elif p > 1: 
            p = 1
        ll = _ll_beta_mix(self.endog, self.exog, mu1, nu1, mu2, nu2, p)
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
            start_params = np.array([1,1,1,1,0.5])
        return super(MyMixBetaML, self).fit(start_params=start_params, 
                                  maxiter=maxiter, maxfun=maxfun, 
                                  **kwds)
# =============================================================================
#  Single Beta
# =============================================================================

def MoM_sb(x):
    m1 = x.mean()
    m2 = (x**2).mean()
    alpha = m1 * (m1 - m2) /(m2 - m1**2)
    beta = (m1/m2 - 1) * (alpha +1)
    mu, nu = abToMn(alpha, beta)
    return mu, nu

def _ll_beta(y, X, mu1, nu1):
    alpha1, beta1 = mnToAb(mu1, nu1)
    B = beta(alpha1, beta1)
    return B.logpdf(y).sum()    

def pdf_b(y, mu, nu):
    alpha1, beta1 = mnToAb(mu, nu)
    B = beta(alpha1, beta1).pdf(y)
    return B

def cdf_b(y, mu, nu):
    alpha1, beta1 = mnToAb(mu, nu)
    B = beta(alpha1, beta1).cdf(y)
    return B

class MyBetaML(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(MyBetaML, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        mu, nu = params
        ll = _ll_beta(self.endog, self.exog, mu, nu)
        return -ll
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.pop()
        self.exog_names.append('alpha')
        self.exog_names.append('beta')                               
        if start_params == None:
            # Reasonable starting values
            start_params = np.append([1,1])
        return super(MyBetaML, self).fit(start_params=start_params, 
                                  maxiter=maxiter, maxfun=maxfun, 
                                  **kwds)
    
