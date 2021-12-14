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

def _ll_beta_mix(y, X, alpha1, beta1, alpha2, beta2, p):
    B1 = beta(alpha1, beta1).pdf(y)
    B2 = beta(alpha2, beta2).pdf(y)
    H = p * B1 + (1 - p) * B2
    return np.log(H).sum()    

def pdf_bmix(y, alpha1, beta1, alpha2, beta2, p):
    B1 = beta(alpha1, beta1).pdf(y)
    B2 = beta(alpha2, beta2).pdf(y)
    if p < 0: 
        p = 0
    elif p > 1: 
        p = 1
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
        if p < 0: 
            p = 0
        elif p > 1: 
            p = 1
        ll = _ll_beta_mix(self.endog, self.exog, alpha1, beta1, alpha2, beta2, p)
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

def _ll_beta(y, X, alpha1, beta1):
    B = beta(alpha1, beta1)
    return B.logpdf(y).sum()    

def pdf_b(y, alpha1, beta1):
    B = beta(alpha1, beta1).pdf(y)
    return B


class MyBetaML(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(MyBetaML, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        alpha, beta = params
        ll = _ll_beta(self.endog, self.exog, alpha, beta)
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
    
    
    
def _ll_beta_global(y, X, r, rho):
    y = CTHtoUnitInt(y)
    h = X[:,0]
    d = X[:,1]
    mu = r[0] + r[1] * h
    lognu = rho[0] + rho[1] * d + (rho[2] + rho[3] * d) * (h - rho[4]) ** 2
    nu = np.exp(lognu)
    alpha1 = mu * nu
    beta1 = nu - alpha1
    return beta(alpha1, beta1).logpdf(y).sum()    

def model1_cth(h, d, r, rho):
    mu = r[0] + r[1] * h
    lognu = rho[0] + rho[1] * d + (rho[2] + rho[3] * d) * (h - rho[4]) ** 2
    nu = np.exp(lognu)
    return mu, nu
    

class MyDepBetaML(GenericLikelihoodModel):
    """
    y : cth in m
    X = [h ,d] : cth and log of cod [m , .]
    """
    def __init__(self, endog, exog, **kwds):
        super(MyDepBetaML, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        r = params[:2]
        rho = params[2:7]
        ll = _ll_beta_global(self.endog, self.exog, r, rho)
        return -ll
    def fit(self, start_params=None, maxiter=10000, maxfun=50000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.clear()
        self.exog_names.append('r0')
        self.exog_names.append('r1')
        self.exog_names.append('rho0')
        self.exog_names.append('rho1')
        self.exog_names.append('rho2')
        self.exog_names.append('rho3')
        self.exog_names.append('rho4')
        if start_params == None:
            # Reasonable starting values
            start_params = np.array([0, 1 / 15e3, 1, 1, 1e-6, 1e-6, 8e3])
        return super(MyDepBetaML, self).fit(start_params=start_params, 
                                  maxiter=maxiter, maxfun=maxfun, 
                                  **kwds)