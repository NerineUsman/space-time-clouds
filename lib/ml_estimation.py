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
h_max = 15e3 # m , maximum cloud top height TODO check

# functions

def redtoInt(x, a, b):
    if x < a:
        x = a
    elif x> b:
        x = b
    return x

def _ll_ols(y, X, beta, gamma):
    mu = X.dot(beta)
    sigma = X[:,:2].dot(gamma)
    return norm(mu,sigma).logpdf(y).sum()    

def model1(h, d, beta, gamma):
    X = np.array([1, h, d, h*d])
    mu = X.dot(beta)
    sigma = X[:2].dot(gamma)
    return mu, sigma
    
def pdf_norm(y, mu, sigma):
    return norm(mu, sigma).pdf(y)

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
    if p <= 0: 
        return B2
    elif p >= 1: 
        return B1
    H = p * B1 + (1 - p) * B2
    return H

def CTHtoUnitInt(h):
    return (h + 1) / (h_max + 1.1)

def UnitInttoCTH(h_):
    return (h_ + .001) * h_max *  .998

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
    
# =============================================================================
#     MIX beta global
# =============================================================================
    #start param global 
def mu1_est(h, d):
    return CTHtoUnitInt(h)

gamma1 = np.array([5, 2 * np.pi / h_max , 3, .2])
a = np.array([.8, .02])
ab = np.array([.9e-4, .3, .5, -.6, -.05, .35, 7e3])
gamma2 = np.array([-2/5 *1e-7, 3])

def nu1_est(h, d, gamma1 = gamma1):
    a = gamma1[2] + gamma1[3] *d
    return (gamma1[0] - a) * np.cos(2 * np.pi / h_max * h) + a


def prob(h, d, a = a):
    p = a[0] + a[1] * d
    p = np.where(p>1, 1, 
                 np.where(p < 0 , 0 , p))
    return p


def mu2_est(h, d, ab = ab):
    a = ab[:5]
    b = ab[5:7]
    
    if a[1] + a[4] * d + a[0] * h < b[0]:
        out = a[1] + a[4] * d + a[0] * h 
    elif h < b[1]:
        out = a[2] + a[4] * d + a[0] * h
    else:
        out = a[3] - a[4] * d + a[0] * h
    
    if out <= 0:
        out = .001
    elif out >= 1:
        out = .999
    return out
    

def nu2_est(h, d, gamma2 = gamma2):
    return gamma2[0] * (h - 7e3) ** 2 + gamma2[1]
    
def _ll_mixbeta_global(y, X, gamma1, a, ab, gamma2):
    y = CTHtoUnitInt(y)
    h = X[:,0]
    d = X[:,1]
    mu1 = mu1_est(h, d)
    lognu1 = nu1_est(h, d, gamma1 = gamma1)
    p = prob(h, d, a = a)
    mu2 = np.array([mu2_est(hi, di, ab = ab) for (hi,di) in zip (h, d)] )
    lognu2 = nu2_est(h, d, gamma2 = gamma2)
    
    
    
    nu1 = np.exp(lognu1)
    alpha1 = mu1 * nu1
    beta1 = nu1 - alpha1
    
    nu2 = np.exp(lognu2)
    alpha2 = mu2 * nu2
    beta2 = nu2 - alpha2
    return _ll_beta_mix(y, X, alpha1, beta1, alpha2, beta2, p)

# def model1_mix_cth(h, d, r, rho):
#     mu = r[0] + r[1] * h
#     lognu = rho[0] + rho[1] * d + (rho[2] + rho[3] * d) * (h - rho[4]) ** 2
#     nu = np.exp(lognu)
#     return mu, nu
    


class MyDepMixBetaML(GenericLikelihoodModel):
    """
    y : cth in m
    X = [h ,d] : cth and log of cod [m , .]
    """
    def __init__(self, endog, exog, **kwds):
        super(MyDepMixBetaML, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        gamma1 = params[:4]
        a = params[4:6]
        ab = params[6:13]
        gamma2 = params[13: 15]    
        print(params)
        ll = _ll_mixbeta_global(self.endog, self.exog, gamma1, a, ab, gamma2)
        return -ll
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.clear()
        self.exog_names.append('gamma10')
        self.exog_names.append('gamma11')
        self.exog_names.append('gamma12')
        self.exog_names.append('gamma13')
        self.exog_names.append('pa0')
        self.exog_names.append('pa1')
        self.exog_names.append('a0')
        self.exog_names.append('a1')
        self.exog_names.append('a2')
        self.exog_names.append('a3')
        self.exog_names.append('a4')
        self.exog_names.append('b0')
        self.exog_names.append('b1')
        self.exog_names.append('gamma20')
        self.exog_names.append('gamma21')
        if start_params == None:
            # Reasonable starting values
            start_params = np.concatenate([np.array([5, 2 * np.pi / h_max , 3, .2]),
                                     np.array([.8, .02]),
                                     np.array([.9e-4, .3, .5, -.6, -.05, .35, 7e3]),
                                     np.array([-2/5 *1e-7, 3])])
        return super(MyDepMixBetaML, self).fit(start_params=start_params, 
                                  maxiter=maxiter, maxfun=maxfun, 
                                  **kwds)
    
    
# =============================================================================
#   Cloud to clear sky
# =============================================================================
 
def model1_p_cs(h, d, param = [11e3, .5e-5, .1,  1.2]):
    p =  (np.exp(-1/((h - 7.5e3)/param[0])**2)  -  param[1] * h + param[2] ) * \
        np.exp(-d) * param[3]
    p = p * (p > 0)
    return p

    
def _ll_p_cs_global(y, X, params):
    h = X[:,0]
    d = X[:,1]
    
    p = model1_p_cs(h, d, params)
    
    return bernoulli(p).logpmf(y).sum()  



class MyDepPcsML(GenericLikelihoodModel):
    """
    y : 1, if next state is cs, 0 if next state is cloud
    X = [h ,d] : cth and log of cod [m , .]
    """
    def __init__(self, endog, exog, **kwds):
        super(MyDepPcsML, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        print(params)
        ll = _ll_p_cs_global(self.endog, self.exog, params)
        return -ll
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.clear()
        self.exog_names.append('c1')
        self.exog_names.append('c2')
        self.exog_names.append('c3')
        self.exog_names.append('c4')

        if start_params == None:
            # Reasonable starting values
            start_params = np.array([11e3, .5e-5, .1,  1.2])
        return super(MyDepPcsML, self).fit(start_params=start_params, 
                                  maxiter=maxiter, maxfun=maxfun, 
                                  **kwds)
    
    