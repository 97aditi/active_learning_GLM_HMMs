# Script for computing negative loglikelihood and its first and second-order derivates for Bernoulli GLM
#--------------------------------------------------------------------------------------------------------
import numpy as np
import scipy.stats as st

def softplus(x):
    """ Function to compute softplus and return its first and second derivatives
        f(x) = log(1+exp(x))
        Also known as the soft-rectification function
    """
    f = np.log(1+np.exp(x))
    df = np.exp(x)/(1+np.exp(x))
    ddf = np.exp(x)/((1+np.exp(x))**2)

    # Check for small values to avoid underflow errors
    if np.any(x<-20):
        iix = np.where(x<-20)
        f[iix] = np.exp(x[iix])
        df[iix] = f[iix]
        ddf[iix] = f[iix]

    # Check for large values to avoid overflow errors
    if np.any(x>30):
        iix = np.where(x>30)
        f[iix] = x[iix]
        df[iix] = 1;
        ddf[iix] = 0;

    return f, df, ddf


def only_softplus(x):
    """ Function to compute softplus and return its first and second derivatives
        f(x) = log(1+exp(x))
        Also known as the soft-rectification function
    """
    f = np.log(1+np.exp(x))

    # Check for small values to avoid underflow errors
    if np.any(x<-20):
        iix = np.where(x<-20)
        f[iix] = np.exp(x[iix])

    # Check for large values to avoid overflow errors
    if np.any(x>30):
        iix = np.where(x>30)
        f[iix] = x[iix]

    return f


def negloglike_bernoulliGLM(weights, datas, inputs):
    """ 
    Compute negative log-likelihood of data under logistic regression model, plus gradient and Hessian
    Inputs:
    -------
    weights [M x 1] - weights 
    inputs [T x M] - regressors
    datas [T x 1] - output (binary vector of 1s and 0s).
    
    Outputs:
    --------
    negL [1 x 1] - negative loglikelihood
    dnegL [M x 1] - gradient
    H [M x M] - Hessian (2nd deriv matrix)
    """
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Negative LL
    negL = -datas.T@wTx + np.sum(f)
    return negL

def dnegL(weights, datas, inputs):
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Derivative
    dnegL = (inputs.T@(df[:,np.newaxis]-datas)).ravel()
    return dnegL

def expecteddnegL(weights, datas, inputs, qzt_ks):
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Derivative
    expecteddnegL = ((inputs).T@(qzt_ks[:,np.newaxis]*(df[:,np.newaxis]-datas))).ravel()
    return expecteddnegL

def HessnegL(weights, datas, inputs):
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Hessian
    H = inputs.T@np.multiply(inputs, ddf[:,np.newaxis])
    return H

def expectedHessnegL(weights, datas, inputs, qzt_ks):
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Hessian
    expectedH = (qzt_ks[:,np.newaxis]*inputs).T@np.multiply(inputs, ddf[:,np.newaxis])
    return expectedH


def neglogp_bernoulliGLM(weights, datas, inputs, prior_mean, prior_sigma):
    """
    weights: [nsamp x M]
    """
    M = inputs.shape[1]
    prior_covariance = (prior_sigma**2)*np.eye(M,M)
    prior_w0 = prior_mean*np.ones((M))*0.1

    nsamp = weights.shape[0]
    negloglike = negloglike_bernoulliGLM(weights.T, datas, inputs).ravel()
    neglogp = negloglike - st.multivariate_normal.logpdf(weights, mean=prior_w0, cov=prior_covariance)

    return  neglogp 

