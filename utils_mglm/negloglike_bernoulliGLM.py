# Script for computing negative loglikelihood and its first and second-order derivates for Bernoulli GLM
#--------------------------------------------------------------------------------------------------------
import numpy as np
from utils_mglm.softplus import softplus
import scipy.stats as st

def negloglike_bernoulliGLM(weights, datas, inputs):
    """ 
    Compute negative log-likelihood of data under logistic regression model, plus gradient and Hessian
    Inputs:
    -------
    weights [M x nsamp] - weights 
    inputs [T x M] - regressors
    datas [T x 1] - output (binary vector of 1s and 0s).
    
    Outputs:
    --------
    negL [nsamp x 1] - negative loglikelihood
    dnegL [M x 1] - gradient
    H [M x M] - Hessian (2nd deriv matrix)
    """
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    nsamp = weights.shape[1]
    T = inputs.shape[0]
    if nsamp>1:
        wTx = wTx.ravel()
        f, df, ddf = softplus(wTx)
        f = np.reshape(f, (T,nsamp))
        wTx = np.reshape(wTx, (T,nsamp))
        # Negative LL
        negL = -(datas.T@wTx).T + np.sum(f, axis=0)[:,None]
    else:
        f, df, ddf = softplus(wTx)
        negL = -datas.T@wTx + np.sum(f)
    return negL

def dnegL(weights, datas, inputs):
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    # nsamp = weights.shape[1]
    # T = inputs.shape[0]
    # if nsamp>1:
    #     wTx = wTx.ravel()
    #     f, df, ddf = softplus(wTx)
    #     df = np.reshape(df, (T,nsamp))
    #     wTx = np.reshape(wTx, (T,nsamp))
    #     # Negative LL
    #     dnegL = (inputs.T@(df-datas)).ravel()
    # else:
    f, df, ddf = softplus(wTx)
    # Derivative
    dnegL = (inputs.T@(df[:,None]-datas)).ravel()
    return dnegL


def HessnegL( weights, datas, inputs):
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Hessian
    H = inputs.T@np.multiply(inputs, ddf[:,np.newaxis])
    return H


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


def logp_bernoulliGLM(weights, datas, inputs, prior_mean, prior_sigma):
    M = inputs.shape[1]
    prior_covariance = (prior_sigma**2)*np.eye(M,M)
    prior_w0 = prior_mean*np.ones((M))*0.1

    weights = weights[:,None]
    negloglike = negloglike_bernoulliGLM(weights, datas, inputs).ravel()
    logp = - negloglike + st.multivariate_normal.logpdf(weights.ravel(), mean=prior_w0, cov=prior_covariance).ravel()

    dnegloglike = dnegL(weights.ravel(), datas, inputs)
    inv_prior_covariance = (1/(prior_sigma**2))*np.eye(M,M)
    prior_w0 = prior_mean*np.ones((M))
    dlogp = -dnegloglike - (inv_prior_covariance@(weights.ravel()-prior_w0)).ravel()
    return  logp, dlogp