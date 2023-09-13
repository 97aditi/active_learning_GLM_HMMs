# Script for computing negative loglikelihood and its first and second-order derivates for Bernoulli GLM
#--------------------------------------------------------------------------------------------------------
import numpy as np
from utils_mglm.softplus import softplus
import scipy.stats as st

def negloglike_Mstep_bernoulliGLM(weights, datas, inputs, pstate, prior_mean, prior_sigma):
    """ 
    Compute negative log-likelihood of data under logistic regression model, plus gradient and Hessian
    Inputs:
    -------
    inputs [T x M] - regressors
    datas [T x 1] - output (binary vector of 1s and 0s).
    pstate [N x 1] - probability of being in state at each time bin (serve as weights)
    
    Outputs:
    --------
    negL [1 x 1] - negative loglikelihood
    dnegL [M x 1] - gradient
    H [M x M] - Hessian (2nd deriv matrix)
    """
    M = inputs.shape[1]
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Negative LL
    prior_w0 = prior_mean*np.ones((M))
    prior_covariance = (prior_sigma**2)*np.eye(M,M)
    negL = -(pstate[:, np.newaxis]*datas).T@wTx + np.sum(pstate*f)- st.multivariate_normal.logpdf(weights, mean=prior_w0, cov=prior_covariance)
    return negL

def dnegL_Mstep(weights, datas, inputs, pstate, prior_mean, prior_sigma):
    M = inputs.shape[1]
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Derivative
    inv_prior_covariance = (1/(prior_sigma**2))*np.eye(M,M)
    prior_w0 = prior_mean*np.ones((M))
    dnegL = (inputs.T@((df-datas.ravel())*pstate)[:, np.newaxis]).ravel()+ inv_prior_covariance@(weights-prior_w0)
    return dnegL



def HessnegL_Mstep( weights, datas, inputs, pstate, prior_mean, prior_sigma):
    M = inputs.shape[1]
    # Compute projection of inputs onto GLM weights for each class
    wTx = inputs@weights
    # Evaluating softplus and its derivatives
    f, df, ddf = softplus(wTx)
    # Hessian
    inv_prior_covariance = (1/(prior_sigma**2))*np.eye(M,M)
    H = inputs.T@np.multiply(inputs, (pstate*ddf)[:,np.newaxis])+ inv_prior_covariance
    return H