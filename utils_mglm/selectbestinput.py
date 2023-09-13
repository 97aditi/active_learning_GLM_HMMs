import numpy as np 
import scipy.stats as st
from utils_mglm.softplus import *
import time 


def selectbestinput(test_mglm, input_list, weights_samples, pis_samples):
    """
    Function to select the next best input to train the model
    Params:
        test_mglm: A mixture of GLMs instance
        datas: Tx1
        inputs: TXM
        imput_list: List of all possible inputs
        weights_samples: samples of weight from the posterior
    Returns:
    Best stimuli from input_list 
    """

    # Number of samples from posterior
    nsamp = len(weights_samples)
    # Number of states
    K = test_mglm.K
    # Input dimension
    M = input_list.shape[1]

    xgrid = np.array(input_list)
    nstim = xgrid.shape[0]

    C = 2
    ygrid = np.arange(start=0, stop=C, step=1)
    ygrid = np.reshape(ygrid, (ygrid.shape[0],1))
    nygrid = ygrid.shape[0]

    # For every state, unravelling all w_j^Tx_t, this current implementation only works for Bernoulli GLM
    wTx = np.empty((nsamp*nstim, K))
    for k in range(K):
        wTx[:,k] = np.ravel(weights_samples[:,k,:]@xgrid.T)

    # Compute conditional and marginal over y
    # Evaluate p(y_t|k, theta_j) for each time point (theta_j is the jth set of sampled parameters)
    p = np.empty((nsamp,nstim,nygrid, K))
    # Evaluate p(y_t|theta_j)
    py_thetaj = 0
    for k in range(K):
        # log p(y_t = 1)
        logp = np.reshape(wTx[:,k] - only_softplus(wTx[:,k]), (nsamp, nstim))
        prob_1 = np.exp(logp)
        p[:,:,0,k] = 1-prob_1
        p[:,:,1,k] = prob_1
        py_thetaj = py_thetaj + (pis_samples[:,k][:,None]*np.ones((nsamp, nstim)))[:,:,None]*p[:,:,:,k]

    py = np.mean(np.reshape(py_thetaj, (nsamp, nstim, nygrid)), axis=0)
    # py = np.sum((impwts[:,None]*np.ones((nsamp,nstim)))[:,:,None]*py_thetaj, axis=0)
 
    # Compute entropy and conditional entropy
    Hy_theta_j = -np.nansum(py_thetaj*np.log(py_thetaj), axis=2)
    Hy_theta = np.mean(np.reshape(Hy_theta_j, (nsamp, nstim)), axis=0)
    # Hy_theta = Hy_theta_j.T@impwts[:,None]
    Hy = -np.nansum(py*np.log(py), axis=1)

    # Compute MI
    mi = Hy-Hy_theta.ravel()


    x_new = xgrid[np.argmax(mi)]
    return x_new, np.argmax(mi)
