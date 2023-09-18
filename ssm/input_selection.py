import numpy as np 
from ssm.bernoulliglm import only_softplus
import time 
import scipy.optimize as opt
import scipy.stats as st

def input_selection(seed, hmm, stimuli, params_samples, pzts_persample):
    """
    Function select the next best input to train IO-HMMs via active sampling (only works when C=2)
    Input:
        hmm: Instance of the IO-HMM class
        datas: [T x D]
        inputs: [T X M]
        stimuli: list of potential stimuli
        params_samples: Sampled parameters of HMM (through Gibbs Sampling)
        pzts_persample: [nsamp x K] forward probabilities of states computed during Gibbs Sampling
    Output:

    """

    np.random.seed(seed)

    # Number of samples from posterior
    nsamp = len(params_samples[0])
    # Number of states
    K = hmm.K

    xgrid = np.array(stimuli)
    nstim = xgrid.shape[0]

    C = 2
    ygrid = np.arange(start=0, stop=C, step=1)
    ygrid = np.reshape(ygrid, (ygrid.shape[0],1))
    nygrid = ygrid.shape[0]

    # Extract paramaters
    pi0_samples, Ps_samples, weights_samples = params_samples
    # print(pzts_persample)
    # First, compute probability of states (p(z_{t+1} \mid x_{1:t+1}, y_{1:t}, \theta_j)) 
    # [CHECKED MULTIPLE TIMES]
    pz_samples = np.sum((pzts_persample[:,:,None]*Ps_samples), axis=1)

    # For every state, unravelling all w_j^Tx_t, this current implementation only works for Bernoulli GLM
    wTx = np.empty((nsamp, nstim, K))
    for k in range(K):
        wTx[:,:, k] = weights_samples[:,k,:]@xgrid.T
    wTx = np.reshape(wTx, (nsamp*nstim, K))

    # Compute Likelihood and marginal over y
    # Evaluate p(y_t|k, theta_j) for each time point (theta_j is the jth set of sampled parameters)
    p = np.empty((nsamp,nstim,nygrid, K))
    # Evaluate p(y_t|theta_j)
    py_thetaj = np.zeros((nsamp, nstim, nygrid))
    for k in range(K):
        # Computing observation likelihoods first (can't this code be made generic for all observationss?)
        # log p(y_t = 1)
        logp = np.reshape(wTx[:,k] - only_softplus(wTx[:,k]), (nsamp, nstim))
        prob_1 = np.exp(logp)
        p[:,:,0,k] = 1-prob_1
        p[:,:,1,k] = prob_1
        # CHECK THIS
        py_thetaj = py_thetaj + (pz_samples[:,k][:,None]*np.ones((nsamp, nstim)))[:,:,None]*p[:,:,:,k]
    
    # sum of py_thetaj along axis 2 should be 1
    assert np.allclose(np.sum(py_thetaj, axis=2), np.ones((nsamp,nstim))), "py_thetaj does not sum to 1"
    py = np.mean(np.reshape(py_thetaj, (nsamp, nstim, nygrid)), axis=0)

    # Compute entropy and conditional entropy
    Hy_theta_j = -np.nansum(py_thetaj*np.log2(py_thetaj), axis=2)
    Hy_theta = np.mean(np.reshape(Hy_theta_j, (nsamp, nstim)), axis=0)
    Hy = -np.nansum(py*np.log2(py), axis=1)

    # Compute MI
    mi = Hy-Hy_theta
    x_new = xgrid[np.argmax(mi)]
    return x_new, mi


def input_selection_cont(seed, hmm, stimuli_range, params_samples, pzts_persample):
    """
    Function select the next best stimuli to train GLM-HMMs via active sampling (only works when C=2)
    Input:
        hmm: Instance of the GLM-HMM class
        stimuli_range: range of potential inputs
        params_samples: Sampled parameters of HMM (through Gibbs Sampling)
        pzts_persample: [nsamp x K] forward probabilities of states computed during Gibbs Sampling
    Output:

    """
    np.random.seed(seed)
    
    # Number of samples from posterior
    nsamp = len(params_samples[0])
    # Number of states
    K = hmm.K
    M = hmm.M


    C = 2
    ygrid = np.arange(start=0, stop=C, step=1)
    ygrid = np.reshape(ygrid, (ygrid.shape[0],1))
    nygrid = ygrid.shape[0]

    # Function to optimize mutual info
    def max_info(x, params_samples, pzts_persample):
         # Extract paramaters
        pi0_samples, Ps_samples, weights_samples = params_samples
        # First, compute probability of states (p(z_{t+1} \mid x_{1:t+1}, y_{1:t}, \theta_j)) 
        pz_samples = np.sum((pzts_persample[:,:,None]*Ps_samples), axis=1)

        # For every state, unravelling all w_j^Tx_t, this current implementation only works for Bernoulli GLM
        x_full = np.ones((M,1))
        x_full[0] = x
        wTx = np.empty((nsamp, K))        
        for k in range(K):
            wTx[:,k] = np.ravel(weights_samples[:,k,:]@x_full) 

        # Compute Likelihood and marginal over y
        # Evaluate p(y_t|k, theta_j) for each time point (theta_j is the jth set of sampled parameters)
        p = np.empty((nsamp,nygrid, K))
        # Evaluate p(y_t|theta_j)
        py_thetaj = 0
        for k in range(K):
            # Computing observation likelihoods first (can't this code be made generic for all observationss?)
            # log p(y_t = 1)
            logp = np.reshape(wTx[:,k] - only_softplus(wTx[:,k]), (nsamp,))
            prob_1 = np.exp(logp)
            p[:,0,k] = 1-prob_1
            p[:,1,k] = prob_1
            # CHECK THIS
            py_thetaj = py_thetaj + (pz_samples[:,k][:,None])*p[:,:,k]
        
        py = np.mean(np.reshape(py_thetaj, (nsamp, nygrid)), axis=0)

        # Compute entropy and conditional entropy
        Hy_theta_j = -np.nansum(py_thetaj*np.log2(py_thetaj), axis=1)
        Hy_theta = np.mean(np.reshape(Hy_theta_j, (nsamp,)), axis=0)
        Hy = -np.nansum(py*np.log2(py), axis=0)

        # Compute MI
        negmi = Hy_theta - Hy
        return negmi

    results = opt.minimize(max_info, x0 = (0,), args=(params_samples, pzts_persample), bounds = stimuli_range)
    x_new = np.ones((M,1))
    x_new[0] = results['x']

    return x_new