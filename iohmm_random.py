import numpy as np
import numpy.random as npr
import random
from ssm.input_selection import input_selection
import time
from ssm.util import one_hot, find_permutation, permute_params
import scipy.stats as st


def iohmm_random_gibbs(seed, T, initial_inputs, K, true_iohmm, test_iohmm, input_list, method = 'gibbs', **kwargs):
    """ Randomly selecting inputs and using Gibbs sampling for fitting the model"""

    print("Random input selection for IO-HMMs; using "+str(method)+" for fitting the model")

    # Fixing random seed
    npr.seed(seed)
    # True parameters
    M = input_list.shape[1]
    true_pi0 = np.exp(true_iohmm.params[0])
    true_Ps = np.exp(true_iohmm.params[1])
    true_obsparams = np.reshape(true_iohmm.params[2], (K,M))

    # Observations for initial samples
    init_time_bins = len(initial_inputs[0])
    # Since we only have one sessiond
    latents, obs = true_iohmm.sample(init_time_bins, input=initial_inputs[0])
    observations = []
    observations.append(obs)
    zs = []
    zs.append(latents)

    # To keep track of inference after every time_step: 
    obsparams_list = np.empty((T+1, K, M))
    pi0_list = np.empty((T+1, K))
    Ps_list = np.empty((T+1, K, K))
    error_obsparams = np.empty((T+1))
    error_Ps = np.empty((T+1))
    posteriorcov = np.empty((T+1))

    # Run inference using the existing dataset, also returns the posterior over states which is useful for calcumating MI
    if method=='gibbs_parallel':
        obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=initial_inputs, method=method, zs=zs, **kwargs ) 
    else:
        obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=initial_inputs, method=method, **kwargs )
    
    # Store the model parameters
    obsparams_list[0] = np.mean(obsparams_sampled, axis=0)
    pi0_list[0] = np.mean(pi0_sampled, axis=0)
    Ps_list[0] = np.mean(Ps_sampled, axis=0)
    ravelled_obsparams = np.reshape(obsparams_sampled, (obsparams_sampled.shape[0],obsparams_sampled.shape[1]*obsparams_sampled.shape[2]))
    ravelled_Ps = np.reshape(Ps_sampled[:,:,:-1], (Ps_sampled.shape[0], (K)*(K-1)))
    params = np.hstack((ravelled_obsparams,ravelled_Ps))
    cov = np.cov(params, rowvar = False)
    posteriorcov[0] =  0.5*np.linalg.slogdet(cov)[1] 

    # To store selected inputs at each step
    selected_inputs = []
        
    inputs = initial_inputs

    for t in range(T):
        print("Computing parameters of IO-HMM using "+str(t+1+init_time_bins)+" samples")
        # Select next input from a list of possible input input
        index = np.random.choice(np.arange(len(input_list)))
        x_new = input_list[index,:]
        # Obtain output from the true model
        z_new, obs_new = true_iohmm.sample(T=1, input = np.reshape(np.array(x_new), (1,M)), prefix=(zs[0], observations[0]))
        # Append this to the list of inputs and outputs
        observations[0] = np.concatenate((observations[0], obs_new), axis=0)
        inputs[0] = np.concatenate((inputs[0],np.reshape(np.array(x_new), (1,M))),  axis=0)
        zs[0] = np.concatenate((zs[0], z_new))

        # Run inference using the now new dataset
        if method=="gibbs_parallel":
            obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method=method, zs=zs, **kwargs)
        else:
            obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method=method, **kwargs)

        perm = find_permutation(zs[0], test_iohmm.most_likely_states(observations[0], input=inputs[0]), K, K)
        test_iohmm.permute(perm)

        obsparams_sampled = obsparams_sampled[:,perm,:]
        Ps_sampled = Ps_sampled[:,perm,:]
        Ps_sampled = Ps_sampled[:,:,perm]
        pi0_sampled = pi0_sampled[:, perm]

        # Store the model parameters
        obsparams_list[t+1] = np.mean(obsparams_sampled, axis=0)
        Ps_list[t+1] = np.mean(Ps_sampled, axis=0)
        pi0_list[t+1] = np.mean(pi0_sampled, axis=0)
        ravelled_obsparams = np.reshape(obsparams_sampled, (obsparams_sampled.shape[0],obsparams_sampled.shape[1]*obsparams_sampled.shape[2]))
        ravelled_Ps = np.reshape(Ps_sampled[:,:,:-1], (Ps_sampled.shape[0], (K)*(K-1)))
        params = np.hstack((ravelled_obsparams,ravelled_Ps))
        cov = np.cov(params, rowvar = False)
        posteriorcov[t+1] =  0.5*np.linalg.slogdet(cov)[1]  

        # To store selected inputs at each step
        selected_inputs.append(x_new)

    return pi0_list, Ps_list, obsparams_list, posteriorcov, selected_inputs


