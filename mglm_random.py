import numpy as np
import numpy.random as npr
import random
from utils_mglm.selectbestinput import selectbestinput
import time



def mglm_random(seed, T, initial_inputs, K,true_mglm, test_mglm, input_list, burnin = 150, n_iters=300):
    """ Random sampling for fitting the model"""

    print("Fitting MGLM using random sampling")

    # Fixing random seed
    npr.seed(seed)
    # True parameters
    M = input_list.shape[1]
    true_pi = true_mglm.params[0]
    true_ws = np.reshape(true_mglm.params[1], (K,M))

    # Observations for initial samples
    init_time_bins = initial_inputs.shape[0]
    zs, observations = true_mglm.sample_y(init_time_bins, inputs=initial_inputs)

    # To keep track of inference after every time_step: 
    # ws_list contains posterior means for w
    # pis_list contains mean of pis
    weights_list = np.empty((T+1, K, M))
    pis_list = np.empty((T+1, K))
    posteriorcov = np.empty((T+1))


    # Run inference using the existing dataset, also returns the posterior over states which is useful for calcumating MI
    weights_sampled, pis_sampled, ll = test_mglm.fit_gibbs(observations, initial_inputs, burnin=burnin, n_iters = n_iters)

    # Store the model parameters
    weights_list[0] = np.mean(weights_sampled, axis=0)
    pis_list[0] = np.mean(pis_sampled, axis=0)
    # Store posterior covariance over model parameters
    if K>1:
        posteriorcov[0] = np.log(np.abs(np.var(pis_sampled[:,0])))
        for k in range(K):
            posteriorcov[0] = posteriorcov[0] +  np.linalg.slogdet(np.cov(weights_sampled[:,k], rowvar=False))[1] 
    else:
        posteriorcov[0] = np.linalg.slogdet(np.cov(weights_sampled, rowvar=False))[1] 

    # To store selected inputs at each step
    selected_inputs = []
        
    inputs = initial_inputs
    for t in range(T):
        init_samples = len(initial_inputs)
        print("Computing parameters of MGLM using "+str(t+1+init_samples)+" samples")
        # Select next stimuli randomly
        index = np.random.choice(np.arange(len(input_list)))
        x_new = input_list[index,:]

        # Obtain output from the true model
        z_new, observation_new = true_mglm.sample_y(T=1, inputs = np.reshape(np.array(x_new), (1,M)))
        # Append this to the list of inputs and outputs
        observations = np.concatenate((observations, observation_new), axis=0)
        inputs = np.concatenate((inputs,np.reshape(np.array(x_new), (1,M))),  axis=0)
        zs = np.concatenate((zs, z_new))

        # Run inference using the now new dataset
        initialize_w = np.reshape(weights_list[t], (K, 1, M))
        weights_sampled, pis_sampled, ll = test_mglm.fit_gibbs(observations, inputs, burnin=burnin, n_iters = n_iters, initialize = [pis_list[t], initialize_w])
        
        # Permute
        pis = np.mean(pis_sampled, axis=0)
        ws = np.mean(weights_sampled, axis=0)
        # permuting
        if K==2:
            if np.abs(pis[0]-true_pi[0])>np.abs(pis[0]-true_pi[1]):
                ws[0], ws[1] = (ws[1]).copy(), (ws[0]).copy()
                pis[0], pis[1] = (pis[1]).copy(), (pis[0]).copy()

        # Store the model parameters
        weights_list[t+1] = ws
        pis_list[t+1] = pis
        
        if K>1:
            posteriorcov[t+1] = np.abs(np.var(pis_sampled[:,0]))
            for k in range(K):
                posteriorcov[t+1] = posteriorcov[t+1] +  np.linalg.slogdet(np.cov(weights_sampled[:,k], rowvar=False))[1]   
        else:
            posteriorcov[t+1] = np.linalg.slogdet(np.cov(weights_sampled, rowvar=False))[1] 


        # To store selected inputs at each step
        selected_inputs.append(x_new)

    return pis_list, weights_list, selected_inputs, posteriorcov


