import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import random
from ssm.input_selection import input_selection
import time
from ssm.util import one_hot, find_permutation, permute_params
import scipy.stats as st


def iohmm_infomax_gibbs(seed, T, initial_inputs, K, true_iohmm, test_iohmm, input_list, method = 'gibbs', num_iters = 400, burnin=100, **kwargs):
    """ Bayesian active learning using gibbs sampling for fitting the model"""

    print("Infomax (Gibbs) learning for IO-HMMs; using "+str(method)+" for fitting the model")

    # Fixing random seed
    npr.seed(seed)
    # True parameters
    M = input_list.shape[1]

    true_pi0 = np.exp(true_iohmm.params[0])
    true_Ps = np.exp(true_iohmm.params[1])
    true_obsparams = np.reshape(true_iohmm.params[2], (K,M))

    # Observations for initial samples
    init_time_bins = len(initial_inputs[0])
    # Since we only have one session
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
        obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=initial_inputs, method=method, num_iters = num_iters, burnin=burnin, **kwargs )

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
        x_new, mi = input_selection(seed, test_iohmm, input_list, [pi0_sampled, Ps_sampled, obsparams_sampled], pzts_persample)
        print("selected input: "+str(x_new))
        # Obtain output from the true model
        z_new, observation_new = true_iohmm.sample(T=1, input = np.reshape(np.array(x_new), (1,M)), prefix=(zs[0], observations[0]))
        # Append this to the list of inputs and outputs
        observations[0] = np.concatenate((observations[0], observation_new), axis=0)
        inputs[0] = np.concatenate((inputs[0],np.reshape(np.array(x_new), (1,M))),  axis=0).copy()
        zs[0] = np.concatenate((zs[0], z_new))

        # Run inference using the now new dataset
        if method=="gibbs_parallel":
            obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method=method, zs=zs, **kwargs)
        else:
            obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method=method, num_iters = num_iters, burnin=burnin, **kwargs)

        # Permute
        perm = find_permutation(zs[0], test_iohmm.most_likely_states(observations[0], input=inputs[0]), K, K)
        test_iohmm.permute(perm)
        obsparams_sampled = obsparams_sampled[:,perm,:]
        Ps_sampled = Ps_sampled[:,perm,:]
        Ps_sampled = Ps_sampled[:,:,perm] 
        pi0_sampled = pi0_sampled[:,perm] 
        pzts_persample = pzts_persample[:,perm]  

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


def iohmm_infomax_VI(seed, T, initial_inputs, K, true_iohmm, test_iohmm, input_list, n_iters=500):
    """ Bayesian active learning  using VI for fitting the model"""

    print("Infomax (VI) learning for IO-HMMs")

    # Fixing random seed
    npr.seed(seed)
    # True parameters
    M = initial_inputs[0].shape[1]
    true_pi0 = np.exp(true_iohmm.params[0])
    true_Ps = np.exp(true_iohmm.params[1])
    true_obsparams = np.reshape(true_iohmm.params[2], (K,M))

    # Observations for initial samples
    init_time_bins = len(initial_inputs[0])
    # Since we only have one session
    latents, obs = true_iohmm.sample(init_time_bins, input=initial_inputs[0])
    observations = []
    observations.append(obs)
    zs = []
    zs.append(latents)

    # Run inference using the existing dataset, also returns the posterior over states which is useful for calcumating MI
    posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs, lls  = test_iohmm.fit(observations, inputs=initial_inputs, method="variational", num_iters=n_iters)
    learned_parameters = [posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs]

    # Permute based on states
    perm = find_permutation(zs[0], test_iohmm.most_likely_states(observations[0], input=initial_inputs[0]), K, K)
    # Permute the model's parameters
    test_iohmm.permute(perm)
    # Permute the learned distributions over parameters
    posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs = permute_params(posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs, perm) 

    # Sample model parameters (\theta_j) from the learned variational distributions
    num_samples = 500
    pi0_sampled = st.dirichlet.rvs(posterior_alpha_0, size=num_samples)
    Ps_sampled = np.empty((num_samples, K, K))
    for k in range(K):
        Ps_sampled[:,k] = st.dirichlet.rvs(posterior_alpha_Ps[k], size=num_samples)
    obsparams_sampled= np.empty((num_samples, K, M))
    for k in range(K):
        obsparams_sampled[:,k] = st.multivariate_normal.rvs(mean=posterior_means[k], cov=posterior_covs[k], size=num_samples)

    # Computing p(z_t \mid x_{1:t}, y_{1:t}, \theta_j) for every parameter setting \theta_j, we need this for input selection later 
    pzts_persample = np.empty((num_samples, K))
    Ts = [true_choice.shape[0] for true_choice in observations]
    for n in range(num_samples):
        # Obtain observation Likelihoods (TXKXC)
        test_iohmm.observations.Wk = obsparams_sampled[n].reshape((K,1,M))
        log_likes = [test_iohmm.observations.calculate_logits(inpt) for inpt in initial_inputs]
        # Extract only emission potentials for the given observations (TXK)
        log_Ls = [np.empty((T,K)) for T in Ts]
        for sess in range(len(Ts)):
            for t in range(Ts[sess]):
                log_Ls[sess][t,:] = log_likes[sess][t,:,int(observations[sess][t].ravel())]
        pzts = [test_iohmm.posterior_over_states_using_past_samples(pi0_sampled[n], np.reshape(Ps_sampled[n], (1,K,K)), log_L) for log_L in log_Ls]
        pzts_persample[n] = pzts[0]

    # To store selected inputs at each step
    selected_inputs = np.zeros((T, M))
        
    inputs = initial_inputs
    for t in range(T):
        print("Selecting inputs for trial #"+str(t+1+init_time_bins))
        # Select next input from a list of possible input input
        x_new, mi = input_selection(seed, test_iohmm, input_list, [pi0_sampled, Ps_sampled, obsparams_sampled], pzts_persample)
       
        # Obtain output from the true model
        z_new, observation_new = true_iohmm.sample(T=1, input = np.reshape(np.array(x_new), (1,M)), prefix=(zs[0], observations[0]))
        # Append this to the list of inputs and outputs
        observations[0] = np.concatenate((observations[0], observation_new), axis=0)
        inputs[0] = np.concatenate((inputs[0],np.reshape(np.array(x_new), (1,M))),  axis=0)
        zs[0] = np.concatenate((zs[0], z_new))

        # Run inference using the now new dataset
        posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs, lls  = test_iohmm.fit(observations, inputs, method="variational", num_iters=n_iters)
        learned_parameters = [posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs]

        # Permute
        perm = find_permutation(zs[0], test_iohmm.most_likely_states(observations[0], input=inputs[0]), K, K)
        test_iohmm.permute(perm)
        posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs = permute_params(posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs, perm) 

        # Sample model parameters
        pi0_sampled = st.dirichlet.rvs(posterior_alpha_0, size=num_samples)
        Ps_sampled = np.empty((num_samples, K, K))
        for k in range(K):
            Ps_sampled[:,k] = st.dirichlet.rvs(posterior_alpha_Ps[k], size=num_samples)
        obsparams_sampled= np.empty((num_samples, K, M))
        for k in range(K):
            obsparams_sampled[:,k] = st.multivariate_normal.rvs(mean=posterior_means[k], cov=posterior_covs[k], size=num_samples) 

        # Computing p(z_t \mid x_{1:t}, y_{1:t}, \theta_j) for every parameter setting \theta_j, we need this for input selection later  
        pzts_persample = np.empty((num_samples, K))
        # Obtain observation Likelihoods (TXKXC)
        Ns = [true_choice.shape[0] for true_choice in observations]
        for n in range(num_samples):
            test_iohmm.observations.Wk = obsparams_sampled[n].reshape((K,1,M))
            log_likes = [test_iohmm.observations.calculate_logits(inpt) for inpt in inputs]
            # Extract only emission potentials for the given observations (TXK)
            log_Ls = [np.empty((N,K)) for N in Ns]
            for sess in range(len(Ns)):
                for l in range(Ns[sess]):
                    log_Ls[sess][l,:] = log_likes[sess][l,:,int(observations[sess][l].ravel())]
            pzts = [test_iohmm.posterior_over_states_using_past_samples(pi0_sampled[n], np.reshape(Ps_sampled[n], (1,K,K)), log_L) for log_L in log_Ls]
            pzts_persample[n] = pzts[0]

        # To store selected inputs at each step
        selected_inputs[t, :M] = x_new.ravel()

    # now running inference using selected inputs
    pi0_list, Ps_list, obsparams_list, posteriorcov = fit_gibbs(seed, T, K, true_iohmm, test_iohmm, selected_inputs, initial_inputs)

    return pi0_list, Ps_list, obsparams_list, posteriorcov, selected_inputs

# Training IO-HMMS using selected inputs
def fit_gibbs(seed, T, K, true_iohmm, test_iohmm, selected_inputs, initial_inputs, burnin = 200, n_iters=500):
    """ Bayesian active learning for fitting the model"""
    print("Fitting IO-HMM using inputs selected by VI")
    # Fixing random seed
    np.random.seed(seed)
    M = 2
    initial_T = len(initial_inputs[0])

    # Since we only have one session
    latents, obs = true_iohmm.sample(initial_T, input=initial_inputs[0])
    observations = []
    observations.append(obs)
    zs = []
    zs.append(latents)

    # To keep track of inference after every time_step: 
    obsparams_list = np.empty((T+1, K, M))
    pi0_list = np.empty((T+1, K))
    Ps_list = np.empty((T+1, K, K))
    acceptanceprobs = np.empty((T+1))
    posteriorcov = np.empty((T+1))

    # Run inference using the existing dataset, also returns the posterior over states which is useful for calcumating MI
    obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, _  = test_iohmm.fit(observations, inputs=initial_inputs, method="gibbs", num_iters=n_iters, burnin=burnin)

    # Store the model parameters
    obsparams_list[0] = np.mean(obsparams_sampled, axis=0)
    pi0_list[0] = np.mean(pi0_sampled, axis=0)
    Ps_list[0] = np.mean(Ps_sampled, axis=0)
    ravelled_obsparams = np.reshape(obsparams_sampled[burnin:], (obsparams_sampled[burnin:].shape[0],obsparams_sampled.shape[1]*obsparams_sampled.shape[2]))
    ravelled_Ps = np.reshape(Ps_sampled[burnin:,:,:-1], (Ps_sampled[burnin:].shape[0], (K)*(K-1)))
    params = np.hstack((ravelled_obsparams,ravelled_Ps))
    cov = np.cov(params, rowvar = False)
    posteriorcov[0] =  0.5*np.linalg.slogdet(cov)[1] 
        
    inputs = initial_inputs
    for t in range(T):
        print("Computing parameters of IO-HMM using "+str(t+1+initial_T)+" samples")
        # Select next stimuli from a list of possible input stimuli
        x_new = selected_inputs[t]
        # Obtain output from the true model
        z_new, observation_new = true_iohmm.sample(T=1, input = np.reshape(np.array(x_new), (1,M)), prefix=(zs[0], observations[0]))
        # Append this to the list of inputs and outputs
        observations[0] = np.concatenate((observations[0], observation_new), axis=0)
        inputs[0] = np.concatenate((inputs[0],np.reshape(np.array(x_new), (1,M))),  axis=0)
        zs[0] = np.concatenate((zs[0], z_new))

        # Run inference using the now new dataset
        initialize_w = np.reshape(obsparams_list[t], (K, 1, M))
        obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method="gibbs", num_iters=n_iters, burnin=burnin)
        
        # Permute
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
        ravelled_obsparams = np.reshape(obsparams_sampled[burnin:], (obsparams_sampled[burnin:].shape[0],obsparams_sampled.shape[1]*obsparams_sampled.shape[2]))
        ravelled_Ps = np.reshape(Ps_sampled[burnin:,:,:-1], (Ps_sampled[burnin:].shape[0], (K)*(K-1)))
        params = np.hstack((ravelled_obsparams,ravelled_Ps))
        cov = np.cov(params, rowvar = False)
        posteriorcov[t+1] =  0.5*np.linalg.slogdet(cov)[1]  

    return pi0_list, Ps_list, obsparams_list, posteriorcov



