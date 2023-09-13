import ssm
from ssm.util import one_hot, find_permutation
import numpy as np
import argparse
from utils_mglm.selectbestinput import selectbestinput
from src_mglm.mglms import MGLM

parser = argparse.ArgumentParser(description='Run MLR experiments')
parser.add_argument('--seed', type=int, default='1',
                    help='Enter random seed')

args = parser.parse_args()

# Fixing random seed
#seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
seed = args.seed
np.random.seed(seed)


# Set the parameters of the data-generating GLM-HMM
num_states = 3   # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions

# Make an IO-HMM
true_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.02, 0.03, 0.94]]]))
true_iohmm.observations.params = gen_weights
true_iohmm.transitions.params = gen_log_trans_mat
gen_trans_mat = np.exp(gen_log_trans_mat)[0]


# fit a single GLM on this data using active learning 
# Choosing parameters of the single glm
num_states = 1  # number of discrete states
glm = MGLM(K=num_states, D=obs_dim, M=input_dim, C=num_categories)


# List of possible inputs
stim_vals = np.arange(-5,5,step=0.01).tolist()
input_list = np.ones((len(stim_vals), input_dim))
input_list[:,0] = stim_vals

initial_T = 100
# Initial inputs
initial_inputs_array = np.ones((initial_T, input_dim)) # initialize inpts array
initial_inputs_array[:,0] = np.random.choice(stim_vals, initial_T) # generate random sequence osf stimuli
initial_inputs = []
initial_inputs.append(initial_inputs_array)

# Sample observations from true mixture of GLMs
latents, obs = true_iohmm.sample(initial_T, input = np.reshape(initial_inputs, (initial_T, input_dim)))
# Correct format
observations = []
observations.append(obs)
zs = []
zs.append(latents)

# Time steps
T = 2
# Things to store for the single GLM
weights_list = np.empty((T+1, num_states, input_dim))
posteriorcov = np.empty((T+1))

print("Model mismatch analysis for IO-HMMs; using a single GLM for selecting inputs and later fitting using Gibbs sampling")

# Train the single GLM 
weights_sampled, pis_sampled, ll = glm.fit_gibbs(observations[0], initial_inputs[0], burnin=100, n_iters=400)
weights_list[0] = np.reshape(np.mean(weights_sampled, axis=0), (1, num_states, input_dim))
posteriorcov[0] = np.linalg.slogdet(np.cov(weights_sampled[:,0], rowvar=False))[1] 
inputs = initial_inputs
for t in range(T):
    print("Selecting input at trial #"+str(t+1))
    x_new, _ = selectbestinput(glm, input_list, np.reshape(weights_sampled, (weights_sampled.shape[0], num_states, input_dim)), pis_sampled)
    # Obtain output from the true model
    z_new, observation_new = true_iohmm.sample(T=1, input = np.reshape(np.array(x_new), (1,input_dim)), prefix=(zs[0], observations[0]))
    # Append this to the list of inputs and outputs
    observations[0] = np.concatenate((observations[0], observation_new), axis=0)
    inputs[0] = np.concatenate((inputs[0],np.reshape(np.array(x_new), (1,input_dim))),  axis=0)
    zs[0] = np.concatenate((zs[0], z_new))


    weights_sampled, pis_sampled, ll = glm.fit_gibbs(observations[0], inputs[0], burnin=100, n_iters=400)
    weights_list[t+1] = np.mean(weights_sampled, axis=0)
    posteriorcov[t+1] = np.linalg.slogdet(np.cov(weights_sampled[:,0], rowvar=False))[1] 

np.save("Results_modelmismatch/GLMweights_atseed_" + str(seed)+".npy", weights_list)
np.save("Results_modelmismatch/GLMposteriorentropy_atseed_" + str(seed)+".npy", posteriorcov)
np.save("Results_modelmismatch/selectedinputs_atseed_" + str(seed)+".npy", inputs[0])
np.save("Results_modelmismatch/observations_atseed_" + str(seed)+".npy", observations[0])


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Now use the selected inputs to train an iohmm

# Set the parameters of the  new IO-HMM
num_states = 3   # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions

# Make an IO-HMM
test_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")

selected_inputs = inputs[0]
# Training it using the selected inputs
def iohmm_al(seed, T, K, true_iohmm, test_iohmm, burnin = 100, n_iters=300):
    """ Bayesian active learning for fitting the model"""
    print("Fitting IO-HMM using active learning using inputs selected by GLM")
    # Fixing random seed
    np.random.seed(seed)
    M = 2
    # Observations for initial samples
    initial_inputs_array = selected_inputs[:initial_T]
    initial_inputs = []
    initial_inputs.append(initial_inputs_array)

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
        x_new = selected_inputs[t+initial_T]
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

method = 'Active'
pi0_list, Ps_list, obsparams_list, posteriorcov = iohmm_al(seed, T , num_states, true_iohmm, test_iohmm)
np.save("Results_modelmismatch/"+method+"_weights_atseed"+str(seed)+".npy", obsparams_list)
np.save("Results_modelmismatch/"+method+"_Ps_atseed"+str(seed)+".npy", Ps_list)
np.save("Results_modelmismatch/"+method+"_posteriorcovariance_atseed"+str(seed)+".npy", posteriorcov)
