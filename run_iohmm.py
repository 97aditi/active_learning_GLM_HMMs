import ssm
from ssm.util import one_hot, find_permutation
import numpy as np
from iohmm_infomax import iohmm_infomax_gibbs, iohmm_infomax_VI
from iohmm_random import iohmm_random_gibbs
import argparse
import multiprocessing as mp
import os

USE_CLUSTER = True

# print cpu core count
print("Number of cpu cores:", mp.cpu_count())

# supressing warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run IOHMM experiments')
parser.add_argument('--seed', type=int, default='1',
                    help='Enter random seed')
parser.add_argument('--input_selection', type=str, default='infomax_gibbs',
                    help='choose one of infomax_gibbs/infomax_VI/random')
parser.add_argument('--fitting_method', type=str, default='gibbs',
                    help='choose one of gibbs/gibbs_parallel/gibbs_PG')     
parser.add_argument('--num_gibbs_samples', type=int, default='400')
parser.add_argument('--num_gibbs_burnin', type=int, default='100')

args = parser.parse_args()

if USE_CLUSTER:
    # load cluster array
    cluster_array = np.load('cluster_array.npy')
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    seed = int(cluster_array[index][1])
    np.random.seed(seed)
    num_gibbs_samples = int(cluster_array[index][0])
    num_trials_per_sess = 1000
else:
    seed = args.seed
    np.random.seed(seed)
    num_gibbs_samples = args.num_gibbs_samples
    num_trials_per_sess = 100

# Set the parameters of the GLM-HMM
num_states = 3   # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions

# Make a GLM-HMM which will be our data generator
true_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.02, 0.03, 0.94]]]))
true_iohmm.observations.params = gen_weights
true_iohmm.transitions.params = gen_log_trans_mat
gen_trans_mat = np.exp(gen_log_trans_mat)[0]

num_sess = 1 # number of example sessions
initial_trials = 100
initial_inputs = np.ones((num_sess, initial_trials, input_dim)) # initialize inpts array
stim_vals = np.arange(-5,5,step=0.01).tolist() # Stimuli values 
initial_inputs[:,:,0] = np.random.choice(stim_vals, (num_sess, initial_trials)) # generate random sequence of potential inputs
initial_inputs = list(initial_inputs) #convert inpts to correct format

stimuli_list = np.ones((len(stim_vals), input_dim))
stimuli_list[:,0] = stim_vals # list of all potential inputs

# Generate a sequence of latents and choices for each session
true_latents, true_choices = [], []
for sess in range(num_sess):
    true_z, true_y = true_iohmm.sample(initial_trials, input=initial_inputs[sess])
    true_latents.append(true_z)
    true_choices.append(true_y)

#Create a new test iohmm
test_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                       observation_kwargs=dict(C=num_categories), transitions="standard")
# Reshaping weights to compute error later
true_weights = np.reshape(gen_weights, (num_states, input_dim))


method = args.fitting_method
# Train iohmm using active learning
if  args.input_selection == 'infomax_gibbs':
    if method=='gibbs':
        pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_infomax_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs", num_iters = num_gibbs_samples, burnin = args.num_gibbs_burnin)
    elif method=='gibbs_PG':
       pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_infomax_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs", polyagamma=True, num_iters = num_gibbs_samples, burnin = args.num_gibbs_burnin)
    elif method=='gibbs_parallel':
        pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_infomax_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs_parallel")
    error_weights = np.linalg.norm(np.linalg.norm(weights_list-true_weights, axis=1), axis=1)
    error_Ps = np.linalg.norm(np.linalg.norm(Ps_list-gen_trans_mat, axis=1), axis=1)
    np.save("Results_IOHMM/infomax_"+method+"_weights_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", weights_list)
    np.save("Results_IOHMM/infomax_"+method+"_errorinweights_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", error_weights)
    np.save("Results_IOHMM/infomax_"+method+"_Ps_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", Ps_list)
    np.save("Results_IOHMM/infomax_"+method+"_errorinPs_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", error_Ps)
    np.save("Results_IOHMM/infomax_"+method+"_posteriorcovariance_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", post_cov)
    np.save("Results_IOHMM/infomax_"+method+"_selectedinputs_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", selected_inputs)

# Train iohmm using active learning
if  args.input_selection == 'infomax_VI':
    pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_infomax_VI(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list)
    error_weights = np.linalg.norm(np.linalg.norm(weights_list-true_weights, axis=1), axis=1)
    error_Ps = np.linalg.norm(np.linalg.norm(Ps_list-gen_trans_mat, axis=1), axis=1)
    np.save("Results_IOHMM/infomax_"+method+"_weights_atseed"+str(seed)+".npy", weights_list)
    np.save("Results_IOHMM/infomax_"+method+"_errorinweights_atseed"+str(seed)+".npy", error_weights)
    np.save("Results_IOHMM/infomax_"+method+"_Ps_atseed"+str(seed)+".npy", Ps_list)
    np.save("Results_IOHMM/infomax_"+method+"_errorinPs_atseed"+str(seed)+".npy", error_Ps)
    np.save("Results_IOHMM/infomax_"+method+"_posteriorcovariance_atseed"+str(seed)+".npy", post_cov)
    np.save("Results_IOHMM/infomax_"+method+"_selectedinputs_atseed"+str(seed)+".npy", selected_inputs)

if args.input_selection == 'random':
# Train iohmm using random sampling
    if method=='gibbs':
        pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_random_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs", num_iters = num_gibbs_samples, burnin = args.num_gibbs_burnin)
    elif method=='gibbs_PG':
       pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_random_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs", polyagamma=True, num_iters = num_gibbs_samples, burnin = args.num_gibbs_burnin)
    elif method=='gibbs_parallel':
        pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_random_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs_parallel")
    error_weights = np.linalg.norm(np.linalg.norm(weights_list-true_weights, axis=1), axis=1)
    error_Ps = np.linalg.norm(np.linalg.norm(Ps_list-gen_trans_mat, axis=1), axis=1)
    np.save("Results_IOHMM/random_"+method+"_weights_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", weights_list)
    np.save("Results_IOHMM/random_"+method+"_errorinweights_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", error_weights)
    np.save("Results_IOHMM/random_"+method+"_Ps_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", Ps_list)
    np.save("Results_IOHMM/random_"+method+"_errorinPs_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", error_Ps)
    np.save("Results_IOHMM/random_"+method+"_posteriorcovariance_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", post_cov)
    np.save("Results_IOHMM/random_"+method+"_selectedinputs_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy", selected_inputs)

