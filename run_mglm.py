from src_mglm.mglms import MGLM 
import numpy as np 
from mglm_al import mglm_al
from mglm_random import mglm_random
import argparse


parser = argparse.ArgumentParser(description='Run MLR experiments')
parser.add_argument('--seed', type=int, default='1',
                    help='Enter random seed')

args = parser.parse_args()

# Fixing random seed
#seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
seed = args.seed
np.random.seed(seed)


# Choosing parameters
num_states = 2  # number of discrete states
obs_dim = 1  # data dimension
input_dim = 2 # input dimension
num_categories = 2 # binary output for now


## Set parameters
true_pi0 = np.array([0.6, 0.4])
true_weights = np.array([[3,-6], [3, 6]])
true_weights = np.reshape(true_weights, (num_states, num_categories-1, input_dim))
true_mglm = MGLM(K=num_states, D=obs_dim, M=input_dim, C=num_categories)
true_mglm.params = [true_pi0, true_weights]

# Number of samples
T = 2000
# Number of initial samples
initial_T = 100
# modified Gibbs or old Gibbs
modified = True
# List of possible inputs
stim_vals = np.arange(-10,10,step=0.01).tolist()
input_list = np.ones((len(stim_vals), input_dim))
input_list[:,0] = stim_vals

# Initial inputs
initial_inputs = np.ones((initial_T, input_dim)) # initialize inpts array
initial_inputs[:,0] = np.random.choice(stim_vals, initial_T) # generate random sequence of input

# Sample observations from true mixture of GLMs
zs, observations = true_mglm.sample_y(initial_T, initial_inputs)


# Train MGLM  using Active Sampling----------------------------------------------------------------------------------------------------------------
method = 'Active'
#Test MGLM
test_mglm = MGLM(K=num_states, D=obs_dim, M=input_dim, C=num_categories)
true_weights = np.reshape(true_weights, (num_states, input_dim))

pis_list, weights_list, selected_inputs, posteriorcov = mglm_al(seed, T, initial_inputs, num_states, true_mglm, test_mglm, input_list, burnin = 150, n_iters=300)
np.save("Results_MGLM/" + str(input_dim) + "dActive_atseed"+str(seed) + "_posteriorcov.npy", posteriorcov)
np.save("Results_MGLM/" + str(input_dim) + "dActive_atseed"+str(seed) + "_selectedinputs.npy", selected_inputs)
np.save("Results_MGLM/" + str(input_dim) + "dActive_atseed"+str(seed) + "_weights.npy", weights_list)
np.save("Results_MGLM/" + str(input_dim) + "dActive_atseed"+str(seed) + "_pis.npy", pis_list)


# Train MGLM with random sampling-------------------------------------------------------------------------------------------------------------
method = 'Random'
#Create a test MGLM
test_mglm = MGLM(K=num_states, D=obs_dim, M=input_dim, C=num_categories)


pis_list, weights_list, selected_inputs, posteriorcovs = mglm_random(seed, T, initial_inputs, num_states, true_mglm, test_mglm, input_list, burnin = 150, n_iters=300)
np.save("Results_MGLM/" + str(input_dim) + "dRandom_atseed"+str(seed) + "_posteriorcov.npy", posteriorcov)
np.save("Results_MGLM/" + str(input_dim) + "dRandom_atseed"+str(seed) + "_selectedinputs.npy", selected_inputs)
np.save("Results_MGLM/" + str(input_dim) + "dRandom_atseed"+str(seed) + "_weights.npy", weights_list)
np.save("Results_MGLM/" + str(input_dim) + "dRandom_atseed"+str(seed) + "_pis.npy", pis_list)

