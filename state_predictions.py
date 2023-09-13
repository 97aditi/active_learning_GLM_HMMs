import numpy as np
import ssm
from ssm.messages import *
import matplotlib.pyplot as plt
import argparse

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure tit
plt.rc('lines', linewidth=2)
cols = ['#ff7f00', '#4daf4a', '#377eb8']
plt.rcParams['font.sans-serif'] = "Helvetica"

parser = argparse.ArgumentParser(description='infer states using randomly and actively trained IO-HMMs')
parser.add_argument('--seed', type=int, default='2',
                    help='Enter random seed')
parser.add_argument('--trial', type=int, default='400',
                    help='choose number of trials to train the IO-HMMs for')
                 
args = parser.parse_args()

np.random.seed(args.seed)
trial = args.trial

# Set the parameters of the IO-HMM
num_states = 3  # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions


# This is the true IO-HMM
true_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.02, 0.03, 0.94]]]))
true_iohmm.observations.params = gen_weights
true_iohmm.transitions.params = gen_log_trans_mat
gen_trans_mat = np.exp(gen_log_trans_mat)[0]


# Generate some data from this true IO-HMM and infer states for these
num_sess = 1 # number of example sessions
num_trials_per_sess = 100 # number of trials in a session
inputs = np.ones((num_sess, num_trials_per_sess, input_dim)) # initialize inpts array
stim_vals = np.arange(-4,4,step=0.01).tolist()
inputs[:,:,0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess)) # generate random sequence of stimuli
inputs = list(inputs) # convert inpts to correct format

# Generate a sequence of latents and choices for each session
true_latents, true_choices = [], []

for sess in range(num_sess):
    true_z, true_y = true_iohmm.sample(num_trials_per_sess, input=inputs[sess])
    true_latents.append(true_z)																																																																									
    true_choices.append(true_y)

random_pzts = 0
active_pzts = 0
for seed in range(5):
    # Making IO-HMM's using parameters learned by active and random inference methods
    active_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                    observation_kwargs=dict(C=num_categories), transitions="standard")

    random_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                    observation_kwargs=dict(C=num_categories), transitions="standard")

    # Load stored weights
    method = 'infomax_gibbs'
    weights_list = np.load("Results_IOHMM/"+method+"_weights_atseed"+str(seed)+".npy")
    Ps_list = np.load("Results_IOHMM/"+method+"_Ps_atseed"+str(seed)+".npy") 
    active_weights = weights_list[trial]
    active_trans_mat = Ps_list[trial]
    active_iohmm.observations.params = active_weights.reshape((num_states, num_categories-1, input_dim))
    active_iohmm.transitions.params = (np.log(active_trans_mat)).reshape((1, num_states, num_states))

    # Load stored weights
    method = 'random_gibbs'
    weights_list = np.load("Results_IOHMM/"+method+"_weights_atseed"+str(seed)+".npy")
    Ps_list = np.load("Results_IOHMM/"+method+"_Ps_atseed"+str(seed)+".npy") 
    random_weights = weights_list[trial]
    random_trans_mat = Ps_list[trial]
    random_iohmm.observations.params = random_weights.reshape((num_states, num_categories-1, input_dim))
    random_iohmm.transitions.params = (np.log(random_trans_mat)).reshape((1, num_states, num_states))

    # Now run forward backward for these IO-HMMs and obtain posterior probabilites over states 
    # First obtain log likelihoods
    Ts = [data.shape[0] for data in true_choices]
    log_Ls = [active_iohmm.observations.log_likelihoods(true_choices[t], inputs[t]) for t in range(num_sess)]
    pi0 = active_iohmm.init_state_distn.initial_state_distn
    Ps = (active_iohmm.transitions.transition_matrix).reshape((1, num_states, num_states))
    # Let's do FB
    alphas = [np.zeros((T, num_states)) for T in Ts]
    [forward_pass(pi0, Ps, log_Ls[sess], alphas[sess]) for sess in range(len(Ts))] 
    betas = [np.zeros((T, num_states)) for T in Ts]
    [backward_pass(Ps, log_Ls[sess], betas[sess]) for sess in range(len(Ts))]
    # Now obtain state posterior probabilites
    log_pzts = active_iohmm.posterior_over_states_using_all_samples(alphas, betas)
    active_pzts = active_pzts + np.exp(log_pzts)

    # Repeat the same for random IO-HMM
    Ts = [data.shape[0] for data in true_choices]
    log_Ls = [random_iohmm.observations.log_likelihoods(true_choices[t], inputs[t]) for t in range(num_sess)]
    pi0 = random_iohmm.init_state_distn.initial_state_distn
    Ps = (random_iohmm.transitions.transition_matrix).reshape((1, num_states, num_states))
    # Let's do FB
    alphas = [np.zeros((T, num_states)) for T in Ts]
    [forward_pass(pi0, Ps, log_Ls[sess], alphas[sess]) for sess in range(len(Ts))] 
    betas = [np.zeros((T, num_states)) for T in Ts]
    [backward_pass(Ps, log_Ls[sess], betas[sess]) for sess in range(len(Ts))]
    # Now obtain state posterior probabilites
    log_pzts = random_iohmm.posterior_over_states_using_all_samples(alphas, betas)
    random_pzts = random_pzts + np.exp(log_pzts)

random_pzts = random_pzts/5
active_pzts = active_pzts/5

# Obtaining probabilities of true latents
true_probs = np.zeros((len(true_latents[0]), num_states))
ind_for_state1 = np.argwhere(true_latents[0]==0)
ind_for_state2 = np.argwhere(true_latents[0]==1)
ind_for_state3 = np.argwhere(true_latents[0]==2)
true_probs[ind_for_state1,0] = 1
true_probs[ind_for_state2,1] = 1
true_probs[ind_for_state3,2] = 1

trials = np.arange(num_trials_per_sess)+1
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_facecolor('0.9')
ax1.set_ylim(-0.05,1.05)
ax1.set_yticks([0,1])
ax1.set_xticks([])
ax1.set_title("true states from the generative model")

for i in range(num_states):
    ax1.plot(trials, true_probs[:,i])
    ax2.plot(trials, active_pzts[0][:,i])
    ax3.plot(trials, random_pzts[0][:,i])  
ax2.set_ylim(0,1)
ax2.set_xticks([])
ax2.set_yticks([0,1])
ax2.set_title("IO-HMM trained using active learning")
ax2.set_ylabel("P(state|data, inputs)")
ax2.set_facecolor('0.9')

ax3.set_ylim(0,1)
ax3.set_yticks([0,1])
ax3.set_title("IO-HMM trained using random sampling")
ax3.set_xlabel("trial #")
ax3.set_facecolor('0.9')


plt.tight_layout()
# plt.savefig("Figs_IOHMM/state_prediction_seed"+str(args.seed)+".png", dpi=400)
# plt.savefig("Figs_IOHMM/state_prediction_seed"+str(args.seed)+".svg")
plt.show()



