from functools import partial
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from autograd.scipy.special import logsumexp

from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from ssm.primitives import hmm_normalizer
from ssm.messages import hmm_expected_states, hmm_filter, hmm_sample, viterbi, backward_pass, forward_pass
from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, \
    replicate, collapse, ssm_pbar

import ssm.observations as obs
import ssm.transitions as trans
import ssm.init_state_distns as isd
import ssm.hierarchical as hier
import ssm.emissions as emssn
from ssm.util import one_hot, find_permutation
from scipy.special import psi
import scipy.stats as st
import time
__all__ = ['HMM', 'HSMM']

from joblib import Parallel, delayed


class HMM(object):
    """
    Base class for hidden Markov models.

    Notation:
    K: number of discrete latent states
    D: dimensionality of observations
    M: dimensionality of inputs

    In the code we will sometimes refer to the discrete
    latent state sequence as z and the data as x.
    """
    def __init__(self, K, D, M=0, init_state_distn=None,
                 transitions='standard',
                 transition_kwargs=None,
                 hierarchical_transition_tags=None,
                 observations="gaussian", observation_kwargs=None,
                 hierarchical_observation_tags=None, **kwargs):

        # Make the initial state distribution
        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        if not isinstance(init_state_distn, isd.InitialStateDistribution):
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.InitialStateDistribution.")

        # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions,
            constrained=trans.ConstrainedStationaryTransitions,
            sticky=trans.StickyTransitions,
            inputdriven=trans.InputDrivenTransitions,
            recurrent=trans.RecurrentTransitions,
            recurrent_only=trans.RecurrentOnlyTransitions,
            rbf_recurrent=trans.RBFRecurrentTransitions,
            nn_recurrent=trans.NeuralNetworkRecurrentTransitions
            )

        if isinstance(transitions, str):
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = \
                hier.HierarchicalTransitions(transition_classes[transitions], K, D, M=M,
                                        tags=hierarchical_transition_tags,
                                        **transition_kwargs) \
                if hierarchical_transition_tags is not None \
                else transition_classes[transitions](K, D, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # This is the master list of observation classes.
        # When you create a new observation class, add it here.
        observation_classes = dict(
            gaussian=obs.GaussianObservations,
            diagonal_gaussian=obs.DiagonalGaussianObservations,
            studentst=obs.MultivariateStudentsTObservations,
            t=obs.MultivariateStudentsTObservations,
            diagonal_t=obs.StudentsTObservations,
            diagonal_studentst=obs.StudentsTObservations,
            exponential=obs.ExponentialObservations,
            bernoulli=obs.BernoulliObservations,
            categorical=obs.CategoricalObservations,
            input_driven_obs=obs.InputDrivenObservations,
            poisson=obs.PoissonObservations,
            vonmises=obs.VonMisesObservations,
            ar=obs.AutoRegressiveObservations,
            autoregressive=obs.AutoRegressiveObservations,
            no_input_ar=obs.AutoRegressiveObservationsNoInput,
            diagonal_ar=obs.AutoRegressiveDiagonalNoiseObservations,
            diagonal_autoregressive=obs.AutoRegressiveDiagonalNoiseObservations,
            independent_ar=obs.IndependentAutoRegressiveObservations,
            robust_ar=obs.RobustAutoRegressiveObservations,
            no_input_robust_ar=obs.RobustAutoRegressiveObservationsNoInput,
            robust_autoregressive=obs.RobustAutoRegressiveObservations,
            diagonal_robust_ar=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_robust_autoregressive=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            )

        if isinstance(observations, str):
            observations = observations.lower()
            if observations not in observation_classes:
                raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(observation_classes.keys())))

            observation_kwargs = observation_kwargs or {}
            observations = \
                hier.HierarchicalObservations(observation_classes[observations], K, D, M=M,
                                        tags=hierarchical_observation_tags,
                                        **observation_kwargs) \
                if hierarchical_observation_tags is not None \
                else observation_classes[observations](K, D, M=M, **observation_kwargs)
        if not isinstance(observations, obs.Observations):
            raise TypeError("'observations' must be a subclass of"
                            " ssm.observations.Observations")

        self.K, self.D, self.M = K, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.observations = observations

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.observations.params

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.observations.params = value[2]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        """
        Initialize parameters given data.
        """
        self.init_state_distn.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.transitions.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.observations.initialize(datas, inputs=inputs, masks=masks, tags=tags, init_method=init_method)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        T : int
            number of time steps to sample

        prefix : (zpre, xpre)
            Optional prefix of discrete states (zpre) and continuous states (xpre)
            zpre must be an array of integers taking values 0...num_states-1.
            xpre must be an array of the same length that has preceding observations.

        input : (T, input_dim) array_like
            Optional inputs to specify for sampling

        tag : object
            Optional tag indicating which "type" of sampled data

        with_noise : bool
            Whether or not to sample data with noise.

        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states

        x_sample : (T x observation_dim) array_like
            Array of sampled data
        """
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        assert T > 0

        # Check the inputs
        if input is not None:
            assert input.shape == (T,) + M

        # Get the type of the observations
        if isinstance(self.observations, obs.InputDrivenObservations):
            dtype = int
        else:
            dummy_data = self.observations.sample_x(0, np.empty(0, ) + D)
            dtype = dummy_data.dtype

        # fit the data array
        if prefix is None:
            # No prefix is given.  Sample the initial state as the prefix.
            pad = 1
            z = np.zeros(T, dtype=int)
            data = np.zeros((T,) + D, dtype=dtype)
            input = np.zeros((T,) + M) if input is None else input
            mask = np.ones((T,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observations.sample_x(z[0], data[:0], input=input[0], with_noise=with_noise)

            # We only need to sample T-1 datapoints now
            T = T - 1

        else:
            # Check that the prefix is of the right type
            zpre, xpre = prefix
            pad = len(zpre)
            assert zpre.dtype == int and zpre.min() >= 0 and zpre.max() < K
            assert xpre.shape == (pad,) + D

            # Construct the states, data, inputs, and mask arrays
            z = np.concatenate((zpre, np.zeros(T, dtype=int)))
            data = np.concatenate((xpre, np.zeros((T,) + D, dtype)))
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            mask = np.ones((T+pad,) + D, dtype=bool)

        # Fill in the rest of the data
        for t in range(pad, pad+T):
            Pt = self.transitions.transition_matrices(data[t-1:t+1], input[t-1:t+1], mask=mask[t-1:t+1], tag=tag)[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            data[t] = self.observations.sample_x(z[t], data[:t], input=input[t], tag=tag,
                                                 with_noise=with_noise)

        # Return the whole data if no prefix is given.
        # Otherwise, just return the simulated part.
        if prefix is None:
            return z, data
        else:
            return z[pad:], data[pad:]

    @ensure_args_not_none
    def expected_states(self, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return hmm_expected_states(pi0, Ps, log_likes)

    @ensure_args_not_none
    def most_likely_states(self, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return viterbi(pi0, Ps, log_likes)

    @ensure_args_not_none
    def filter(self, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return hmm_filter(pi0, Ps, log_likes)

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(data, input, mask)
        return self.observations.smooth(Ez, data, input, tag)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.observations.log_prior()

    @ensure_args_are_lists
    def log_likelihood(self, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ll = 0
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            pi0 = self.init_state_distn.initial_state_distn
            Ps = self.transitions.transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)
            ll += hmm_normalizer(pi0, Ps, log_likes)
            assert np.isfinite(ll)
        return ll

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        return self.log_likelihood(datas, inputs, masks, tags) + self.log_prior()

    def expected_log_likelihood(
            self, expectations, datas, inputs=None, masks=None, tags=None):
        """
        Compute log-likelihood given current model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ell = 0.0
        for (Ez, Ezzp1, _), data, input, mask, tag in \
                zip(expectations, datas, inputs, masks, tags):

            pi0 = self.init_state_distn.initial_state_distn
            log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)

            ell += np.sum(Ez[0] * np.log(pi0))
            ell += np.sum(Ezzp1 * log_Ps)
            ell += np.sum(Ez * log_likes)
            assert np.isfinite(ell)

        return ell

    def expected_log_probability(
            self, expectations, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log-probability of the data given current
        model parameters.
        """
        ell = self.expected_log_likelihood(
            expectations, datas, inputs=inputs, masks=masks, tags=tags)
        return ell + self.log_prior()


    def samplestates(self, pi0, Ps, logL):
        # Find the equivalent function for bwdmessages
        T, K = logL.shape
        betas = np.zeros((T, K))
        backward_pass(Ps, logL, betas)
        # This computes log of backward probabilities
        betas = betas - logsumexp(betas)
        # B = np.exp(betas)

        Ps = np.reshape(Ps, (K,K))
        zs = []
        # Sample first latent state
        logprobs = np.log(pi0) + betas[0,:] + logL[0,:]
        # Normalizes
        logprobs = logprobs - logsumexp(logprobs)
        probs = np.exp(logprobs)
        zs.append(np.random.multinomial(n=1, pvals=probs).argmax())
        for t in np.arange(start=1, stop=T, step=1):
            logprobs = np.log(Ps[zs[t-1],:]) + betas[t,:] +logL[t,:]
            # Normalize
            logprobs = logprobs - logsumexp(logprobs)
            probs = np.exp(logprobs)
            zs.append(np.random.multinomial(n=1, pvals=probs).argmax())
        return zs

    def predictive_distribution_over_states(self, qzts):
        """ Computing p(z_{t+1}=k \mid D_t)  """
        # p(z_{t+1}=k \mid D_t) = \sum_j P_jk p(z_t=j \md D_t)
        # Approximating p(z_t=k \mid D_t) using the variational distribution: qzts
        # Using a one sample estimate for P_jk
        Ps = np.exp(self.transitions.log_Ps)
        predictive_probs = Ps.T@qzts[-1].reshape((self.K,1))
        return predictive_probs
        

    def posterior_over_states_using_past_samples(self, pi0, Ps, logL):
        # We want p(z_t | x_{1:t}, y_{1:t}, \theta) for calculating MI later
        T, K = logL.shape
        alphas = np.zeros((T, K))
        forward_pass(pi0, Ps, logL, alphas)
        logpzt = alphas[-1] - logsumexp(alphas[-1])
        pzt = np.exp(logpzt) 
        return pzt


    def posterior_over_states_using_all_samples(self, alphas, betas):
        """ Computes p(z_t = k) for all t\in[1,T] and all k"""
        Ts = [alpha.shape[0] for alpha in alphas]
        log_pzts = [alphas[sess] + betas[sess] for sess in range(len(Ts))]
        normalizer = [logsumexp(alphas[sess][-1]) for sess in range(len(Ts))]
        log_pzts = [log_pzts[sess] - normalizer[sess] for sess in range(len(Ts))]
        return log_pzts

    def transitions_over_states_using_all_samples(self, alphas, betas, log_Ps, logL):
        K = log_Ps.shape[0]
        Ts = [alpha.shape[0] for alpha in alphas]

        log_joint_pzts = [np.empty((T-1,K,K)) for T in Ts]
        normalizer = [logsumexp(alphas[sess][-1]) for sess in range(len(Ts))]

        # Parallelize in two parts
        for sess in range(len(Ts)):
            alphas_plus = np.vstack((np.zeros((1,K)), alphas[sess]))
            betas_plus = np.vstack((betas[sess], np.zeros((1,K))))
            logL_plus =np.vstack((logL[sess], np.zeros((1,K))) )
            intermediate = alphas_plus.reshape((Ts[0]+1,K,1)) + (betas_plus + logL_plus).reshape((Ts[0]+1,1,K))
            log_joint_pzts[sess] =  intermediate[1:Ts[0]] +  log_Ps.reshape((1,K,K))

        # Normalize now
        log_joint_pzts = [(log_joint_pzts[sess] - normalizer[sess]) for sess in range(len(Ts))]
        return log_joint_pzts


    # Model fitting
    def _fit_sgd(self, optimizer, datas, inputs, masks, tags, verbose = 2, num_iters=1000, **kwargs):
        """
        Fit the model with maximum marginal likelihood.
        """
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = self.log_probability(datas, inputs, masks, tags)
            return -obj / T

        # Set up the progress bar
        lls  = [-_objective(self.params, 0) * T]
        pbar = ssm_pbar(num_iters, verbose, "Epoch {} Itr {} LP: {:.1f}", [0, 0, lls[-1]])

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            self.params, val, g, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
            lls.append(-val * T)
            if verbose == 2:
              pbar.set_description("LP: {:.1f}".format(lls[-1]))
              pbar.update(1)
        return lls


    def _fit_stochastic_em(self, optimizer, datas, inputs, masks, tags, verbose = 2, num_epochs=100, **kwargs):
        """
        Replace the M-step of EM with a stochastic gradient update using the ELBO computed
        on a minibatch of data.
        """
        M = len(datas)
        T = sum([data.shape[0] for data in datas])

        # A helper to grab a minibatch of data
        perm = [np.random.permutation(M) for _ in range(num_epochs)]
        def _get_minibatch(itr):
            epoch = itr // M
            m = itr % M
            i = perm[epoch][m]
            return datas[i], inputs[i], masks[i], tags[i][i]

        # Define the objective (negative ELBO)
        def _objective(params, itr):
            # Grab a minibatch of data
            data, input, mask, tag = _get_minibatch(itr)
            Ti = data.shape[0]

            # E step: compute expected latent states with current parameters
            Ez, Ezzp1, _ = self.expected_states(data, input, mask, tag)

            # M step: set the parameter and compute the (normalized) objective function
            self.params = params
            pi0 = self.init_state_distn.initial_state_distn
            log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)

            # Compute the expected log probability
            # (Scale by number of length of this minibatch.)
            obj = self.log_prior()
            obj += np.sum(Ez[0] * np.log(pi0)) * M
            obj += np.sum(Ezzp1 * log_Ps) * (T - M) / (Ti - 1)
            obj += np.sum(Ez * log_likes) * T / Ti
            assert np.isfinite(obj)

            return -obj / T

        # Set up the progress bar
        lls  = [-_objective(self.params, 0) * T]
        pbar = ssm_pbar(num_epochs * M, verbose, "Epoch {} Itr {} LP: {:.1f}", [0, 0, lls[-1]])

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            self.params, val, _, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
            epoch = itr // M
            m = itr % M
            lls.append(-val * T)
            if verbose == 2:
              pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(epoch, m, lls[-1]))
              pbar.update(1)

        return lls

    def _fit_em(self, datas, inputs, masks, tags, verbose = 2, num_iters=100, tolerance=0,
                init_state_mstep_kwargs={},
                transitions_mstep_kwargs={},
                observations_mstep_kwargs={},
                **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        lls  = [self.log_probability(datas, inputs, masks, tags)]

        pbar = ssm_pbar(num_iters, verbose, "LP: {:.1f}", [lls[-1]])

        for itr in pbar:
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data, input, mask, tag)
                            for data, input, mask, tag,
                            in zip(datas, inputs, masks, tags)]

            # M step: maximize expected log joint wrt parameters
            self.init_state_distn.m_step(expectations, datas, inputs, masks, tags, **init_state_mstep_kwargs)
            self.transitions.m_step(expectations, datas, inputs, masks, tags, **transitions_mstep_kwargs)
            self.observations.m_step(expectations, datas, inputs, masks, tags, **observations_mstep_kwargs)

            # Store progress
            lls.append(self.log_prior() + sum([ll for (_, _, ll) in expectations]))

            if verbose == 2:
              pbar.set_description("LP: {:.1f}".format(lls[-1]))

            # Check for convergence
            if itr > 0 and abs(lls[-1] - lls[-2]) < tolerance:
                if verbose == 2:
                  pbar.set_description("Converged to LP: {:.1f}".format(lls[-1]))
                break

        return lls

    def _fit_gibbs(self, datas, inputs, masks=None, tags=None, verbose = 2, num_iters=500, burnin=200, polyagamma = False, **kwargs):
        """
            Perform gibbs sampling to approximate the posterior (current implementation supports only bernoulli GLM)
        """
        Ts = [data.shape[0] for data in datas]
        K = self.K
        M = self.M        

        # Initial assignments
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrix
        obsparams = self.observations.params
        zs = [npr.choice(self.K, size=T) for T in Ts]

        # Lists to store all samples and Log-likelihood
        obsparams_sampled = np.zeros((num_iters+burnin+1,K, M))
        Ps_sampled = np.zeros((num_iters+burnin+1,K, K))
        pi0_sampled = np.empty((num_iters+burnin+1, K))
        pzts_persample = np.empty((num_iters+burnin+1, K))
        lls = [self.log_probability(datas, inputs, masks, tags)]

        obsparams_sampled[0] = np.reshape(obsparams, (K,M))
        pi0_sampled[0] = pi0
        Ps_sampled[0] = Ps

        pbar = ssm_pbar(burnin+num_iters, verbose, "LP: {:.1f}", [lls[-1]])

        # Gibbs sampler
        for it in pbar:
            # Sample from full conditional of state assignments
            # Obtain observation Likelihoods (TXKXC)
            log_likes = [self.observations.calculate_logits(inpt) for inpt in inputs]
            # Extract only emission potentials
            log_Ls = [np.empty((T,self.K)) for T in Ts]
            for sess in range(len(Ts)):
                log_Ls[sess] = log_likes[sess][np.arange(Ts[sess]),:,datas[sess].ravel()]
           
            zs = [self.samplestates(pi0, np.reshape(Ps, (1,K,K)), log_L) for log_L in log_Ls]
            #---------------------------------------------------------------------------------------  
             # Sample from conditional of Transitions
            Ps = self.transitions.samplefromposterior(zs)
            # Sample from conditional of initial state distributions
            pi0 = self.init_state_distn.samplefromposterior(zs)
            # Sample from full conditional of observation weights
            if polyagamma == False:
                obsparams = self.observations.samplefromposterior(datas, inputs, zs)
            else:
                obsparams = self.observations.polyagammasample(datas, inputs, zs)
            #--------------------------------------------------------------------------------------
            
            # Storing progress
            obsparams_sampled[it, :] = np.reshape(obsparams, (K,M))
            pi0_sampled[it, :] = pi0
            Ps_sampled[it,:] = Ps
            lls.append(self.log_probability(datas, inputs, masks, tags))

            if verbose == 2:
              pbar.set_description("LP: {:.1f}".format(lls[-1]))
              pbar.refresh()

            # Finally, if we want to compute mutual information later for active learning, we want forward probabilities
            pzts = [self.posterior_over_states_using_past_samples(pi0, np.reshape(Ps, (1,K,K)), log_L) for log_L in log_Ls]
            # Since we are only doing a single session
            pzts_persample[it, :] = pzts[0]

        return obsparams_sampled, Ps_sampled, pi0_sampled, lls, pzts_persample


    def _fit_gibbs_parallel(self, datas, inputs, zs, masks=None, tags=None, verbose = 2, num_iters=60, burnin=40, num_chains = 5, polyagamma=False,  **kwargs):
        """
            Perform gibbs sampling with parallel chains to approximate the posterior (current implementation supports only bernoulli GLM)
        """
        # Lists to store all samples and Log-likelihood
        K = self.K
        M = self.M
        total_samples = num_iters*num_chains
        obsparams_sampled = np.zeros((total_samples,K, M))
        Ps_sampled = np.zeros((total_samples,K, K))
        pi0_sampled = np.empty((total_samples, K))
        pzts_persample = np.empty((total_samples, K))
        lls = [self.log_probability(datas, inputs, masks, tags)]

        # Running multiple chains in parallel
        all_results = Parallel(n_jobs=num_chains)(delayed(self._fit_gibbs)(datas, inputs, num_iters=num_iters, burnin=burnin, polyagamma=polyagamma) for chain in range(num_chains))
        # Concatentaing all results throwing away burn-ins
        # Also set current parameters of the model using all chains
        set_obsparams = np.zeros((K,1, self.M))
        set_pi0 = np.zeros(K)
        set_Ps = np.zeros((K,K))
        for chain in range(num_chains):
            obsparams_sampled_chain, Ps_sampled_chain, pi0_sampled_chain, lls_chain, pzts_persample_chain = all_results[chain]
            # First permute
            self.observations.Wk = np.reshape(obsparams_sampled_chain[-1], (K,1,M))
            self.init_state_distn.log_pi0 = np.log(pi0_sampled_chain[-1])
            self.transitions.log_Ps = np.log(Ps_sampled_chain[-1])
            perm = find_permutation(zs[0], self.most_likely_states(datas[0], input=inputs[0]), K, K)
            self.permute(perm)
            obsparams_sampled_chain = obsparams_sampled_chain[:,perm,:]
            Ps_sampled_chain = Ps_sampled_chain[:,perm,:]
            Ps_sampled_chain = Ps_sampled_chain[:,:,perm]   
            pi0_sampled_chain = pi0_sampled_chain[:,perm]

            # Now prepare for setting model parameters to average of the chains
            set_obsparams = set_obsparams + np.reshape(obsparams_sampled_chain[-1], (K,1,M))
            set_Ps = set_Ps + Ps_sampled_chain[-1]
            set_pi0 = set_pi0 + pi0_sampled_chain[-1]

            # Store sampled parameters
            obsparams_sampled[chain*num_iters:(chain+1)*num_iters,:,:] = obsparams_sampled_chain
            Ps_sampled[chain*num_iters:(chain+1)*num_iters] = Ps_sampled_chain
            pi0_sampled[chain*num_iters:(chain+1)*num_iters] = pi0_sampled_chain
            pzts_persample[chain*num_iters:(chain+1)*num_iters] = pzts_persample_chain
            lls = lls + lls_chain

        self.observations.Wk = set_obsparams/num_chains
        self.init_state_distn.log_pi0 = np.log(set_pi0/num_chains)
        self.transitions.log_Ps = np.log(set_Ps/num_chains)

        return obsparams_sampled, Ps_sampled, pi0_sampled, lls, pzts_persample


    

    def _fit_variational(self, datas, inputs, masks=None, tags=None, verbose = 2, tolerance=1e-4, num_iters=60,init_params = [] ,**kwargs):
        """
            Perform variational inference to approximate the posterior (current implementation supports only bernoulli GLM)
        """
        # This performs mean-field variational inference with q(\theta, \pi, A, z_{1:T}) = q(z_{1:T})q(\pi)q(A)q(\theta)

        # We assume a Dirichlet prior over each row of the transition matrix as well as the initial state probabilities
        K = self.K
        M = self.M
        Ts = [data.shape[0] for data in datas]
        # alpha_0 represents the Dirichlet parameters over the prior over initial states
        prior_alpha_0 = self.init_state_distn.prior_alpha
        # every row of alpha_A contains Dirichlet parameters governing each row of the transition matrix
        prior_alpha_Ps = self.transitions.prior_alpha
        # Assume a gaussian prior over GLM weights
        prior_means = np.zeros((K,M))
        prior_covs = np.zeros((K, M, M))
        for k in range(K):
            prior_means[k] = self.observations.prior_mean*np.zeros((M,))
            prior_covs[k] = (self.observations.prior_sigma**2)*np.eye(self.M,self.M)

        #initialize the posterior parameters
        if len(init_params)==0:
            posterior_alpha_0 = prior_alpha_0.copy()
            posterior_alpha_Ps = prior_alpha_Ps.copy()
            posterior_means = prior_means.copy()
            posterior_covs = prior_covs.copy()
        else:
            posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs = init_params

        lls = [self.log_probability(datas, inputs)]
        
        pbar = ssm_pbar(num_iters, verbose, "LP: {:.1f}", [lls[-1]])
        import time
        # Start alternating b/w distribution over states and that over model parameters
        for iter in pbar:
            time0 = time.time()
            # Extract statistics from q(z_{1:T}) and reset the model parameters accordingly so as to compute forward-backward messages using those
            pi0_tilde = np.exp(psi(posterior_alpha_0) - psi(np.sum(posterior_alpha_0)))
            Ps_tilde = np.zeros((K,K))
            for k in range(K):
                Ps_tilde[k] =np.exp(psi(posterior_alpha_Ps[k]) - psi(np.sum(posterior_alpha_Ps[k])))
            # Function under observations to compute this by drawing samples from the observation distribution
            log_Ls = self.observations.compute_expected_loglikelihood(datas, inputs, posterior_means, posterior_covs)

            # Now having obtained statistics from approximating q(z_{1:T}), let's do FB
            alphas = [np.zeros((T, K)) for T in Ts]
            reshaped_Pstilde = np.reshape(Ps_tilde, (1,K,K))
            [forward_pass(pi0_tilde, reshaped_Pstilde, log_Ls[sess], alphas[sess]) for sess in range(len(Ts))] 
            betas = [np.zeros((T, K)) for T in Ts]
            [backward_pass(reshaped_Pstilde, log_Ls[sess], betas[sess]) for sess in range(len(Ts))]
   
            # Now obtain state posterior probabilites
            log_qzts = self.posterior_over_states_using_all_samples(alphas, betas)
            log_joint_qzts = self.transitions_over_states_using_all_samples(alphas, betas, np.log(Ps_tilde), log_Ls)

            # Since we are doing just one session (TODO: handle this for multiple sessions of data)
            qzts = np.exp(log_qzts[0])
            joint_qzts = np.exp(log_joint_qzts[0])

            # Now update the distributions over model parameters using above statistics 
            posterior_alpha_0 = prior_alpha_0 + qzts[0]
            for k in range(K):
                posterior_alpha_Ps[k] = prior_alpha_Ps[k] + np.sum(joint_qzts[:,k,:], axis=0)
            #function to update observation parameters in observations.py
            # TODO: Speed this up if possible
            posterior_means, posterior_covs = self.observations.variationalupdate(qzts, datas, inputs, posterior_means)

            # Set model parameters based on the above parameters
            set_pi0 = np.random.dirichlet(posterior_alpha_0)
            self.init_state_distn.log_pi0 = np.log(set_pi0)
            set_Ps = np.empty((K,K))
            set_weights = np.empty((K,1,M))
            for k in range(K):
                set_Ps[k] = np.random.dirichlet(posterior_alpha_Ps[k])
                set_weights[k] = np.reshape(posterior_means[k], (1,M))
                # set_weights[k] = np.reshape(st.multivariate_normal.rvs(mean=posterior_means[k], cov = posterior_covs[k]), (1,M))

            self.transitions.log_Ps = np.log(set_Ps)
            self.observations.Wk = set_weights    

            # Compute log likelihood of present model
            lls.append(self.log_probability(datas, inputs))

            # # Check for convergence
            # if iter > 0 and abs(lls[-1] - lls[-2]) < tolerance:
            #     if verbose == 2:
            #       pbar.set_description("Converged to LP: {:.1f}".format(lls[-1]))
            #     break

        # To compute entropy for the next set of potenial stimuli, we need predictive probabilities for states
        # predictive_probs = self.predictive_distribution_over_states(qzts)

        return posterior_alpha_0, posterior_alpha_Ps, posterior_means, posterior_covs, lls

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None,
            verbose=2, method="em",
            initialize=True,
            init_method="random",
            **kwargs):

        _fitting_methods = \
            dict(sgd=partial(self._fit_sgd, "sgd"),
                 adam=partial(self._fit_sgd, "adam"),
                 em=self._fit_em,
                 stochastic_em=partial(self._fit_stochastic_em, "adam"),
                 stochastic_em_sgd=partial(self._fit_stochastic_em, "sgd"),
                 gibbs = self._fit_gibbs,
                 gibbs_parallel = self._fit_gibbs_parallel,
                 variational = self._fit_variational
                 )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".
                            format(method, _fitting_methods.keys()))

        if initialize:
            self.initialize(datas,
                            inputs=inputs,
                            masks=masks,
                            tags=tags,
                            init_method=init_method)

        if isinstance(self.transitions,
                      trans.ConstrainedStationaryTransitions):
            if method != "em":
                raise Exception("Only EM is implemented for constrained transitions.")

       # print(verbose)
        return _fitting_methods[method](datas,
                                        inputs=inputs,
                                        masks=masks,
                                        tags=tags,
                                        verbose=verbose,
                                        **kwargs)


class HSMM(HMM):
    """
    Hidden semi-Markov model with non-geometric duration distributions.
    The trick is to expand the state space with "super states" and "sub states"
    that effectively count duration. We rely on the transition model to
    specify a "state map," which maps the super states (1, .., K) to
    super+sub states ((1,1), ..., (1,r_1), ..., (K,1), ..., (K,r_K)).
    Here, r_k denotes the number of sub-states of state k.
    """

    def __init__(self, K, D, *, M=0, init_state_distn=None,
                 transitions="nb", transition_kwargs=None,
                 observations="gaussian", observation_kwargs=None,
                 **kwargs):

        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        if not isinstance(init_state_distn, isd.InitialStateDistribution):
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.InitialStateDistribution")

        # Make the transition model
        transition_classes = dict(
            nb=trans.NegativeBinomialSemiMarkovTransitions,
            )
        if isinstance(transitions, str):
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # This is the master list of observation classes.
        # When you create a new observation class, add it here.
        observation_classes = dict(
            gaussian=obs.GaussianObservations,
            diagonal_gaussian=obs.DiagonalGaussianObservations,
            studentst=obs.MultivariateStudentsTObservations,
            t=obs.MultivariateStudentsTObservations,
            diagonal_t=obs.StudentsTObservations,
            diagonal_studentst=obs.StudentsTObservations,
            bernoulli=obs.BernoulliObservations,
            categorical=obs.CategoricalObservations,
            poisson=obs.PoissonObservations,
            vonmises=obs.VonMisesObservations,
            ar=obs.AutoRegressiveObservations,
            autoregressive=obs.AutoRegressiveObservations,
            diagonal_ar=obs.AutoRegressiveDiagonalNoiseObservations,
            diagonal_autoregressive=obs.AutoRegressiveDiagonalNoiseObservations,
            independent_ar=obs.IndependentAutoRegressiveObservations,
            robust_ar=obs.RobustAutoRegressiveObservations,
            robust_autoregressive=obs.RobustAutoRegressiveObservations,
            diagonal_robust_ar=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_robust_autoregressive=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            )

        if isinstance(observations, str):
            observations = observations.lower()
            if observations not in observation_classes:
                raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(observation_classes.keys())))

            observation_kwargs = observation_kwargs or {}
            observations = observation_classes[observations](K, D, M=M, **observation_kwargs)
        if not isinstance(observations, obs.Observations):
            raise TypeError("'observations' must be a subclass of"
                            " ssm.observations.Observations")

        super().__init__(K, D, M=M, transitions=transitions,
                        transition_kwargs=transition_kwargs,
                        observations=observations,
                        observation_kwargs=observation_kwargs,
                        **kwargs)

    @property
    def state_map(self):
        return self.transitions.state_map

    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        T : int
            number of time steps to sample

        prefix : (zpre, xpre)
            Optional prefix of discrete states (zpre) and continuous states (xpre)
            zpre must be an array of integers taking values 0...num_states-1.
            xpre must be an array of the same length that has preceding observations.

        input : (T, input_dim) array_like
            Optional inputs to specify for sampling

        tag : object
            Optional tag indicating which "type" of sampled data

        with_noise : bool
            Whether or not to sample data with noise.

        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states

        x_sample : (T x observation_dim) array_like
            Array of sampled data
        """
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        assert T > 0

        # Check the inputs
        if input is not None:
            assert input.shape == (T,) + M

        # Get the type of the observations
        dummy_data = self.observations.sample_x(0, np.empty(0,) + D)
        dtype = dummy_data.dtype

        # Initialize the data array
        if prefix is None:
            # No prefix is given.  Sample the initial state as the prefix.
            pad = 1
            z = np.zeros(T, dtype=int)
            data = np.zeros((T,) + D, dtype=dtype)
            input = np.zeros((T,) + M) if input is None else input
            mask = np.ones((T,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observations.sample_x(z[0], data[:0], input=input[0], with_noise=with_noise)

            # We only need to sample T-1 datapoints now
            T = T - 1

        else:
            # Check that the prefix is of the right type
            zpre, xpre = prefix
            pad = len(zpre)
            assert zpre.dtype == int and zpre.min() >= 0 and zpre.max() < K
            assert xpre.shape == (pad,) + D

            # Construct the states, data, inputs, and mask arrays
            z = np.concatenate((zpre, np.zeros(T, dtype=int)))
            data = np.concatenate((xpre, np.zeros((T,) + D, dtype)))
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            mask = np.ones((T+pad,) + D, dtype=bool)

        # Convert the discrete states to the range (1, ..., K_total)
        m = self.state_map
        K_total = len(m)
        _, starts = np.unique(m, return_index=True)
        z = starts[z]

        # Fill in the rest of the data
        for t in range(pad, pad+T):
            Pt = self.transitions.transition_matrices(data[t-1:t+1], input[t-1:t+1], mask=mask[t-1:t+1], tag=tag)[0]
            z[t] = npr.choice(K_total, p=Pt[z[t-1]])
            data[t] = self.observations.sample_x(m[z[t]], data[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Collapse the states
        z = m[z]

        # Return the whole data if no prefix is given.
        # Otherwise, just return the simulated part.
        if prefix is None:
            return z, data
        else:
            return z[pad:], data[pad:]

    @ensure_args_not_none
    def expected_states(self, data, input=None, mask=None, tag=None):
        m = self.state_map
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        Ez, Ezzp1, normalizer = hmm_expected_states(replicate(pi0, m), Ps, replicate(log_likes, m))

        # Collapse the expected states
        Ez = collapse(Ez, m)
        Ezzp1 = collapse(collapse(Ezzp1, m, axis=2), m, axis=1)
        return Ez, Ezzp1, normalizer

    @ensure_args_not_none
    def most_likely_states(self, data, input=None, mask=None, tag=None):
        m = self.state_map
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        z_star = viterbi(replicate(pi0, m), Ps, replicate(log_likes, m))
        return self.state_map[z_star]

    @ensure_args_not_none
    def filter(self, data, input=None, mask=None, tag=None):
        m = self.state_map
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        pzp1 = hmm_filter(replicate(pi0, m), Ps, replicate(log_likes, m))
        return collapse(pzp1, m)

    @ensure_args_not_none
    def posterior_sample(self, data, input=None, mask=None, tag=None):
        m = self.state_map
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        z_smpl = hmm_sample(replicate(pi0, m), Ps, replicate(log_likes, m))
        return self.state_map[z_smpl]

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        m = self.state_map
        Ez, _, _ = self.expected_states(data, input, mask)
        return self.observations.smooth(Ez, data, input, tag)

    @ensure_args_are_lists
    def log_likelihood(self, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        m = self.state_map
        ll = 0
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            pi0 = self.init_state_distn.initial_state_distn
            Ps = self.transitions.transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)
            ll += hmm_normalizer(replicate(pi0, m), Ps, replicate(log_likes, m))
            assert np.isfinite(ll)
        return ll

    def expected_log_probability(self, expectations, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        raise NotImplementedError("Need to get raw expectations for the expected transition probability.")

    def _fit_em(self, datas, inputs, masks, tags, verbose = 2, num_iters=100, **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        lls = [self.log_probability(datas, inputs, masks, tags)]

        pbar = ssm_pbar(num_iters, verbose, "LP: {:.1f}", [lls[-1]])

        for itr in pbar:
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data, input, mask, tag)
                            for data, input, mask, tag in zip(datas, inputs, masks, tags)]

            # E step: also sample the posterior for stochastic M step of transition model
            samples = [self.posterior_sample(data, input, mask, tag)
                       for data, input, mask, tag in zip(datas, inputs, masks, tags)]

            # M step: maximize expected log joint wrt parameters
            self.init_state_distn.m_step(expectations, datas, inputs, masks, tags, **kwargs)
            self.transitions.m_step(expectations, datas, inputs, masks, tags, samples, **kwargs)
            self.observations.m_step(expectations, datas, inputs, masks, tags, **kwargs)

            # Store progress
            lls.append(self.log_prior() + sum([ll for (_, _, ll) in expectations]))
            if verbose == 2:
                pbar.set_description("LP: {:.1f}".format(lls[-1]))

        return lls

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None, verbose = 2,
            method="em", initialize=True, **kwargs):
        _fitting_methods = dict(em=self._fit_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        if initialize:
            self.initialize(datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

        return _fitting_methods[method](datas, inputs=inputs, masks=masks, tags=tags, verbose = verbose, **kwargs)

