import autograd.numpy as np 
from scipy.special import logsumexp
import scipy.stats as st
import autograd.numpy.random as npr
from scipy import optimize
from scipy.special import softmax
from utils_mglm.softplus import softplus
from utils_mglm.negloglike_bernoulliGLM import *
from utils_mglm.negloglike_Mstep_bernoulliGLM import *
from utils_mglm.multinomial_rvs import *
import time



class MGLM(object):
    """
    This class implements mixture of GLMS

    Notation:
    K: number of discrete latent states
    D: dimensionality of observations (D=1 in this code)
    M: dimensionality of inputs
    C: Categories of outputs

    In the code we will refer to the discrete
    latent state sequence as z, input as x, and the outputs as y.
    """

    def __init__(self, K, D=1, M=0, C=2, prior_mean=0, prior_sigma=np.sqrt(10)):
        self.K, self.D, self.M, self.C = K, D, M, C

        # Parameters of prior over weights
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma 

        # Observation weights
        self.weights = np.ones((K,C-1,M))*self.prior_mean
        # state distribution
        self.state_distn = np.ones(K)/K

    @property
    def params(self):
        """
        Return parameters 
        """
        return self.state_distn, self.weights


    @params.setter
    def params(self, value):
        """ 
        Set parameters 
        """
        self.state_distn = value[0]
        self.weights = value[1]


    def initialize(self):
        """
        Initialize parameters at random
        """
        # Uniformly initializing the state distribution
        K = self.K
        M = self.M
        self.state_distn = np.ones(K)/K
        # Initializing weights by drawing samples from the prior
        mean = self.prior_mean*np.ones((self.M,))
        cov = (self.prior_sigma**2)*np.eye(self.M,self.M)
        self.weights = np.reshape(st.multivariate_normal.rvs(mean=mean, cov = cov, size=K*(self.C-1)), (K,self.C-1,M))
        return self.state_distn, self.weights


    def log_likelihood(self, datas, inputs):
        """
        Returns log-likelihood of the mixture of GLMs
        """
        T = inputs.shape[0]
        # Logits KXCXT
        logits = np.empty((self.K,self.C,T))
        for k in range(self.K):
            logits[k] = self.compute_logits(k, inputs)

        probs = np.empty((self.K, T))
        for k in range(self.K):
            pr_y = np.exp(logits[k])
            pr_y_c = pr_y[datas.ravel().astype(int).tolist(), np.arange(pr_y.shape[1])]
            probs[k,:] = self.state_distn[k]*pr_y_c

        marginalised_prob = np.sum(probs, axis=0)[:, np.newaxis]
        ll = np.sum(np.log(marginalised_prob))
        return ll


    def compute_logits(self, z, inputs):
        """"
        Returns log p(y=c|x,z) for all c (for a single GLM), size CXT
        """
        #  Transposing to C-1 X K X M
        wsT = np.transpose(self.weights, (1,0,2))
        # Adding a row of zeros to the weights for category 0
        # C X K X M
        wsT = np.vstack([np.zeros((1,self.K,self.M)), wsT])
        wzT = np.reshape(wsT[:,z,:], (self.C,self.M))
        # Computing log p(y=c|z,x)
        wTx = np.reshape(wzT@inputs.T, (self.C,inputs.shape[0]))
        # CXT
        logits = (wTx) - logsumexp(wTx, axis=0,  keepdims=True)
        return logits



    def sample_y(self, T, inputs=None, with_noise=True):
        """
        Sample synthetic data from the model. 

        Parameters
        ----------
        T : int
            number of time steps to sample

        input : (T, input_dim) array_like
            Inputs to specify for sampling

        with_noise : bool
            Whether or not to sample data with noise.

        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states

        y_sample : (T x observation_dim) array_like
            Array of sampled observations
        """

        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        assert T > 0

        # Check the inputs
        if inputs is not None:
            assert inputs.shape == (T,) + M

        z = np.zeros(T, dtype=int)
        datas = np.zeros((T,) + D)

        # Sample observation at each time point
        pi = self.state_distn
        z =  npr.choice(self.K, p=pi, size=(T,))
        # Sample y 
        for t in range(T):
            log_pr_y = self.compute_logits(z[t], np.reshape(inputs[t], (1, self.M)))
            datas[t] = npr.choice(self.C, p=np.exp(log_pr_y.ravel()))
        # Return the whole data.
        return z, datas



    def posterior_over_states(self, datas, inputs):
        """
        Function to compute posterior over latent states at each sample
        , equivalent to E step in EM
        """
        T = datas.shape[0]
        K = self.K 

        pz = np.zeros([T, K])
        #  Computing posterior over zs
        for k in range(K):
            # (CXT), p(y|z,x)
            pr_y = np.exp(self.compute_logits(k,inputs))
            # Extracting p(y=y_sampled|z,x)
            pr_y_c = pr_y[datas.ravel().astype(int), np.arange(pr_y.shape[1])]
            pz[:,k] = self.state_distn[k]*pr_y_c
        # Normalize
        pz = pz/np.sum(pz, axis=1)[:, np.newaxis]
        return pz



    def sampleposterior_bernoulliGLM(self, weights, datas, inputs, k):
        """
        Sampling from the poster over GLM weights by Laplace Approximation 
        #TODO: Extend this to multi-class
        """
        # Inputs should be (TXM)
        T = inputs.shape[0]  

        # Sample weights from prior_covariance
        prior_covariance = (self.prior_sigma**2)*np.eye(self.M,self.M)
        prior_w0 = self.prior_mean*np.ones((self.M))*0.1

        def dnegP(weights, datas, inputs):
            """
            Function to compute derivative of negative log posterior of bernoulli GLM
            """
            inv_prior_covariance = (1/(self.prior_sigma**2))*np.eye(self.M,self.M)
            prior_w0 = self.prior_mean*np.ones((self.M))
            return dnegL(weights, datas, inputs) + inv_prior_covariance@(weights-prior_w0)

        def HessnegP(weights, datas, inputs):
            """
            Function to compute hessian of negative log posterior of bernoulli GLM
            """
            inv_prior_covariance = (1/(self.prior_sigma**2))*np.eye(self.M,self.M)
            return HessnegL(weights, datas, inputs) + inv_prior_covariance

        # MAP inference for weights
        def w_map(weights, datas, inputs):
            # Compute projection of inputs onto GLM weights for each class
            wTx = inputs@weights
            # Evaluating softplus and its derivatives
            f, df, ddf = softplus(wTx)
            # Negative LL
            negL = -datas.T@wTx + np.sum(f)
            # Negative LP
            negP = negL - st.multivariate_normal.logpdf(weights, mean=prior_w0, cov=prior_covariance)
            return (negP, dnegP(weights, datas, inputs))

        Results = optimize.minimize(w_map, x0=weights, method='trust-exact', jac = True, hess = HessnegP, args = (datas, inputs))
        weights_MAP = Results['x']
        # This might be the bottleneck, when increasing input dimensionality/ See if Lewi's idea applies here
        covariance = np.linalg.inv(HessnegP(weights_MAP, datas, inputs))
        # Sample from multivariate normal with above mean and var
        weights_k = st.multivariate_normal.rvs(mean=weights_MAP, cov =covariance)
        weights_k = np.reshape(weights_k, (self.C-1, self.M))

        return weights_k, weights_MAP, covariance


    def Mstep_bernoulliGLM(self, weights, datas, inputs, pstate):
        weights = np.reshape(weights, (self.M,1))
        # MLE inference for weights 
        Results = optimize.minimize(negloglike_Mstep_bernoulliGLM, x0=weights, method='trust-exact', jac = dnegL_Mstep, hess = HessnegL_Mstep, args = (datas, inputs, pstate, self.prior_mean, self.prior_sigma))
        weights_MLE = Results['x']
        return weights_MLE



    def fit_EM(self, datas, inputs, initialize = None, n_iters=500):
        """
        Perform EM to recover parameters  
        """
        K = self.K
        M = self.M
        D = self.D
        T = datas.shape[0]

        # Initial assignments
        if initialize:
            self.state_distn, self.weights = initialize
        else:
            self.state_distn, self.weights = self.initialize()
           
        # Lists to store results
        ll = np.zeros((n_iters))
        weights_ateverystep = np.zeros((n_iters,K, M))
        pis_ateverystep = np.zeros((n_iters, K))

        for it in range(n_iters):
            # print("Doing EM: At iteration: "+str(it))
            # E-step
            # [T X K]
            pzs = self.posterior_over_states(datas, inputs)
            # M-step
            for k in range(K):
                weights_k = self.Mstep_bernoulliGLM(self.weights[k], datas, inputs, pzs[:,k]) 
                self.weights[k] = np.reshape(weights_k, (self.C-1, self.M))
                self.state_distn[k] = np.sum(pzs[:,k])/T

            ll[it] = self.log_likelihood(datas, inputs)
            weights_ateverystep[it] = np.reshape(self.weights, (K,M))
            pis_ateverystep[it] = self.state_distn

        return weights_ateverystep, pis_ateverystep, ll



    def fit_gibbs(self, datas, inputs, initialize = None, n_iters=500, burnin=100):
        """
        Perform gibbs sampling to approximate the posterior with modified weight samplign stage (current implementation supports only bernoulli GLM)
        """
        K = self.K
        M = self.M
        D = self.D
        T = datas.shape[0]

        # Initial assignments
        zs = np.zeros(T, dtype=int)  

        if initialize:
            self.state_distn, self.weights = initialize
        else:
            self.state_distn, self.weights = self.initialize()    
           
        # Lists to store all results
        weights_sampled = np.zeros((n_iters+burnin,K, M))
        pis_sampled = np.zeros((n_iters+burnin,K))
        ll = np.zeros((n_iters+burnin,))

        # Gibbs sampler
        for it in range(n_iters+burnin):
            # Sample from full conditional of assignment
            # z ~ p(z) \propto pi*p(y|pi)
            probs = self.posterior_over_states(datas, inputs)
            # For each data point, draw the state assignment
            zs = np.argmax(multinomial_rvs(np.ones((T,), dtype=int), probs), axis=1)
            #---------------------------------------------------------------------------------------    
            # Sample from full conditional of observation weights
            # number of samples per state
            Ns = np.zeros(K, dtype='int')
            # Store means of the posterior over weights
            w_means = np.empty((K,M))
            for k in range(self.K):
                Xk = inputs[zs==k]
                Xk = np.reshape(Xk, (Xk.shape[0], M))
                Yk = datas[zs==k]
                Yk = np.reshape(Yk, (Yk.shape[0], D))
                Ns[k] = Xk.shape[0]
                # Previous weight
                weight_old = self.weights[k]
                # Define proposal distribution Q conditional on other parameters and data 
                weight_proposed, w_mean, covariance = self.sampleposterior_bernoulliGLM(weight_old, Yk, Xk, k)
                # P(\theta^*_i \mid \theta_{-i}, D)
                logp1 = - neglogp_bernoulliGLM(weight_proposed, Yk, Xk, self.prior_mean, self.prior_sigma)
                # Q(\theta^*_i \mid \theta_{-i}, D)
                logq1 = st.multivariate_normal.logpdf(weight_proposed, mean=w_mean, cov=covariance)
                # P(\theta_i \mid \theta_{-i}, D)
                logp2 = - neglogp_bernoulliGLM(weight_old, Yk, Xk, self.prior_mean, self.prior_sigma)
                # Q(\theta_i \mid \theta_{-i}, D)
                logq2 = st.multivariate_normal.logpdf(weight_old, mean=w_mean, cov=covariance)
                # Compute acceptance probability
                acceptprob = np.min([1, np.exp(logp1+logq2 - (logp2+logq1))])

                accept = np.random.uniform(size=1)<acceptprob
                if it==0 or accept==1:
                    weight = weight_proposed
                else:
                    weight = weight_old
                # Ensure this is (1,C-1,M)
                self.weights[k] = np.reshape(weight, (1,self.C-1, self.M))
            #---------------------------------------------------------------------------------------
            # Sample from full conditional of the mixing weight
            # pi ~ Dir(alpha + n)
            self.state_distn = np.random.dirichlet(1+Ns) 
            #------------------------------------------------------------------------------------------
            # Keeping track to send to plot
            weights_sampled[it, :] = np.reshape(self.weights, (K,M))
            pis_sampled[it, :] = self.state_distn
            ll[it] = self.log_likelihood(datas, inputs)
            # latents[it,:] = zs 

        return weights_sampled[burnin:], pis_sampled[burnin:], ll[burnin:]


        


