# samplers/bayesian_optimization_sampler.py
"""
---

## Overview

This sampler Class uses Bayesian Optimization techniques to data efficiently
sample through the search space to yield optimial information gain as 
specified by the acquisition strategy.
Bayesian Optimization sampler using the BoTorch library.

---
"""

from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.samplers.base_sampler import Sampler
# import matplotlib as mpl
# import scienceplots

import botorch, gpytorch, torch
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize, unnormalize

log = get_logger(__name__)


class BayesianOptimizationSampler(Sampler):
    """
    ## Configuration

    To use the `BayesianOptimizationSampler`, specify it in the configuration file as follows:

    ```yaml
    sampler:
        type: BayesianOptimizationSampler
        budget: 50
        initial_samples: 20
        acquisition_batch_size: 10
        acquisition_function: qEI
        random_fraction: 0.2
        bounds: [[0.0, 1.0], [1.0, 5.0]]
        parameters: ['x', 'y']
        observations: ['distance']
        base_run_dir: ./runs
        fully_bayesian: false
        async_samp: false
        failure_prob_filter: false
        ucb_beta: 2.0
        parser: Parser
        parser_config:
        key: value
    ```

    Attributes:
        initial_samples (int): Number of initial samples required.
        verbose (bool): Whether to print verbose output.
        fully_bayesian (bool): Whether to use fully Bayesian models.
        acquisition_batch_size (int): Number of samples in each acquisition batch.
        observations (list): List of observations.
        bounds (list): Bounds for the search space.
        acquisition_function (str): Acquisition function to use.
        random_fraction (float): Fraction of random samples.
        failure_prob_filter (bool): Whether to filter based on failure probability.
        ucb_beta (float): Beta parameter for UCB acquisition function.
        async_samp (bool): Whether to use asynchronous sampling.
        parameters (list): List of parameter names.
        parser (type): Parser type for collecting sample information.
        parser_config: Parser kwargs

    ---
    
    ## Assumptions and notes

     - The sampler assumes continuous numeric parameters and bounded search spaces.

     - Bayesian optimization relies on existing evaluation results stored in base_run_dir.

     - Sampling proceeds in two phases:
          Random sampling until initial_samples are collected.
          Model-based sampling using a Gaussian Process surrogate.

    ---
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initializes the BayesianOptimization sampler 
        with the given parameters.

        Args:
            bounds (list[tuple[float, float]]): Search space bounds.
            parameters (list[str]): Parameter names.
            initial_samples (int, optional): Number of initial random samples.
            acquisition_function (str, optional): Acquisition strategy.
            acquisition_batch_size (int, optional): Number of samples per batch.
            random_fraction (float, optional): Fraction of random exploration.
            async_samp (bool, optional): Enable asynchronous sampling.
        
        """
        log.info('INITIALISING BAYESIAN OPTIMIZATION SAMPLER')
        self.bounds            = kwargs.get('bounds', [None])
        self.parameters        = kwargs.get('parameters', [])  

        # Sampling parameters
        self.initial_samples   = kwargs.get('initial_samples', 50)
        self.acq_batch_size    = kwargs.get('acquisition_batch_size', 20)
        self.acq_function      = kwargs.get('acquisition_function', 'qLEI')
        self.random_fraction   = kwargs.get('random_fraction', 0.2)
        self.fail_p_filter     = kwargs.get('failure_prob_filter', False)
        self.ucb_beta          = kwargs.get('ucb_beta', 2.0)  
        self.async_samp        = kwargs.get('async_samp', False)

        # GPR parameters
        self.fully_bayesian    = kwargs.get('fully_bayesian', False)
        self.covar             = kwargs.get('covar_kernel', 'Matern-3/2')

        # Printing flags
        self.verbose           = kwargs.get('verbose', False)

        self.futures = []


    def get_next_samples(self):
        """
        Generates the next parameter samples based on the probabilistic 
        surrogate model and the acquisition function. 

        Returns:
            list: List of dictionaries containing next parameter samples.

        """
        
        if self.result_dictionary == [None]:
            self.collected_samples = 0
        else:
            self.collected_samples = len(self.result_dictionary['inputs'])

        # Assume random sampling if self.collected_samples is below 
        # the number of initial samples.
        batch_samples = []

        if self.collected_samples < self.initial_samples:
            log.info(str(100*self.collected_samples/self.initial_samples),
                  ' % OF INITIAL SAMPLES FOR BAYESIAN OPTIMIZATION COLLECTED')            
            for _ in range(int(self.acq_batch_size)):
                params = [
                    torch.distributions.Uniform(lb, ub).sample().item()
                    for (lb, ub) in self.bounds
                ]
                param_dict = dict(zip(self.parameters, params))
                batch_samples.append(param_dict)
        else:
            # Fit the surrogate model
            self.train_surrogate()
            # The acquisition stragety is chosen here.  
            acq = None  # Initialize acq to avoid 
                        # possibly-used-before-assignment error
            match self.acq_function:
                case 'qLEI':
                    acq = botorch.acquisition.qLogExpectedImprovement(
                        model=self.model, 
                        best_f=self.best_f)
                case 'qUCB':
                    acq = botorch.acquisition.qUpperConfidenceBound(
                        model=self.model, 
                        beta=self.ucb_beta)
                case 'qEI':
                    acq = botorch.acquisition.qExpectedImprovement(
                        model=self.model, 
                        best_f=self.best_f)
                case 'qPI':
                    acq = botorch.acquisition.qProbabilityOfImprovement(
                        model=self.model, 
                        best_f=self.best_f)
                case 'EI':
                    acq = botorch.acquisition.ExpectedImprovement(
                        model=self.model,
                        best_f=self.best_f)
                case 'LEI':
                    acq = botorch.acquisition.LogExpectedImprovement(
                        model=self.model,
                        best_f=self.best_f)
                case _:
                    raise ValueError("Unsupported acquisition function:", 
                                      f"{self.acq_function}")

            if self.async_samp:
                if torch.rand(1) < self.random_fraction:
                    params = [
                        torch.distributions.Uniform(lb, ub).sample().item()
                        for (lb, ub) in self.bounds
                    ]
                    param_dict = dict(zip(self.parameters, params))
                    batch_samples.append(param_dict)
                    self.submitted += len(batch_samples)
                    return batch_samples
                else:
                    qval = 1
            else:
                acq_f =  (1 - self.random_fraction) 
                rand_f = self.random_fraction
                qval = int(acq_f*self.acq_batch_size)
                for _ in range(int(rand_f*self.acq_batch_size)):
                    params = [
                        torch.distributions.Uniform(lb, ub).sample().item()
                        for (lb, ub) in self.bounds
                    ]
                    param_dict = dict(zip(self.parameters, params))
                    batch_samples.append(param_dict)

            # Acquisition function is optimized in [0, 1]**d domain
            #boundtensor = torch.DoubleTensor(self.bounds).T
            lower_bound = torch.zeros(len(self.bounds), dtype=float)
            upper_bound = torch.ones(len(self.bounds), dtype=float)
            lower_bound = lower_bound.unsqueeze(0)
            upper_bound = upper_bound.unsqueeze(0)
            boundtensor = torch.cat((lower_bound, upper_bound))
            candidates, acq_values = optimize_acqf(
                acq, 
                bounds=boundtensor,
                sequential=False, 
                q=qval,
                num_restarts=10,
                raw_samples=1024
                )

            bounds = torch.tensor(self.bounds)
            candidates = unnormalize(candidates, bounds.T)
            # This part of the code can be cleaned by implementing 
            # failure probability in the acquisition function. 
            if self.fail_p_filter:
                if self.result_dictionary_failed != [None]:
                    cand_accept = []
                    not_enough = True
                    target_len = len(candidates[:,0])
                    while not_enough:
                        norm_inp = normalize(candidates, bounds.T)
                        pred = self.model_failed(norm_inp)
                        pred = pred.mean
                        for i in range(len(pred)):
                            if torch.rand(1) > pred[i]:
                                cand_accept.append(candidates[i,:].numpy())
                        if len(cand_accept) < target_len:
                            candidates, acq_values = optimize_acqf(
                                acq, 
                                bounds=boundtensor,
                                sequential=False, 
                                q=target_len - len(cand_accept),
                                num_restarts=10,
                                raw_samples=1024
                                )
                            candidates = unnormalize(candidates, bounds.T)
                        else:
                            not_enough = False
                            candidates = torch.tensor(cand_accept)
                
            for index, _ in enumerate(range(candidates.size(dim=0))):
                params_dict = dict(zip(self.parameters, 
                                       candidates[index,:].numpy()))
                batch_samples.append(params_dict)

        self.submitted += len(batch_samples)
        return batch_samples
    
    def register_future(self, future):
        self.futures.append(future)

    # Fitting the GPR.
    def train_surrogate(self):
        if self.verbose:
            log.info('FITTING THE GPR')
        # Presently implemented as single objective model. Therefore,
        # sum over the distances and norm
        distances = torch.tensor(self.result_dictionary['distances'][:])
        distances = torch.sum(distances, axis=1)
        distances = standardize(distances)

        inputs = torch.tensor(self.result_dictionary['inputs'][:])
        bounds = torch.tensor(self.bounds)
        input_vector = normalize(inputs, bounds.T)

        # Check if normzalized inputs are below 0 or larger than 1.
        #dummy = torch.abs(input_vector - 0.5)
        #dummy = torch.max(dummy, axis=1).values < 0.5
        #idx = torch.where(dummy) 
        #input_vector = input_vector[idx]
        #distances = distances[idx]

        distances = distances.unsqueeze(distances.ndim)
        # Multiply by -1 the task to a maximization problem.
        distances = -distances

        # Default covar module
        covar_module = gpytorch.kernels.MaternKernel(nu=1.5)

        if self.covar == 'Matern-5/2':
            covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        if self.covar == 'Matern-3/2':
            covar_module = gpytorch.kernels.MaternKernel(nu=1.5)
        if self.covar == 'Matern-1/2':
            covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
        if self.covar == 'RBF':
            covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=len(bounds.T))
        if self.fully_bayesian:
            if self.verbose:
                log.info('SaasFullyBayesianSingleTaskGP')
            # Default kernel is Matern-5/2
            gp = botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP(
                     input_vector, 
                     distances)
            gp.covar_module = covar_module
            botorch.fit.fit_fully_bayesian_model_nuts(gp)
        else:
            if self.verbose:
                log.info('SingleTaskGP')
            gp = botorch.models.SingleTaskGP(
                                             input_vector,
                                             distances, 
                                             covar_module=covar_module)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood=gp.likelihood, 
                model=gp)
            botorch.fit.fit_gpytorch_mll(mll)
        self.model = gp
        self.best_f, idx = torch.max(distances, 0)
        self.best_f_loc = input_vector[idx,:]
        if self.result_dictionary_failed != [None]:

            inp = torch.tensor(self.result_dictionary['inputs'])
            inp_f = torch.tensor(self.result_dictionary_failed['inputs'])
            f_0 = torch.tensor(self.result_dictionary['failure'],
                               dtype=torch.float64)
            f_1 = torch.tensor(self.result_dictionary_failed['failure'],
                               dtype=torch.float64)

            gp_failed = botorch.models.SingleTaskGP(
                normalize(torch.cat([inp, inp_f]), bounds.T),
                torch.cat([f_0, f_1]).unsqueeze(0).T)
            mll_gp_failed = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood=gp_failed.likelihood, 
                model=gp_failed)

            botorch.fit.fit_gpytorch_mll(mll_gp_failed)
            self.model_failed = gp_failed


    def skip(self, index):
        raise NotImplementedError("skip not implemented for BayesianOptimizationSampler.")


