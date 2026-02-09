from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_parser
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import scipy.optimize as optimize

import botorch, gpytorch, torch, pyro, lampe
import lampe.plots as plots
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize, unnormalize



class BayesianOptimizationSampler(Sampler):
    """
    ---

    ## Overview

    This sampler Class uses Bayesian Optimization techniques to data efficiently
    sample through the search space to yield optimial information gain as 
    specified by the acquisition strategy.
    Bayesian Optimization sampler using the BoTorch library.

    ---

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
        print('INITIALISING BAYESIAN OPTIMIZATION SAMPLER')
        self.base_run_dir      = kwargs.get('base_run_dir', '')
        self._budget           = kwargs.get('budget', 20)
        self.bounds            = kwargs.get('bounds', [None])
        self.parameters        = kwargs.get('parameters', [])  
   
        # Observations and parser configs
        self.observations      = kwargs.get('observations', [None])
        self.parser_type       = kwargs.get('parser', None)  
        self.parser_config     = kwargs.get('parser_config',{})

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

        # Plotting and printing flags
        self.verbose           = kwargs.get('verbose', False)
        self.plot_GPR_flag     = kwargs.get('plot_GPR', False)
        self.GPR_plot_dim      = kwargs.get('GPR_plot_dim', [0])
        self.plot_GPR_file     = kwargs.get('plot_file', False)
        self.plot_frequency    = kwargs.get('plot_frequency', 1)
        self.plot_debug        = kwargs.get('plot_debug', False)
        self.plot_labels       = kwargs.get('plot_labels', None)
        self.plot_progress     = kwargs.get('plot_progress', False)

        #if self.parser_config == None:
        if self.parser_type == None:
            self.parser = None
        else:
        #        self.parser = import_parser(self.parser_type)
        #else:
            self.parser = import_parser(self.parser_type, self.parser_config)
        self.futures = []


    def get_next_samples(self):
        """
        Generates the next parameter samples based on the probabilistic 
        surrogate model and the acquisition function. 

        Returns:
            list: List of dictionaries containing next parameter samples.

        """
        # Build the result dictionary
        self.build_result_dictionary(self.base_run_dir)
        
        if self.result_dictionary == [None]:
            self.collected_samples = 0
        else:
            self.collected_samples = len(self.result_dictionary['inputs'])

        # Assume random sampling if self.collected_samples is below 
        # the number of initial samples.
        batch_samples = []

        if self.collected_samples < self.initial_samples:
            print(str(100*self.collected_samples/self.initial_samples),
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
            if self.acq_function == 'qLEI':
                acq = botorch.acquisition.qLogExpectedImprovement(
                    model=self.model, 
                    best_f=self.best_f)
            elif self.acq_function == 'qUCB':
                acq = botorch.acquisition.qUpperConfidenceBound(
                    model=self.model, 
                    beta=self.ucb_beta)
            elif self.acq_function == 'qEI':
                acq = botorch.acquisition.qExpectedImprovement(
                    model=self.model, 
                    best_f=self.best_f)
            elif self.acq_function == 'qPI':
                acq = botorch.acquisition.qProbabilityOfImprovement(
                    model=self.model, 
                    best_f=self.best_f)
            elif self.acq_function == 'EI':
                acq = botorch.acquisition.ExpectedImprovement(
                    model=self.model,
                    best_f=self.best_f)
            elif self.acq_function == 'LEI':
                acq = botorch.acquisition.LogExpectedImprovement(
                    model=self.model,
                    best_f=self.best_f)
            if acq is None:
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
            if self.plot_progress:
                self.progress_plots()  

        self.submitted += len(batch_samples)
        return batch_samples
    
    def register_future(self, future):
        self.futures.append(future)

    # Fitting the GPR.
    def train_surrogate(self):
        if self.verbose:
            print('FITTING THE GPR')
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
                print('SaasFullyBayesianSingleTaskGP')
            # Default kernel is Matern-5/2
            gp = botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP(
                     input_vector, 
                     distances)
            gp.covar_module = covar_module
            botorch.fit.fit_fully_bayesian_model_nuts(gp)
        else:
            if self.verbose:
                print('SingleTaskGP')
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
        if self.plot_GPR_flag:
            if (len(self.result_dictionary['distances']) % 
                self.plot_frequency == 0):
                self.plot_GPR(plot_dims=self.GPR_plot_dim)

        if self.plot_debug:
            self.plot_distances()
            self.plot_result_sequence()

    # Result dictionary building features. This function will be moved to the
    # supervisory module in future developments.

    def build_result_dictionary(self, base_run_directory: str, 
                                normalize=True): 
        """
        This function can be used to build the result_dictionary based on the
        existing runs in the base_run_directory.
        
        Args:
            base_run_directory (str): Path to the base run directory

        Kwargs:
            normalize: A Boolean flag to turn of normalization. 
                       This is True by default.

        Returns:
            This function does not return anything directly. However, 
            the function establishes self.result_dictionary as well as
            self.result_dictionary_norm, if normalize is set to True.         

        """
        result_dictionary = None
        result_dictionary_failed = None
        if self.verbose:
            print('BUILDING RESULT DICTIONARY')    
    
        # Load a stored result_dictionary file, if such a file exists.
        if os.path.isfile(os.path.join(base_run_directory, 
                                       'result_dictionary.pkl')):
            resdict = open(os.path.join(base_run_directory,  
                                        'result_dictionary.pkl'),'rb')
            result_dict = pkl.load(resdict)
            result_dictionary = result_dict['result_dictionary']
            # To be implemented. Presently just reconstruct the dictionary 
            # of failed cases everytime. This works but is a bit inefficient. 
            #result_dictionary_failed = result_dict['result_dictionary_failed']
            resdict.close()

        # Obtain a list of run_directories within the base_run_directory
        dirlist = os.listdir(base_run_directory)
        # Loop over the established runs. This can be streamlined if needed.
        
        # List of tags to identify entries to skip in dirlist. 
        # Present implementation loops through the run-directory.
        # This list gives identifiers to recognize files & directories that
        # do not represent samples. 
        skiplist = ['yaml', 'worker_out', 'FINISHED', '.pkl', '.csv', '_RUN',
                    'GPR', 'Fig']

        for dirname in dirlist:
            # See if the dirname is on the skiplist
            skiptags = [tag in dirname for tag in skiplist]
            if any(skiptags):
                continue
            else:
                if result_dictionary != None:
                    if (os.path.join(base_run_directory, dirname) in
                        result_dictionary['run_dir']):
                        continue
                    else:
                        sample_dict = self.parser.collect_sample_information(
                            os.path.join(base_run_directory, dirname),
                            self.observations)
                else:
                    sample_dict = self.parser.collect_sample_information(
                        os.path.join(base_run_directory, dirname),
                        self.observations)
            if sample_dict['failure'] == 0:
                if result_dictionary != None:
                    for key in sample_dict.keys():
                        # Append values to the lists corresponding to each 
                        # key.
                        result_dictionary[key].append(sample_dict[key]) 
                else:
                    result_dictionary = sample_dict
                    for key in result_dictionary.keys():
                        result_dictionary[key]=[result_dictionary[key]]
            else:
                if result_dictionary_failed != None:
                    for key in sample_dict.keys():
                        # Append values to the lists corresponding to each 
                        # key.
                        result_dictionary_failed[key].append(sample_dict[key]) 
                else:
                    result_dictionary_failed = sample_dict
                    for key in result_dictionary_failed.keys():
                        result_dictionary_failed[key]=\
                            [result_dictionary_failed[key]]
        if result_dictionary != None:
            self.result_dictionary = result_dictionary
        else:
            self.result_dictionary = [None]
        if result_dictionary_failed != None:
            self.result_dictionary_failed = result_dictionary_failed
        else:
            self.result_dictionary_failed = [None]
        # Save the result_dictionary into a pickle file.
        if result_dictionary != None:
            res_dump = {'result_dictionary':result_dictionary,
                        'result_dictionary_failed':result_dictionary_failed}
            resdict = open(
                os.path.join(base_run_directory, 
                'result_dictionary.pkl'),'wb')
            pkl.dump(res_dump, resdict)
            resdict.close()

    # Plotting ------------------------------------------------------

    def progress_plots(self):
        # Nice style for plotting
        plt.style.use(['science','no-latex'])

        # Plot the distances as a function of sample number
        plt.figure(1)
        self.plot_result_sequence()
        plt.savefig(self.base_run_dir+'/Fig_distance_vs_sample.svg')
        plt.close()

        # Plot distances as a function of variable        
        plt.figure(1)
        self.plot_distances()
        plt.savefig(self.base_run_dir+'/Fig_distances_vs_variable.svg')
        plt.close()

        # Sample the posterior
        self.posterior_samples()
        self.find_MAP()
        # Corner plot
        plt.figure(1)
        self.corner_plot(point=np.array(self.MAP['x']))
        plt.savefig(self.base_run_dir+'/Fig_corner_plot.svg')
        plt.close()

        # Plot the best case
        try:
            dists = np.array(self.result_dictionary['distances'])
            ndistances = np.shape(dists)[1]
            if ndistances > 1:
                for i in range(ndistances):
                    idx = np.where(dists[:,i] == 
                                   np.min(dists[:,i]))
                    idx = idx[0][0]
                    run_dir = self.result_dictionary['run_dir'][idx]
                    self.plot_forward(run_dir, str(i))
                idx = np.where(np.sum(dists, axis=1) ==
                               np.min(np.sum(dists, axis=1)))
                idx = idx[0][0]
                run_dir = self.result_dictionary['run_dir'][idx]
                self.plot_forward(run_dir, 'sum')
            else:
                idx = np.where(dists == 
                               np.min(dists))
                idx = idx[0][0]
                run_dir = self.result_dictionary['run_dir'][idx]
                plt.figure(1)
                self.plot_forward(run_dir, str(i))
            for i in range(ndistances):
                if ndistances == 1:
                    idx = np.where(dists == 
                                   np.min(dists))
                else:
                    idx = np.where(dists[:,i] == 
                                   np.min(dists[:,i]))             
                idx = idx[0][0]
                run_dir = self.result_dictionary['run_dir'][idx]
                self.plot_forward(run_dir, str(i))
        except:
            plt.close()
            print('Result plotting did not work. Have you implemented',
                  ' plotting features in the parser?')
            
    def plot_forward(self, run_dir, label='_'):
        plt.figure(1)
        self.parser.collect_sample_information(
            run_dir,
            self.observations, 
            plot_comparison=True)
        plt.savefig(self.base_run_dir+'/Fig_result_'+label+'.svg')
        plt.close()

    def plot_result_sequence(self):
        dists = np.array(self.result_dictionary['distances'])
        ndistances = np.shape(dists)[1]
        colorvec = ['k', 'r', 'b', 'm']
        for i in range(ndistances):
            plt.plot(dists[:,i],'.', color=colorvec[i])
        plt.ylabel('Distance')
        plt.xlabel('# sample')
        #plt.show()

    def plot_distances(self):
        """
        This is a helper function for plotting the sample distributions.
        The coding is not particularly elegant and can certainly be improved.
        """
        inputs = torch.tensor(self.result_dictionary['inputs'])
        distances = torch.tensor(self.result_dictionary['distances'])

        numdist = distances[1].shape[0]
        numinp = inputs[1].shape[0]
        ncols = int(numinp)
        nrows = int(numdist)
        if ncols == 1:
            if nrows == 1:
                plt.plot(inputs[:], 
                         distances[:],'ko')
                plt.xlabel(self.plot_labels[0])
            else:
                fig, axs = plt.subplots(nrows=nrows, 
                                        ncols=ncols, 
                                        sharex='col')
                for i in range(numdist):
                    axs[int(i)].plot(
                        inputs[:], 
                        distances[:,i],'ko')
                    if i == numdist - 1:
                        axs[int(i)].set_xlabel(self.plot_labels[0])
        elif nrows == 1:
            fig, axs = plt.subplots(nrows=nrows, 
                                    ncols=ncols, 
                                    sharey='row',
                                    figsize=(3.0*ncols, 3.0)
                                    )
            for j in range(numinp):
                axs[int(j)].plot(inputs[:,j], 
                                 distances[:],'ko')
                axs[int(j)].set_xlabel(self.plot_labels[j])
        else:
            fig, axs = plt.subplots(
                nrows=nrows, 
                ncols=ncols, 
                sharex='col', 
                sharey='row',
                figsize=(3.0*ncols, 3.0*nrows)
                )
            for i in range(numdist):
                for j in range(numinp):
                    axs[int(i), int(j)].plot(
                        inputs[:,j], 
                        distances[:,i], 'ko')
                    if i == numdist - 1:
                        axs[int(i), int(j)].set_xlabel(self.plot_labels[j])
        plt.tight_layout()
        #plt.show()

    def initialize_plot(self, plot_dims=[0, 1], base_point = 0):
        distances = torch.tensor(self.result_dictionary['distances'])
        distances = torch.sum(distances, axis=1)
        distances = standardize(distances)
        inputs = torch.tensor(self.result_dictionary['inputs'])
        bounds = torch.tensor(self.bounds)

        if base_point == 0:
            idx = torch.where(distances == torch.min(distances))
            idx = idx[0]
            base_point = inputs[idx]

        # GP is operating in [0,1]^d domain.
        base_point = base_point
        inpmin = torch.min(bounds, axis=1).values
        inpmax = torch.max(bounds, axis=1).values
        inprang = inpmax - inpmin
        input_scaled = (base_point - inpmin)/inprang
        # Establish the plotting axes.
        xtt_vals = torch.zeros((10000,len(self.bounds)))
        xt1 = 0
        xt2 = 0
        if len(plot_dims) == 2:
            xt1 = torch.linspace(
                self.bounds[plot_dims[0]][0], 
                self.bounds[plot_dims[0]][1], 100)
            xt2 = torch.linspace(
                self.bounds[plot_dims[1]][0], 
                self.bounds[plot_dims[1]][1], 100)
            xtt1, xtt2 = torch.meshgrid(
                torch.linspace(0,1,100), 
                torch.linspace(0,1,100))
            xtt_vals[:,plot_dims[0]] = xtt1.flatten()
            xtt_vals[:,plot_dims[1]] = xtt2.flatten()
        if len(plot_dims) == 1:
            xt1 = torch.linspace(
                self.bounds[plot_dims[0]][0], 
                self.bounds[plot_dims[0]][1], 10000)
            xt2 = 0
            xtt_vals[:,plot_dims[0]] = torch.linspace(0,1,10000)

        for i in range(len(self.bounds)):
            if i not in plot_dims:
                xtt_vals[:,i] = torch.ones(10000)*input_scaled[0][i]

        return xt1, xt2, xtt_vals, inputs, distances

    def plot_GPR(self, plot_dims=[0, 1], base_point = 0):
        """
        This is a helper function for plotting the GPR.
        """

        xt1, xt2, xtt_vals, inputs, distances = \
            self.initialize_plot(plot_dims=plot_dims, base_point=base_point)

        with torch.no_grad():
            pred = self.model.likelihood(self.model(xtt_vals))
            if len(plot_dims) == 2:
                pred = torch.reshape(pred.mean, (100, 100))
            plt.style.use(['science','no-latex'])
            if len(plot_dims) == 2:
                plt.contourf(xt1, xt2, pred, cmap='jet')
                plt.xlabel(self.parameters[plot_dims[0]])
                plt.ylabel(self.parameters[plot_dims[1]])
            if len(plot_dims) == 1:
                plt.plot(
                    inputs[:,plot_dims[0]], 
                    standardize(distances),'k.')
                plt.plot(xt1, -pred.mean, 'r-')
                lower, upper = pred.confidence_region()
                plt.fill_between(xt1, -lower, y2=-upper, alpha=0.5) 
                plt.xlabel(self.parameters[plot_dims[0]])
            if self.plot_GPR_file:
                plt.savefig(os.path.join(self.base_run_dir,
                            'GPR_'+\
                            str(len(self.result_dictionary['distances']))+\
                            '.svg'))
                plt.close()
            else:
                plt.show()

    def plot_posterior(self, plot_dims=[0,1], base_point=0, threshold=-1):
        """
        This is a helper function for plotting the posterior.
        """
        xt1, xt2, xtt_vals, inputs, distances = \
            self.initialize_plot(plot_dims=plot_dims, base_point=base_point)
        
        posterior = self.posterior(xtt_vals, threshold=threshold)

        if len(plot_dims) == 2:
            posterior = torch.reshape(posterior, (100, 100))
        with torch.no_grad():
            plt.style.use(['science','no-latex'])
            if len(plot_dims) == 2:
                plt.contourf(xt1, xt2, posterior, cmap='jet')
                plt.xlabel(self.parameters[plot_dims[0]])
                plt.ylabel(self.parameters[plot_dims[1]])
            if len(plot_dims) == 1:
                plt.plot(xt1, posterior, 'k-')
                plt.xlabel(self.parameters[plot_dims[0]])
            plt.show()
    
    def posterior(self, x, threshold=-1):
        modelpred = self.model(x)
        normal = torch.distributions.normal.Normal(0,1)
        if threshold == -1:
            threshold = self.best_f
        internal_value = (-threshold + modelpred.mean)/modelpred.stddev
        output_value = normal.cdf(internal_value)
        return output_value

    def find_MAP(self, assume_near_the_bestf=True, niter=100):
        def fun(x):
            x = torch.tensor(x).unsqueeze(0)
            y = -float(self.posterior(x).detach())
            return y
        if assume_near_the_bestf:
            h = optimize.minimize(fun, self.best_f_loc.squeeze())
            minval = h.fun
            xmin = unnormalize(torch.tensor(h.x), 
                               torch.tensor(self.bounds).T)
        else:
            minval = 0
            xmin = 0
            for _ in range(niter):
                h = optimize.minimize(fun, np.random.rand(len(self.parameters)))
                if h.fun < minval:
                    minval = h.fun
                    xmin = unnormalize(torch.tensor(h.x), 
                           torch.tensor(self.bounds).T)
        maxval = -minval
        MAP = {'x':xmin, 'val':maxval}
        self.MAP = MAP

    # Posterior sampling and plotting ---------------------------------

    # This is the used default approach at the moment.
    # This is a bit of a heuristic method combining random sampling 
    # and using the collected result_dictionary. Intention is to be
    # efficient.
    def posterior_samples(self, add_rand=10000):
        x_output = torch.tensor(self.result_dictionary['inputs'])
        x_in = normalize(x_output, torch.tensor(self.bounds).T)
        x_in = x_in + 1e-4*torch.randn(len(x_in[:,0]), len(self.parameters))
        y_output = self.posterior(x_in)
        rand_vector = [0.001, 0.01, 0.1, 0.5]
        for i in rand_vector:
            x_o = torch.tensor(self.result_dictionary['inputs'])
            x_in = normalize(x_o, torch.tensor(self.bounds).T)
            x_in = x_in + i*torch.randn(len(x_in[:,0]), len(self.parameters))
            y_o = self.posterior(x_in)
            x_o = unnormalize(x_in, torch.tensor(self.bounds).T)
            x_output = torch.cat((x_output, x_o))
            y_output = torch.cat((y_output, y_o))
        x_output2 = torch.rand(add_rand, len(self.parameters))
        y_output2 = self.posterior(x_output2)
        y_output2 = y_output2        
        x_output2 = unnormalize(x_output2, torch.tensor(self.bounds).T)
        x_output = torch.cat((x_output, x_output2))
        y_output = torch.cat((y_output, y_output2))
        samples = {'x':x_output.detach().numpy(), 
                   'y':y_output.detach().numpy()}
        self.samples = samples

    # Direct MC sampling of the posterior. This is the preferred option
    # presently.
    def posterior_MC(self, nsamples=100000):
        x_output = torch.rand(nsamples, len(self.parameters))
        y_output = self.posterior(x_output)
        y_output = y_output.detach().numpy()         
        x_output = unnormalize(x_output, torch.tensor(self.bounds).T)
        samples = {'x':x_output, 'y':y_output}
        self.samples = samples 

    # An implemention of NUTS MCMC sampler using Pyro. This does not yet
    # work very well/efficiently.
    def posterior_MCMC(self, 
                       num_samples=1000, 
                       warmup_steps=300,
                       num_chains=8,
                       MC_start=True):
        def model():
            xprop = pyro.sample('x', 
                                pyro.distributions.Uniform(
                                    torch.zeros(len(self.parameters)),
                                    torch.ones(len(self.parameters))))
            y = self.posterior(xprop.unsqueeze(0))
            return y
        # Assume No-U-Turn kernel. More options can be added in future.
        init_params = None
        nuts_kernel = pyro.infer.mcmc.NUTS(model,
                                           adapt_step_size=True,
                                           target_accept_prob=0.95)
        mcmc = pyro.infer.mcmc.MCMC(nuts_kernel, 
                                    num_samples=num_samples, 
                                    warmup_steps=warmup_steps,
                                    num_chains=num_chains,
                                    initial_params=init_params)
        mcmc.run()
        y = self.posterior(mcmc.get_samples()['x']) 
        samples = unnormalize(mcmc.get_samples()['x'], 
                                 torch.tensor(self.bounds).T)
        outputdict = {'x':samples, 'y':y.detach().numpy()}
        self.samples = outputdict

    def corner_plot(self, smooth=1.5, point=[]):
        plt.style.use(['science','no-latex'])
        lower_array = []
        upper_array = []
        for i in range(len(self.parameters)):
            lower_array.append(self.bounds[i][0])
            upper_array.append(self.bounds[i][1])
        domain = [lower_array, upper_array]
        fig = plots.corner(self.samples['x'], 
                           self.samples['y'],
                           domain=domain, 
                           smooth=smooth, 
                           labels=self.plot_labels)
        if len(point) > 0:
            plots.mark_point(fig, point, color='black')
        self.cp_figure = fig

    
    def skip(self, index):
        raise NotImplementedError("skip not implemented for BayesianOptimizationSampler.")


