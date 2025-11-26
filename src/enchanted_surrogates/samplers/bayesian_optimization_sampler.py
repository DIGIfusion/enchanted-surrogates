"""
samplers/bayesian_optimization_sampler.py

This sampler Class uses Bayesian Optimization techniques to data efficiently
sample through the search space to yield optimial information gain as 
specified by the acquisition strategy.
"""
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_parser
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import scienceplots

try:
    import botorch, gpytorch, torch
    from botorch.optim import optimize_acqf
except:
    print(
        "import botorch, gpytorch, torch failed.",
        "Please make sure that you have torch, botorch,",
        "and gpytorch installed",
    )


class BayesianOptimizationSampler(Sampler):
    """
    Bayesian Optimization sampler using the BoTorch library.

    Attributes:
        initial_samples (int):        Number of initial samples.
        verbose (bool):               True for detailed output.
        fully_bayesian (bool):        True for fully Bayesian models.
        acquisition_batch_size (int): Number of samples in each 
                                      acquisition batch.
        observations (list):          List of observations.
        bounds (list):                Bounds for the search space.
        acquisition_function (str):   Acquisition function to use.
        random_fraction (float):      Fraction of random samples.
        failure_prob_filter (bool):   True for using failure information
                                      to filter samples.
        ucb_beta (float):             Beta parameter for the UCB 
                                      acquisition function.
        async_samp (bool):            True for asynchronous sampling.
        parameters (list):            List of parameter names.
        parser (type):                Parser type for collecting 
                                      sample information.
        parser_config:                Parser kwargs
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initializes the BayesianOptimization sampler 
        with the given parameters.
        """
        print('INITIALISING BAYESIAN OPTIMIZATION SAMPLER')
        self._budget = kwargs.get('budget', 20)
        self.initial_samples = kwargs.get('initial_samples', 50)
        self.collected_samples = 0
        self.base_run_dir = kwargs.get('base_run_dir', '')
        self.verbose = kwargs.get('verbose', False)
        self.fully_bayesian = kwargs.get('fully_bayesian', False)
        self.acquisition_batch_size = kwargs.get('acquisition_batch_size', 20)
        self.observations = kwargs.get('observations', [None])
        self.bounds = kwargs.get('bounds', [None])
        self.acquisition_function = kwargs.get('acquisition_function', 'qLEI')
        self.random_fraction = kwargs.get('random_fraction', 0.2)
        self.failure_prob_filter = kwargs.get('failure_prob_filter', False)
        self.ucb_beta = kwargs.get('ucb_beta', 2.0)  
        self.async_samp = kwargs.get('async_samp', False)  
        self.parameters = kwargs.get('parameters', [])  
        self.parser_type = kwargs.get('parser', None)  
        self.parser_config = kwargs.get('parser_config',{})
        self.plot_GPR_flag = kwargs.get('plot_GPR', False)
        self.GPR_plot_dim = kwargs.get('GPR_plot_dim', [0])
        self.plot_GPR_file = kwargs.get('plot_file', False)
        self.plot_frequency = kwargs.get('plot_frequency', 1)
        self.plot_debug = kwargs.get('plot_debug', False)
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
            for _ in range(int(self.acquisition_batch_size)):
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
            if self.acquisition_function == 'qLEI':
                acq = botorch.acquisition.qLogExpectedImprovement(
                    model=self.model, 
                    best_f=self.best_f)
            elif self.acquisition_function == 'qUCB':
                acq = botorch.acquisition.qUpperConfidenceBound(
                    model=self.model, 
                    beta=self.ucb_beta)
            elif self.acquisition_function == 'qEI':
                acq = botorch.acquisition.qExpectedImprovement(
                    model=self.model, 
                    best_f=self.best_f)
            elif self.acquisition_function == 'qPI':
                acq = botorch.acquisition.qProbabilityOfImprovement(
                    model=self.model, 
                    best_f=self.best_f)
            elif self.acquisition_function == 'EI':
                acq = botorch.acquisition.ExpectedImprovement(
                    model=self.model,
                    best_f=self.best_f)
            elif self.acquisition_function == 'LEI':
                acq = botorch.acquisition.LogExpectedImprovement(
                    model=self.model,
                    best_f=self.best_f)
            if acq is None:
                raise ValueError("Unsupported acquisition function:", 
                                  f"{self.acquisition_function}")

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
                qval = int(acq_f*self.acquisition_batch_size)
                for _ in range(int(rand_f*self.acquisition_batch_size)):
                    params = [
                        torch.distributions.Uniform(lb, ub).sample().item()
                        for (lb, ub) in self.bounds
                    ]
                    param_dict = dict(zip(self.parameters, params))
                    batch_samples.append(param_dict)

            boundtensor = torch.DoubleTensor(self.bounds).T
            
            candidates, acq_values = optimize_acqf(
                acq, 
                bounds=boundtensor,
                sequential=False, 
                q=qval,
                num_restarts=10,
                raw_samples=1024
                )

            # This part of the code can be cleaned by implementing 
            # failure probability in the acquisition function. 
            if self.failure_prob_filter:
                if self.result_dictionary_failed != [None]:
                    cand_accept = []
                    not_enough = True
                    target_len = len(candidates[:,0])
                    while not_enough:
                        norm_inp = self.normalize_input(candidates)
                        pred = self.model_failed(norm_inp)
                        pred = pred.mean
                        for i in range(len(pred)):
                            if torch.rand(1) > pred[i]:
                                cand_accept.append(candidates[i,:].numpy())
                        if len(cand_accept) < target_len:
                            candidates, acq_values = optimize_acqf(
                                acq, 
                                bounds=torch.FloatTensor(self.bounds).T,
                                sequential=False, 
                                q=target_len - len(cand_accept),
                                num_restarts=10,
                                raw_samples=1024
                                )
                        else:
                            not_enough = False
                            candidates = np.array(cand_accept)
                            candidates = torch.tensor(candidates)
                
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
            print('FITTING THE GPR')
        # Presently implemented as single objective model. Therefore,
        # sum over the distances and norm
        distances = torch.from_numpy(np.sum(
                self.result_dictionary_norm['distances'][:],
                axis=1))
        distances = (distances - torch.mean(distances))/torch.std(distances)
        
        # Filter those parts of the result-dictionary that 
        # are outside the bounds.
        input_vector = self.result_dictionary_norm['inputs'][:]
        idx = np.where(np.max(np.abs(input_vector-0.5), axis=1)>0.5) 
        input_vector = torch.from_numpy(input_vector[idx])
        distances = distances[idx]
        distances = distances.unsqueeze(distances.ndim)
        # Multiply by -1 the task to a maximization problem.
        distances = -distances


        if self.fully_bayesian:
            if self.verbose:
                print('SaasFullyBayesianSingleTaskGP')
            gp = botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP(
                     input_vector, 
                     distances)
            botorch.fit.fit_fully_bayesian_model_nuts(gp)
        else:
            if self.verbose:
                print('SingleTaskGP')
            covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
            gp = botorch.models.SingleTaskGP(
                                             input_vector,
                                             distances, 
                                             covar_module=covar_module)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood=gp.likelihood, 
                model=gp)
            botorch.fit.fit_gpytorch_mll(mll)
        self.model = gp
        self.best_f = torch.max(distances)
        if self.result_dictionary_failed != [None]:

            inp = self.result_dictionary['inputs']
            inp_f = self.result_dictionary_failed['inputs']
            f_0 = self.result_dictionary['failure']
            f_1 = self.result_dictionary_failed['failure'][:].astype(float)

            gp_failed = botorch.models.SingleTaskGP(torch.from_numpy(
                self.normalize_input(np.concatenate([inp, inp_f]))),
                torch.from_numpy(np.concatenate([f_0, f_1])).unsqueeze(0).T)
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
                    'GPR']

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
                        result_dictionary[key] = np.concatenate(
                            (result_dictionary[key], 
                             [sample_dict[key]])) 
                else:
                    result_dictionary = sample_dict
                    for key in result_dictionary.keys():
                        result_dictionary[key]=[result_dictionary[key]]
            else:
                if result_dictionary_failed != None:
                    for key in sample_dict.keys():
                        # Append values to the lists corresponding to each 
                        # key.
                        result_dictionary_failed[key] = np.concatenate(
                            (result_dictionary_failed[key], 
                             [sample_dict[key]])) 
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
        if normalize:
            self.normalize_results()
        # Save the result_dictionary into a pickle file.
        if result_dictionary != None:
            res_dump = {'result_dictionary':result_dictionary,
                        'result_dictionary_failed':result_dictionary_failed}
            resdict = open(
                os.path.join(base_run_directory, 
                'result_dictionary.pkl'),'wb')
            pkl.dump(res_dump, resdict)
            resdict.close()

    def normalize_results(self):
        # Scale input domain to [0,1]**d
        if self.result_dictionary == [None]:
            self.result_dictionary_norm = [None]
        else:
	    ## Scale the inputs.
            input_scaled = self.normalize_input(self.result_dictionary['inputs'])
            # Scale output N(0, 1)
            outputs = self.result_dictionary['distances']
            means = np.mean(outputs, axis=0)
            stds = np.std(outputs, axis=0)
            if np.min(stds) > 0:
                output_scaled = (outputs - means)/stds
            else:
                output_scaled = outputs - means
            result_dictionary_norm = {'inputs':input_scaled, 
                                      'distances':output_scaled}
            self.result_dictionary_norm = result_dictionary_norm

    def normalize_input(self, x):
        # Scale the inputs based on the bounds of the search domain.
        inpmin = np.min(self.bounds, axis=1)
        inpmax = np.max(self.bounds, axis=1)
        inprang = inpmax - inpmin
        input_scaled = (x - inpmin)/inprang
        return input_scaled

    def denormalize_input(self, x):
        # Scale the inputs based on the bounds of the search domain.
        inpmin = np.min(self.bounds, axis=1)
        inpmax = np.max(self.bounds, axis=1)
        inprang = inpmax - inpmin
        input_scaled = x*inprang + inpmin
        return input_scaled

    # Plotting functionalities

    def plot_result_sequence(self):
        plt.plot(self.result_dictionary['distances'][:],'k.')
        plt.show()

    def plot_distances(self):
        """
        This is a helper function for plotting the sample distributions.
        The coding is not particularly elegant and can certainly be improved.
        """
        numdist = np.shape(self.result_dictionary['distances'])[1]
        numinp = np.shape(self.result_dictionary['inputs'])[1]
        ncols = int(numinp)
        nrows = int(numdist)
        if ncols == 1:
            if nrows == 1:
                plt.plot(self.result_dictionary['inputs'][:], 
                         self.result_dictionary['distances'][:],'ko')
                plt.xlabel(self.parameters[0])
            else:
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex='col')
                for i in range(numdist):
                    axs[int(i)].plot(
                        self.result_dictionary['inputs'][:], 
                        self.result_dictionary['distances'][:,i],'ko')
                    if i == numdist - 1:
                        axs[int(i)].set_xlabel(self.parameters[0])
        elif nrows == 1:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey='row')
            for j in range(numinp):
                axs[int(j)].plot(self.result_dictionary['inputs'][:,j], 
                                 self.result_dictionary['distances'][:],'ko')
                axs[int(j)].set_xlabel(self.parameters[j])
        else:
            fig, axs = plt.subplots(
                nrows=nrows, 
                ncols=ncols, 
                sharex='col', 
                sharey='row')
            for i in range(numdist):
                for j in range(numinp):
                    axs[int(i), int(j)].plot(
                        self.result_dictionary['inputs'][:,j], 
                        self.result_dictionary['distances'][:,i], 'ko')
                    if i == numdist - 1:
                        axs[int(i), int(j)].set_xlabel(self.parameters[j])
        plt.show()

    def plot_GPR(self, plot_dims=[0, 1], base_point = 0):
        """
        This is a helper function for plotting the GPR.
        """
        if base_point == 0:
            idx = np.where(
               self.result_dictionary['distances'] ==
               np.min(self.result_dictionary['distances']))
            idx = idx[0]
            base_point = self.result_dictionary['inputs'][idx]

        # GP is operating in [0,1]^d domain.
        base_point = base_point
        inpmin = np.min(self.bounds, axis=1)
        inpmax = np.max(self.bounds, axis=1)
        inprang = inpmax - inpmin
        input_scaled = (base_point - inpmin)/inprang
        # Establish the plotting axes.
        xtt_vals = torch.zeros((10000,len(self.bounds)))
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
            xtt_vals[:,plot_dims[0]] = torch.linspace(0,1,10000)

        for i in range(len(self.bounds)):
            if i not in plot_dims:
                xtt_vals[:,i] = torch.ones(10000)*input_scaled[0][i]

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
                    self.result_dictionary['inputs'][:,plot_dims[0]], 
                    self.result_dictionary_norm['distances'][:],'k.')
                plt.plot(xt1, -pred.mean, 'r-')
                lower, upper = pred.confidence_region()
                plt.fill_between(xt1, -lower, y2=-upper, alpha=0.5) 
                plt.xlabel(self.parameters[plot_dims[0]])
            if self.plot_GPR_file:
                plt.savefig(os.path.join(self.base_run_dir,
                            'GPR_'+\
                            str(len(self.result_dictionary['distances']))+'.svg'))
                plt.close()
            else:
                plt.show()

    def plot_posterior(self, plot_dims=[0,1], base_point=0, threshold=0):
        """
        This is a helper function for plotting the posterior.
        """
        if base_point == 0:
            idx = np.where(self.result_dictionary['distances'] == \
                           np.min(self.result_dictionary['distances']))
            idx = idx[0]
            base_point = self.result_dictionary['inputs'][idx]

        # GP is operating in [0,1]^d domain.
        base_point = base_point
        inpmin = np.min(self.bounds, axis=1)
        inpmax = np.max(self.bounds, axis=1)
        inprang = inpmax - inpmin
        input_scaled = (base_point - inpmin)/inprang
        # Establish the plotting axes.
        xtt_vals = torch.zeros((10000,len(self.bounds)))
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
            xtt_vals[:,plot_dims[0]] = torch.linspace(0,1,10000)

        for i in range(len(self.bounds)):
            if i not in plot_dims:
                xtt_vals[:,i] = torch.ones(10000)*input_scaled[0][i]
        
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
    
    def posterior(self, x, threshold=0):
        modelpred = self.model(x)
        normal = torch.distributions.normal.Normal(0,1)
        internal_value = (-threshold + modelpred.mean)/modelpred.stddev
        output_value = normal.cdf(internal_value)
        return output_value

