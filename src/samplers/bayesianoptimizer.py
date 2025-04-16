"""
samplers/bayesianoptimizer.py

Text to describe the sampler.
"""


import os
import torch
import numpy as np
from common import S
import parsers
from .base import Sampler
import h5py 
try:
    import botorch
except ImportError as e:
    print(
        "Importing botorch failed.",
        "Please make sure that you have botorch installed",
    )

class BayesianOptimization(Sampler):
    """
    Bayesian Optimization sampler using the BoTorch library.

    Args:
        total_budget (int): Total budget for the optimization task.
        parser_kwargs (dict): Keyword arguments for the parser.
        model_kwargs (dict): Keyword arguments for the model.


    #TO BE UPDATED
        train_kwargs (dict): Keyword arguments for training.

    Attributes:
        total_budget (int): Total budget for the active learning process.
        bounds (list): Bounds for the parameter space.
        parameters (list): Names of the parameters.
        aquisition_batch_size (int): Batch size for acquisition.
        init_num_samples (int): Number of initial samples.
        selection_method (str): Method for selecting samples.
        kernel_transform (list): Kernel transformation methods.
        base_kernel (str): Base kernel for BMDAL.
        parser_kwargs (dict): Keyword arguments for the parser.
        model_kwargs (dict): Keyword arguments for the model.
        train_kwargs (dict): Keyword arguments for training.
        metrics (list): List to store metrics.
    """

    sampler_interface = S.BO

    def __init__(
        self,
        #total_budget: int,
        #parser_kwargs,
        #model_kwargs,
        #train_kwargs,
        *args,
        **kwargs,
    ):
        """ 
        - needs parser_kwargs, model_kwargs and train_kwargs (model training keyword arguments)
        Sets the following parameters from passed kwargs: 
        - set param: keyword argument 
        - acquisition_batch_size: aquisition_batch_size
        - bounds: bounds
        - init_num_samples: num_initial_points
        - selection_method: selection_method
        - kernel_transform: kernel_transform
        - base_kernel: base_kernel
        """

        # fmt: off
        self.total_budget          = kwargs.get('budget', 0)
        self.bounds                = kwargs.get('bounds', [None])
        self.observations          = kwargs.get('observations', [None])
        self.result_dictionary     = kwargs.get('result_dictionary', [None])
        self.result_dictionary_norm= kwargs.get('result_dictionary', [None])
        self.parameters            = kwargs.get('parameters', ['NA']*len(self.bounds))
        self.aquisition_batch_size = kwargs.get('aquisition_batch_size', 50)
        self.init_num_samples      = kwargs.get('num_initial_points', self.aquisition_batch_size)
        self.dry_run               = kwargs.get('dry_run', 'False')
        self.random_fraction       = kwargs.get('random_fraction', 0.5)
        self.async_samp            = kwargs.get('async_samp', 'False')
        #self.selection_method      = kwargs.get('selection_method', 'random')
        #self.kernel_transform      = kwargs.get('kernel_transform', [('rp', [512])])
        #self.base_kernel           = kwargs.get('base_kernel', 'll')
        self.parser_kwargs         = kwargs.get('parser_kwargs', [None])
        #self.model_kwargs          = model_kwargs
        #self.train_kwargs          = train_kwargs
        #self.metrics = []
        if self.parser_kwargs == [None]:
            self.parser            = getattr(parsers, kwargs.pop('parser'))()
        else:
            self.parser            = getattr(parsers, kwargs.pop('parser'))(**self.parser_kwargs)

    def get_initial_parameters(self):
        """
        Generates initial parameter samples.

        Returns:
            list: List of dictionaries containing initial parameter samples.

        """
        assert self.parameters[0] != "NA"
        batch_samples = []
        for _ in range(self.init_num_samples):
            params = [
                torch.distributions.Uniform(lb, ub).sample().item()
                for (lb, ub) in self.bounds
            ]
            param_dict = dict(zip(self.parameters, params))
            batch_samples.append(param_dict)
        return batch_samples

    def train_surrogate(self):
        # Presently implemented as single objective model. Therefore,
        # sum over the distances and normalize.
        distances = torch.from_numpy(np.sum(self.result_dictionary_norm['distances'],axis=1))
        distances = (distances - torch.mean(distances))/torch.std(distances)
        distances = distances.unsqueeze(distances.ndim)

        # Multiply by -1 the task to a maximization problem.
        distances = -distances

        gp = botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP(
            torch.from_numpy(self.result_dictionary_norm['inputs']), 
            distances)
        botorch.fit.fit_fully_bayesian_model_nuts(gp)
        self.model = gp
        self.best_f = torch.max(distances) 

    def get_next_parameter(self):
        """
        Generates the next parameter samples based on the model and the acquisition function. 

        Args:
            TBD 

        Returns:
            list: List of dictionaries containing next parameter samples.

        """
        EI = botorch.acquisition.qLogExpectedImprovement(model=self.model, best_f=self.best_f)
        if self.async_samp == True:
            if torch.rand(1) < self.random_fraction:
                batch_samples = []
                params = [
                    torch.distributions.Uniform(lb, ub).sample().item()
                    for (lb, ub) in self.bounds
                ]
                param_dict = dict(zip(self.parameters, params))
                batch_samples.append(param_dict)
                return batch_samples
            else:
                qval = 1
        else:
            batch_samples = []  
            qval = int((1 - self.random_fraction)*self.aquisition_batch_size)
            for _ in range(int(self.random_fraction*self.aquisition_batch_size)):
                params = [
                    torch.distributions.Uniform(lb, ub).sample().item()
                    for (lb, ub) in self.bounds
                ]
                param_dict = dict(zip(self.parameters, params))
                batch_samples.append(param_dict)
        candidates, acq_values = botorch.optim.optimize_acqf(EI, 
                                                             bounds=torch.FloatTensor(self.bounds).T,
                                                             sequential=False, 
                                                             q=qval,
                                                             num_restarts=10,
                                                             raw_samples=1024
                                                             )
        for index, _ in enumerate(range(candidates.size(dim=0))):
            params_dict = dict(zip(self.parameters, candidates[index,:].numpy()))
            batch_samples.append(params_dict)  
        return batch_samples
    
    def build_result_dictionary(self, base_run_directory: str, normalize=True): 
        """
        This function can be used to build the result_dictionary based on the existing runs
        in the base_run_directory.
        
        Args:
            base_run_directory (str): Path to the base run directory

        Kwargs:
            normalize: A Boolean flag to turn of normalization. This is True by default.

        Returns:
            This function does not return anything directly. However, the function 
            establishes self.result_dictionary as well as self.result_dictionary_norm,
            if normalize is set to True.         

        """

        # Make sure that result_dictionary does not exist in locals
        if 'result_dictionary' in locals():
            locals().pop('result_dictionary')
        if 'result_dictionary_failed' in locals():
            locals().pop('result_dictionary_failed')

        # Obtain a list of run_directories within the base_run_directory
        dirlist = os.listdir(base_run_directory)
        # Loop over the established runs. This can be streamlined if needed.
        for dirname in dirlist:
            if 'CONFIG' in dirname:
                continue
            else:
                sample_dict = self.parser.collect_sample_information(os.path.join(base_run_directory, dirname), self.observations)
            if sample_dict['failure'] == 0:
                if 'result_dictionary' in locals():
                    for key in sample_dict.keys():
                        # Append values to the lists corresponding to each key.
                        result_dictionary[key] = np.concatenate((result_dictionary[key], [sample_dict[key]])) 
                else:
                    result_dictionary = sample_dict
                    for key in result_dictionary.keys():
                        result_dictionary[key]=[result_dictionary[key]]
            else:
                if 'result_dictionary_failed' in locals():
                    for key in sample_dict.keys():
                        # Append values to the lists corresponding to each key.
                        result_dictionary_failed[key] = np.concatenate((result_dictionary_failed[key], [sample_dict[key]])) 
                else:
                    result_dictionary_failed = sample_dict
                    for key in result_dictionary.keys():
                        result_dictionary_failed[key]=[result_dictionary_failed[key]]
        if 'result_dictionary' in locals():
            self.result_dictionary = result_dictionary
        else:
            self.result_dictionary = [None]
        if 'result_dictionary_failed' in locals():
            self.result_dictionary_failed = result_dictionary_failed
        else:
            self.result_dictionary_failed = [None]
        if normalize:
            self.normalize_results()

    def append_result_dictionary(self, run_dir, normalize=True):
        sample_dict = self.parser.collect_sample_information(run_dir, self.observations)
        if sample_dict['failure'] == 0:
            for key in sample_dict.keys():
                self.result_dictionary[key] = np.concatenate((self.result_dictionary[key], [sample_dict[key]]))
            if normalize:
                self.normalize_results()
        else:
            for key in sample_dict.keys():
                if self.result_dictionary_failed == [None]:
                    self.result_dictionary_failed[key] = [sample_dict[key]]
                else:
                    self.result_dictionary_failed[key] = np.concatenate((self.result_dictionary_failed[key], [sample_dict[key]]))     

    def normalize_results(self):
        # Scale input domain to [0,1]**d
        if self.result_dictionary == [None]:
            self.result_dicionary_norm = [None]
        else:
            inpmin = np.min(self.result_dictionary['inputs'], axis=0)
            inpmax = np.max(self.result_dictionary['inputs'], axis=0)
            inprang = inpmax - inpmin
            input_scaled = (self.result_dictionary['inputs'] - inpmin)/inprang
            # Scale output N(0, 1)
            outputs = self.result_dictionary['distances']
            means = np.mean(outputs, axis=0)
            stds = np.std(outputs, axis=0)
            output_scaled = (outputs - means)/stds
            result_dictionary_norm = {'inputs':input_scaled, 'distances':output_scaled}
            self.result_dictionary_norm = result_dictionary_norm
                 







