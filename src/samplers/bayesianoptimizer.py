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
        self.parameters            = kwargs.get('parameters', ['NA']*len(self.bounds))
        self.aquisition_batch_size = kwargs.get('aquisition_batch_size', 50)
        self.init_num_samples      = kwargs.get('num_initial_points', self.aquisition_batch_size)
        #self.selection_method      = kwargs.get('selection_method', 'random')
        #self.kernel_transform      = kwargs.get('kernel_transform', [('rp', [512])])
        #self.base_kernel           = kwargs.get('base_kernel', 'll')
        #self.parser_kwargs         = parser_kwargs
        #self.model_kwargs          = model_kwargs
        #self.train_kwargs          = train_kwargs
        #self.metrics = []

        self.parser                = getattr(parsers, parser_kwargs.pop('type'))(**parser_kwargs)

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

    def get_next_parameter(
        self,
        model: torch.nn.Module,
        train: torch.Tensor,
        pool: torch.Tensor,
        *args,
        **kwargs,
    ) -> list[dict[str, float]]:
        """
        Generates the next parameter samples based on the model predictions given the train and pool.
        NOTE: The train and pool are expected to be normalised, as model and BMDAL expect data to be normalized. 

        Args:
            model (torch.nn.Module): PyTorch model.
            train (torch.Tensor): 
            pool (torch.Tensor):  

        Returns:
            list: List of dictionaries containing next parameter samples.

        """
        # train, pool = self.parser.train, self.parser.pool
        new_idxs = self._get_new_idxs_from_pool(model, train, pool)
        results = []
        for idx in new_idxs:
            results.append(
                {
                    "input": self.parser.pool[idx, self.parser.input_col_idxs],
                    "output": None,
                    "pool_idxs": idx,
                }
            )
        # TODO: convert indicies to unscaled parameters
        return results
    
    def build_result_dictionary(self, base_run_directory: str): #-> torch.Tensor:
        """
        This function can be used to build the result_dictionary based on the existing runs
        in the base_run_directory.
        """

        # Make sure that result_dictionary does not exist in locals
        if 'result_dictionary' in locals():
            locals().pop('result_dictionary')

        # Obtain a list of run_directories within the base_run_directory
        dirlist = os.listdir(base_run_directory)
      
        # Loop over the established runs. This can be streamlined if needed.
        for dirname in dirlist:
            if 'CONFIG' in dirname:
                continue
            else:
                sample_dict = self.parser.collect_sample_information(os.path.join(base_run_directory, dirname), 
                                                                     self.observations)
            if 'result_dictionary' in locals():
                for key in sample_dict.keys():
                    # Append values to the lists corresponding to each key.
                    result_dictionary[key] = np.concatenate((result_dictionary[key], [sample_dict[key]])) 
            else:
                result_dictionary = sample_dict
                for key in result_dictionary.keys():
                    # Change to a list, such that results can be appended.
                    #if len(result_dictionary[key]) < 2:
                    #    result_dictionary[key] = [result_dictionary[key]]
                    result_dictionary[key]=[result_dictionary[key]]
        if 'result_dictionary' in locals():
            self.result_dictionary = result_dictionary
        else:
            self.result_dictionary = [None]                    

    def collect_batch_results(self, results: list[dict[str, dict]]) -> torch.Tensor:
        """
        Collects results from the batch.

        Args:
            results (list): List of dictionaries containing batch results.

        Returns:
            torch.Tensor: Tensor containing batch results.

        """
        outputs_as_tensor = torch.empty(len(results), len(self.parser.keep_keys))
        for n, result in enumerate(results):
            x = result["inputs"]
            y = result["output"]
            outputs_as_tensor[n] = torch.cat((x, y))
        return outputs_as_tensor


    def dump_iteration_results(self, base_run_dir: str, iterations: int, trained_model_state_dict: dict):
        """ 
        iteration number, size of train set, and metrics get appened to the file 'final_metrics.txt'
        while the current train set is saved  "train_{iterations}.pth" # TODO: update train set saving
        model is saved every 5 iterations 
        """

        
        train_fname = os.path.join(base_run_dir, f"train_{iterations}.pth")
        torch.save(self.parser.train, train_fname)
        

        final_metric_file = os.path.join(base_run_dir, "final_metrics.txt")
        with open(final_metric_file, "a") as f:
            f.write(f"{iterations}, {len(self.parser.train)}, {self.metrics[-1]}\n")
        
        model_path = os.path.join(base_run_dir, f'model_{iterations}.pth')
        if iterations % 5 == 0: 
            torch.save(trained_model_state_dict, model_path)


