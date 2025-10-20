# from .base import Sampler
import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler
class ProvideListSampler(Sampler):
    """
    """
    def __init__(self, parameters, samples_lists, *args, **kwargs):
        """
        """
        self.parameters = parameters
        self.samples_lists = samples_lists    
        self.samples = self.generate_parameters()
        self.num_samples = len(self.samples)
        self.current_index = 0
        
        self.batch_size = kwargs.get('batch_size', self.num_samples)
        self.batch_number = 0
    def get_initial_samples(self, *args, **kwargs):
        """
        Gets the initial parameters.

        Returns:
            list[dict[str, float]]: The initial parameters.
        """
        samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.samples]
        self.submitted += len(samples)
        return samples

    def generate_parameters(self):
        """
        Generates the parameter combinations.

        Yields:
            list of float: The next parameter combination.
        """
        samples = np.array(self.samples_lists).T
        return samples
        
    def get_next_samples(self):
        """
        """
        if self.batch_number == 0:
            return self.get_initial_samples()
        else:
            raise NotImplementedError

    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
