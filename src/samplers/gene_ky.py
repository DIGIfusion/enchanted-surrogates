# sampler/grid.py

from .base import Sampler
import numpy as np
from itertools import product
from common import S

import samplers

class GENEky(Sampler):
    """
    When sampling ky in GENE you need to ensure you cover the ion scale and electron scale seperatly.
    Ion scale, small ky ~ 1, many instabilities
    Electron scale, large ky, only (ETG)
    
    This sampler takes in another sampler and applies ot over both scales.

    Attributes:
        
    Raises:

    Methods:
        get_initial_parameters: Gets the initial parameters.
        generate_parameters: Generates the parameter combinations.
        get_next_parameter: Gets the next parameter combination.
    """

    sampler_interface = S.SEQUENTIAL

    def __init__(self, ion_scale_bounds, electron_scale_bounds, num_ion_scale_samples, num_electron_scale_samples, sub_sampler, *args, **kwargs):
        """
        Initializes Sampler
        Args:
            sub_sampler: args: Arguments for a sub sampler. The arguments bounds, number of samples and parameters will be provided by GENEky sampler.
            If any additional parameters are needed the should be defined here.
            Must at least specify type.
        """
        self.sub_sampler_args = sub_sampler
        if self.sub_sampler_args == None: 
            raise ValueError('You must define a sub sampler.') 
                
        self.ion_scale_bounds = ion_scale_bounds
        self.electron_scale_bounds = electron_scale_bounds
        self.num_ion_scale_samples = num_ion_scale_samples
        self.num_electron_scale_samples = num_electron_scale_samples
        
        sampler_type = self.sub_sampler_args.pop('type') 
        
        self.sub_sampler_ions_class = getattr(samplers, sampler_type)
        self.sub_sampler_electrons_class = getattr(samplers, sampler_type)
        
        self.samples = self.generate_parameters()
        
        self.parameters = ['ky']
        self.num_samples = self.num_initial_points =  num_ion_scale_samples + num_electron_scale_samples
        self.current_index = 0

    def get_initial_parameters(
        self,
    ):
        """
        Gets the initial parameters.

        Returns:
            list[dict[str, float]]: The initial parameters.
        """
        # self.samples[:self.num_initial_points]
        return [self.get_next_parameter() for _ in range(self.num_initial_points)]

    def generate_parameters(self):
        """
        Generates the parameter combinations.

        Yields:
            list of float: The next parameter combination.
        """
        self.sub_sampler_ions = self.sub_sampler_ions_class(parameters=[('box','kymin')], bounds=[self.ion_scale_bounds], 
                                                            num_samples=self.num_ion_scale_samples, **self.sub_sampler_args)
        self.sub_sampler_electrons = self.sub_sampler_electrons_class(parameters=[('box','kymin')], bounds=[self.electron_scale_bounds], 
                                                                    num_samples=self.num_electron_scale_samples, **self.sub_sampler_args)
        return list(self.sub_sampler_ions.samples) + list(self.sub_sampler_electrons.samples)
    def get_next_parameter(self):
        """
        Gets the next parameter combination.

        Returns:
            dict or None: The next parameter combination, or None if all combinations have been used.
        """
        if self.current_index < len(self.samples):
            params = self.samples[self.current_index]
            self.current_index += 1
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            return param_dict
        else:
            return None  # TODO: implement when done iterating!
