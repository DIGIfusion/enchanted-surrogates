from common import S
from .base import Sampler 


class ActiveLearner(Sampler): 
    sampler_interface = S.ACTIVE # NOTE: This could likely also be batch...
    def __init__(self, bounds, num_samples, parameters): 
        self.total_budget = num_samples 
        pass 
    
    
    def get_next_parameter(self):
        pass 

    def get_initial_parameters(self):
        # random samples 

        pass 