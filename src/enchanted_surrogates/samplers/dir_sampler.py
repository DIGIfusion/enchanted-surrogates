import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler
import warnings
import os

class DirSampler(Sampler):
    def __init__(self, parameter, root_dir, **kwargs):
        self.parameter = parameter
        self.root_dir = root_dir
        self.batch_number = 0        
    def get_next_samples(self) -> list[dict]:
        # TODO batch samples
        print('debug root_dir', self.root_dir)
        
        sub_dirs = [
            os.path.join(self.root_dir, sub_dir) for sub_dir in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, sub_dir))
        ]
        
        print('debug sub dirs', sub_dirs)
        if len(sub_dirs) == 0:
            raise FileNotFoundError('THE ROOT DIR HAS NO SUB DIRECTORIES INSIDE')
        samples = [{self.parameter: sub_dir} for sub_dir in sub_dirs]
                
        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None