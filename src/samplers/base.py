# samplers/base.py

from abc import ABC, abstractmethod
from typing import Dict, Union
class Sampler(ABC):
    """ Parameter Space Samplers """
    """
    @abstractmethod
    def sample_parameters(self):
        raise NotImplementedError('Sampling not implemented yet')
        pass
    """
    @abstractmethod
    def get_next_parameter(self)-> Dict[str, float]:
        raise NotImplementedError('Sampling not implemented yet')
        pass 