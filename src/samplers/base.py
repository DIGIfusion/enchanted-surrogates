# samplers/base.py

from abc import ABC, abstractmethod

class Sampler(ABC):
    """ Parameter Space Samplers """
    @abstractmethod
    def sample_parameters(self):
        pass