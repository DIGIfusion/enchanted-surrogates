# samplers/base.py

from abc import ABC, abstractmethod
from typing import Union
from common import S


class Sampler(ABC):
    """ Parameter Space Samplers """
    # TODO: param_dict = {key: value for key, value in zip(self.parameters, params)} globally

    @abstractmethod
    def get_next_parameter(self) -> Union[list[dict[str, float]], dict[str, float]]:
        raise NotImplementedError('Sampling not implemented yet')

    @property 
    @abstractmethod
    def sampler_interface(self) -> S: 
        raise NotImplementedError('sampler_interface is not set')

    @abstractmethod 
    def get_initial_parameters(self,) -> list[dict[str, float]]: 
        raise NotImplementedError('Getting initial parameters is not implemented yet!')