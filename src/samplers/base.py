# samplers/base.py

from abc import ABC, abstractmethod
from typing import Union
from common import S


class Sampler(ABC):
    """
    Abstract base class for parameter space samplers.

    Attributes:
        sampler_interface (S): The type of sampler interface.
    """

    # TODO: param_dict = {key: value for key, value in zip(self.parameters, params)} globally

    @abstractmethod
    def get_next_parameter(
        self, *args, **kwargs
    ) -> Union[list[dict[str, float]], dict[str, float]]:
        """
        Abstract method for getting the next parameter set.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[list[dict[str, float]], dict[str, float]]: The next parameter set.
        """
        raise NotImplementedError("Sampling not implemented yet")

    @property
    @abstractmethod
    def sampler_interface(self) -> S:
        """
        Abstract property for getting the sampler interface type.

        Returns:
            S: The sampler interface type.
        """
        raise NotImplementedError("sampler_interface is not set")

    @abstractmethod
    def get_initial_parameters(
        self,
    ) -> list[dict[str, float]]:
        """
        Abstract method for getting the initial parameters.

        Returns:
            list[dict[str, float]]: The initial parameters.
        """
        raise NotImplementedError("Getting initial parameters is not implemented yet!")
