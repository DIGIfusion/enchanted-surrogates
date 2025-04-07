
from .base import Sampler
from common import S
from typing import Union
import torch


def get_parameter_distributions(
    bounds: list[list[Union[int, float]]], parameter_distributions: list[str]
) -> list[torch.distributions.distribution.Distribution]:
    """
    Generates parameter distributions based on the specified bounds and distribution types.

    Args:
        bounds (list of list of Union[int, float]): The bounds of each parameter.
        parameter_distributions (list of str): The names of the parameter distributions.

    Returns:
        list of torch.distributions.distribution.Distribution: The parameter distributions.
    """
    param_distributions = []
    for param_dist_string, bound in zip(parameter_distributions, bounds):
        lbound, upbound = bound
        param_dist = getattr(torch.distributions, param_dist_string)(lbound, upbound)
        param_distributions.append(param_dist)
    return param_distributions


class RandSampler(Sampler):
    """
    Generates random parameter samples sequentially.

    Attributes:
        sampler_interface (S): The type of sampler interface.
        parameters (list of str): The names of the parameters.
        bounds (list of list of Union[int, float]): The bounds of each parameter.
        num_initial_points (int): The number of initial points in the sample space.
        total_budget (int): The total number of parameter combinations.
        parameter_distributions (list of torch.distributions.distribution.Distribution): The parameter distributions.

    Args:
        bounds (list of list of Union[int, float]): The bounds of each parameter.
        num_samples (int): The number of samples to generate.
        parameters (list of str): The names of the parameters.
    """

    sampler_interface = S.SEQUENTIAL

    def __init__(self, bounds, num_samples, parameters):
        """
        Generates the next random parameter.

        Returns:
            dict: The next parameter combination.
        """
        self.parameters = parameters
        self.bounds = bounds
        self.num_initial_points = num_samples
        self.total_budget = num_samples
        parameter_strings = ["Uniform" for _ in range(len(bounds))]

        self.parameter_distributions = get_parameter_distributions(
            bounds, parameter_strings
        )

    def get_next_parameter(
        self,
    ):
        """
        Gets the initial random parameters.

        Returns:
            list of dict: The initial parameter combinations.
        """
        params = [dist.sample().item() for dist in self.parameter_distributions]
        param_dict = {key: value for key, value in zip(self.parameters, params)}
        return param_dict

    def get_initial_parameters(
        self,
    ):
        """
        Gets the initial random parameters.

        Returns:
            list of dict: The initial parameter combinations.
        """
        return [self.get_next_parameter() for _ in range(self.num_initial_points)]
