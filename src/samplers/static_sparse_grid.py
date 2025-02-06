import sys, os
import sg_lib

from sg_lib.grid.grid import Grid #add . for relative import
from sg_lib.algebraic.multiindex import Multiindex #add . for relative import


from .base import Sampler
import numpy as np
from itertools import product
from common import S


class StaticSparseGrid(Sampler):
    """
    !!!!NEEDS UPDATED!!!!!!!!!!
    Creates a grid of equidistant points that spans bounds and num samples.
    i.e., (num_samples)**(len(parameters))

    Attributes:
        bounds (list of tuple of float): The bounds of each parameter.
        num_samples (list of int): The number of samples for each parameter.
        parameters (list of str): The names of the parameters.
        total_budget (int): The total number of parameter combinations.
        num_initial_points (int): The number of initial points in the sample space.
        samples (list of list of float): The generated parameter combinations.
        current_index (int): The index of the current parameter combination.
        sampler_interface (S): The type of sampler interface.

    This throws errors if you are asking for something insane,
        e.g., 10 parameters for 10 samples each -> 10 billion
        so hard limit at 100.000

    Methods:
        get_initial_parameters: Gets the initial parameters.
        generate_parameters: Generates the parameter combinations.
        get_next_parameter: Gets the next parameter combination.
    """

    sampler_interface = S.SEQUENTIAL

    def __init__(self, bounds, parameters, level:int, level_to_nodes:int=1):
        """
        Initializes the Grid sampler.

        Args:
            bounds (list of tuple of float): The bounds of each parameter.
            num_samples (list of int): The number of samples for each parameter.
            parameters (list of str): The names of the parameters.
        """
        self.level=level
        self.level_to_nodes = level_to_nodes
        self.dim=len(bounds)
        self.bounds=bounds
        self.left_stoch_boundary, self.right_stoch_boundary = zip(*self.bounds)
        self.left_stoch_boundary, self.right_stoch_boundary = np.array(self.left_stoch_boundary), np.array(self.right_stoch_boundary)
        ### setup for the standard uniform distribution, taken from ionuts config file
        self.weights = [lambda x: 1. for d in range(self.dim)]# for uniform dist
        self.left_bounds = np.zeros(self.dim)
        self.right_bounds = np.ones(self.dim)
        ## objects setup

        self.Grid_obj = Grid(self.dim, self.level, self.level_to_nodes, self.left_bounds, self.right_bounds, self.weights)	
        self.Multiindex_obj = Multiindex(self.dim)


        self.parameters = parameters
        self.num_samples = None
        self.samples, self.num_samples, self.samples_array, self.samples_array_norm = self.generate_parameters()
        self.num_initial_points = len(self.samples)
        self.current_index = 0
        
        # self.samples = list(self.generate_parameters())
        

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
        # samples = [
        #     np.linspace(self.bounds[i][0], self.bounds[i][1], self.num_samples[i])
        #     for i in range(len(self.bounds))
        # ]
        # # Use itertools.product to create a Cartesian product of sample
        # # points, representing the hypercube
        # for params_tuple in product(*samples):
        #     # Convert tuples to list to ensure serializability
        #     yield list(params_tuple)
        ## sparse grid multi-index set 
        multiindex_set = self.Multiindex_obj.get_std_total_degree_mindex(self.level)
        ## first, we get the grid points in [0, 1]^dim
        # std_sg_points = self.Grid_obj.get_std_sg_surplus_points(multiindex_set)
        std_sg_points = []
        for m, multiindex in enumerate(multiindex_set):
            mindex_grid_inputs = self.Grid_obj.get_sg_surplus_points_multiindex(multiindex)
            std_sg_points.append(mindex_grid_inputs)
        std_sg_points = np.concatenate(std_sg_points)

        print('POINTS SHAPE',std_sg_points.shape)
        num_samples = std_sg_points.shape[0]        
        self.num_samples = num_samples
        print("\033[1m no points for dim = {} and level = {} is n = {}\033[0m".format(self.dim, self.level, num_samples))
        ## we then map the grid points to our domain of interest
        # returns a list of lists, each sublist is a set of points
        mapped_sg_points = self.Grid_obj.map_std_sg_surplus_points(std_sg_points, self.left_stoch_boundary, self.right_stoch_boundary)
        
        #making a samples dictionary like this {param1:[list of values in points], param2:[list of values in points]}
        # the first value in each list makes 1 point, the second is the second point and so on. 
        # samples = {k:v for k,v in zip(self.parameters, [[] for _ in range(len(self.parameters))])}
        # for point in mapped_sg_points:
        #     i = 0
        #     for param in self.parameters:
        #         samples[param].append(point[i])
        #         i+=1
        # samples = {k:np.array(v) for k,v in samples.items()}
        return mapped_sg_points, num_samples, mapped_sg_points, std_sg_points

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
