# sampler/static_sparse_grid.py

from os import sys
from os import getcwd
from os import path

try:
    from base import Sampler # add .
    from static_sparse_grid_approximations.sg_lib.grid.grid import Grid #add . for relative import
    from static_sparse_grid_approximations.sg_lib.algebraic.multiindex import Multiindex #add . for relative import

except:
    try:
        from .base import Sampler # add . for relative import
        from .static_sparse_grid_approximations.sg_lib.grid.grid import Grid #add . for relative import
        from .static_sparse_grid_approximations.sg_lib.algebraic.multiindex import Multiindex #add . for relative import

    except:
        raise ImportError


import numpy as np
from itertools import product

class StaticSparseGrid(Sampler):
    """
    Creates a grid of points that aims to evenly spread the interpolation error and select 
    enough points to train a legendre high dimensional polynomial interpolator of selected
    order.

    It takes the specified range for each parameter and the order of the polynomial to be 
    used for interpolation in the downstream training.
    """
    def __init__(self, bounds, parameters, level:int, level_to_nodes:int=1):
        self.level=level
        self.level_to_nodes = level_to_nodes
        self.dim=len(bounds)
        self.bounds=bounds
        self.left_stoch_boundary, self.right_stoch_boundary = zip(*self.bounds)
        ### setup for the standard uniform distribution, taken from ionuts config file
        self.weights = [lambda x: 1. for d in range(self.dim)]
        self.left_bounds = np.zeros(self.dim)
        self.right_bounds = np.ones(self.dim)
        ## objects setup
        self.Grid_obj = Grid(self.dim, self.level, self.level_to_nodes, self.left_bounds, self.right_bounds, self.weights)	
        self.Multiindex_obj = Multiindex(self.dim)


        self.parameters = parameters
        self.samples, self.num_samples = self.generate_parameters()
        self.current_index = 0

    def generate_parameters(self):
        ## sparse grid multi-index set 
        multiindex_set = self.Multiindex_obj.get_std_total_degree_mindex(self.level)
        ## first, we get the grid points in [0, 1]^dim
        std_sg_points = self.Grid_obj.get_std_sg_surplus_points(multiindex_set)
        num_samples = std_sg_points.shape[0]
        print("\033[1m no points for dim = {} and level = {} is n = {}\033[0m".format(self.dim, self.level, num_samples))
        ## we then map the grid points to our domain of interest
        # returns a list of lists, each sublist is a set of points
        mapped_sg_points = self.Grid_obj.map_std_sg_surplus_points(std_sg_points, self.left_stoch_boundary, self.right_stoch_boundary)
        
        #making a samples dictionary like this {param1:[list of values in points], param2:[list of values in points]}
        # the first value in each list makes 1 point, the second is the second point and so on. 
        samples = {k:v for k,v in zip(self.parameters, [[] for _ in range(len(self.parameters))])}
        for point in mapped_sg_points:
            i = 0
            for param in parameters:
                samples[param].append(point[i])
                i+=1
        
        print('SAMPLES',samples)
        return samples, num_samples

    def get_next_parameter(self):
        if self.current_index < len(self.samples):
            param_dict = {k : v[self.current_index] for k,v in self.samples.items()}
            self.current_index += 1
            return param_dict
        else:
            return None  # TODO: implement when done iterating!

if __name__ == '__main__':
    parameters = ['love', 'peace', 'harmony']
    bounds = [(1,2),(300,400),(5000,6000)]
    samp = StaticSparseGrid(bounds, parameters, level=3)
    # print(samp.samples)
    
    # for _ in range(samp.num_samples):
    #     print(samp.get_next_parameter())