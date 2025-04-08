import numpy as np
import warnings
import os
from common import S
import pysgpp
import samplers
from pysgpp import BoundingBox1D
from scipy.stats import sobol_indices, uniform, entropy
import matplotlib.pyplot as plt
import pandas as pd

from runners.MMMGrunner import MaxOfManyGaussians

class SpatiallyAdaptiveSparseGrids:
    def __init__(self, bounds, parameters, poly_basis_degree=3, initial_level=3, infer_bounds=False, infer_parents=False, **kwargs):
        self.sampler_interface = S.ACTIVE
        self.bounds = np.array(bounds)
        self.parameters = parameters
        if type(parameters[0]) == type([]):
            self.parameters = [tuple(pa) for pa in parameters]
        self.poly_basis_degree= poly_basis_degree
        self.initial_level = initial_level
        self.infer_bounds = infer_bounds
        self.infer_parents = infer_parents
        
        if len(bounds)==2:
            self.mmg = MaxOfManyGaussians(2, bounds)
            std=0.5
            self.mmg.specify_gaussians(means=np.array([[0.25,0.25], [0.75,0.75]]), stds = np.array([[std,std],[std,std]]))
        
        self.custom_limit = np.inf
        self.custom_limit_value = 0
        self.dim=len(parameters)
        
        # The test sampler is used to fill the space and run the surrogate model so that we can calcualte the integral and sobel indicies to check for convergence in UQ situations.
        check_sampler_kwargs = kwargs['check_sampler']
        check_sampler_kwargs['parameters'] = self.parameters
        check_sampler_kwargs['bounds'] = self.bounds
        self.check_sampler = getattr(samplers, check_sampler_kwargs['type'])(**check_sampler_kwargs)
        self.old_check_predictions = None
        self.num_active_cycles = 0
        self.num_samples_by_cycle = []
        
        # Specific to UQ samplers
        self.expectation_by_cycle, self.double_sigma_by_cycle, self.sobol_first_order_by_cycle= [],[],[]
        self.sobol_total_order_by_cycle, self.entropy_diff_by_cycle = [],[]
        
        self.bounds_points = {}
        self.parent_points = {}
        
        self.all_cycle_info = []
        # make uniform distributions for each input, this is used later to calculate sobel indicies for UQ
        self.dists = []
        for b in self.bounds:
            assert b[1] > b[0]
            self.dists.append(uniform(loc=b[0], scale=b[1]-b[0]))
    
    def point_transform_box2unit(self, box_point):
        # min max normalisation
        unit_point = tuple()
        for i, bco in enumerate(box_point):
            min = self.bounds[i][0]
            max = self.bounds[i][1]
            assert max > min
            uco = (bco - min) / (max - min)
            unit_point = unit_point + (uco,)
        return unit_point

    def point_transform_unit2box(self, unit_point):
        # min max normalisation
        box_point = tuple()
        for i, uco in enumerate(unit_point):
            min = self.bounds[i][0]
            max = self.bounds[i][1]
            assert max > min
            bco = uco*(max - min) + min
            box_point = box_point + (bco,)
        return box_point
    
    def points_transform_unit2box(self, unit_points):
        unit_points = np.array(unit_points)
        return unit_points * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    
    def points_transform_box2unit(self, box_points):
        return (box_points - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def get_initial_parameters(self):
        # returns the initial set of parameters for initial trianing of surrogate model
        self.grid = pysgpp.Grid.createPolyBoundaryGrid(self.dim, self.poly_basis_degree)
        self.gridStorage = self.grid.getStorage()
        self.gridGen = self.grid.getGenerator()
        self.gridGen.regular(self.initial_level)
        # array containing labels in order they come from;- for i in range(gridStorage.getSize()): gp = gridStorage.getPoint(i)
        self.alpha = pysgpp.DataVector(self.gridStorage.getSize())
        
        batch_samples = []
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = tuple()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
            box_point = self.point_transform_unit2box(unit_point)
        
            param_dict = {}
            for j, param in enumerate(self.parameters):
                param_dict[param]=box_point[j]
        
        
            batch_samples.append(param_dict)
            
        self.num_samples_by_cycle.append(len(batch_samples))        
        return batch_samples
    
    def get_next_parameters(
        self,
        train: dict,
        cycle_dir:str=None,
        *args,
        **kwargs) -> list[dict[str, float]]:
        # returns a list of parameters that are the next batch to be labeled, based on the training samples, models used and selection criteria.
        """
        args
        train, dict: A dictionary where the keys are a tuple of inputs (x0,x1,x2) (*in the order of the origional parameters) and the value is a label
        """
        # 
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)    
            unit_point = ()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
            box_point = self.point_transform_unit2box(unit_point) 
                        
            if box_point in train.keys():
                self.alpha[i] = train[box_point]
            elif box_point in self.bounds_points.keys():
                self.alpha[i] = self.bounds_points[box_point]
            elif box_point in self.parent_points.keys():
                self.alpha[i] = self.parent_points[box_point] 
        
        
        pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(self.alpha)
        # Now the surrogate is trained we can write cycle info
        if cycle_dir!=None:
            print('WRITING CYCLE INFO')
            self.write_cycle_info(cycle_dir)
        
        number_of_points_to_refine = 1
        print('grid size before refinement:', self.gridStorage.getSize())
        self.gridGen.refine(pysgpp.SurplusRefinementFunctor(self.alpha, number_of_points_to_refine))
        print('grid size after refinement:', self.gridStorage.getSize())
        # print("refinement step {}, new grid size: {}".format(refnum+1, gridStorage.getSize()))  
        self.alpha.resizeZero(self.gridStorage.getSize())
        
        batch_samples = []
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = ()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
            
            box_point = self.point_transform_unit2box(unit_point)

            if box_point not in train.keys() and box_point not in self.bounds_points and box_point not in self.parent_points:
                print('NEW POINT',box_point, unit_point)
                print('IS BOUNDARY', not gp.isInnerPoint())
                print('is parent', not gp.isLeaf())
                
                param_dict = {}
                for j, param in enumerate(self.parameters):
                    param_dict[param]=box_point[j]
                
                if self.infer_bounds and not gp.isInnerPoint() or self.infer_parents and not gp.isLeaf():
                    # If it is both then default to boundary point
                    if self.infer_bounds and not gp.isInnerPoint():
                        print('appending bounds points')
                        self.bounds_points[box_point] = self.surrogate_predict([box_point])[0]
                    elif self.infer_parents and not gp.isLeaf():
                        self.parent_points[box_point] = self.surrogate_predict([box_point])[0]
                        print('appending parent points')

                else:
                    print('adding to batch samples')
                    batch_samples.append(param_dict)
        
        self.num_active_cycles += 1
        self.num_samples_by_cycle.append(len(train)+len(batch_samples))
        print('len batch_samples',len(batch_samples))
        return batch_samples
    
    def update_custom_limit_value(self): # Necessary
        NotImplemented
    
    # currently only works with point in the unit cube.     
    def surrogate_predict(self, box_points):
        unit_points = self.points_transform_box2unit(np.array(box_points))
        unit_points_dm = pysgpp.DataMatrix(unit_points)
        opEval = pysgpp.createOperationMultipleEval(self.grid, unit_points_dm)
        results = pysgpp.DataVector(len(unit_points))
        opEval.eval(self.alpha, results)
        ans = np.array(results.array()) # sometimes this line breaks python. 
        # ans = np.array([results.get(i) for i in range(len(positions))])
        return ans
    
    def integral_approx(self, function_values):
        bounds_array = np.array(self.bounds).T
        space_volume = np.prod(bounds_array[1]-bounds_array[0])
        integral_estimate = np.mean(function_values) * space_volume
        return integral_estimate
    
    # Assumes the inputs used to get the function values were sampled from their uncertainty distributions.
    def expectation_approx(self, function_values, num_bins=200):
        heights, edges = np.histogram(function_values, bins=num_bins)
        weights = heights/sum(heights)
        centers = (edges[:-1] + edges[1:]) / 2
        expectation = np.average(centers,weights=weights)
        return expectation
    
    def confidance_interval(self, function_values, percentage, num_bins=200):
        decimal = percentage/100
        heights, edges = np.histogram(function_values, bins=num_bins, density=True)
        width = edges[1]-edges[0]
        centers = (edges[:-1] + edges[1:]) / 2
        area = 0
        area_limit = (1-decimal)/2
        i = 0
        while area<area_limit:
            area += heights[i]*width
            i+=1
        lower_interval = centers[i]
        
        area = 0
        i = 0
        while area<area_limit:
            area += heights[-i]*width
            i+=1
        upper_interval = centers[-i]
        return lower_interval, upper_interval
        
    def double_sigma_approx(self, function_values):
        double_sigma_estimate = 2*(np.sqrt(self.expectation_approx(function_values**2) - self.expectation_approx(function_values)**2))
        return double_sigma_estimate
    
    def sobel_indicies_approx(self):
        func = lambda positions: self.surrogate_predict(positions.T)
        sobol = sobol_indices(func=func,  n=2**np.log2(self.check_sampler.num_samples), dists=self.dists)
        return sobol.first_order, sobol.total_order
    
    def relative_entropy(self, samples_p, samples_q, num_bins=200):
        """
        Calculate the relative entropy (Kullback-Leibler divergence) between two distributions.

        Parameters:
        samples_p (list or np.array): Samples from the first distribution.
        samples_q (list or np.array): Samples from the second distribution.

        Returns:
        float: The relative entropy between the two distributions.
        """
        if type(samples_p) == type(None) or type(samples_q) == type(None):
            return np.nan
        # Convert samples to numpy arrays
        samples_p = np.array(samples_p)
        samples_q = np.array(samples_q)
        # Calculate the probability density functions
        p_values, _ = np.histogram(samples_p, bins=num_bins, density=True)
        q_values, _ = np.histogram(samples_q, bins=num_bins, density=True)
        # Add a small value to avoid division by zero and log of zero
        p_values += 1e-10
        q_values += 1e-10
        # Calculate the relative entropy
        rel_entropy = entropy(p_values, q_values)
        return rel_entropy
    
    def write_cycle_info(self, cycle_dir): #Necessary
        print(f'+++ \n Write ACTIVE CYCLE: {self.num_active_cycles}')
        print('+++ \n NUM INNER LEAF POINTS', self.num_samples_by_cycle[-1])            
        print('+++ \n NUM PARENT POINTS', len(self.parent_points))
        print('+++ \n NUM bounds POINTS', len(self.bounds_points))
        
        predictions = self.surrogate_predict(self.check_sampler.samples)
        expectation = self.expectation_approx(predictions)
        double_sigma = self.double_sigma_approx(predictions)
        sobol_first_order, sobol_total_order = self.sobel_indicies_approx()
        entropy_diff = self.relative_entropy(self.old_check_predictions, predictions, num_bins=200)
        self.old_check_predictions=predictions
        
        df = pd.DataFrame({"approx_expectation":[expectation], "approx_2sigma":[double_sigma], "approx_entropy_diff":[entropy_diff], "num_samples":[self.num_samples_by_cycle[-1]], "num_parents":[len(self.parent_points)], "num_bounds":[len(self.bounds_points)]})
        for d, sfo in enumerate(sobol_first_order):
            df[f'sobol_first_order_{d}'] = [sfo]
        for d, sto in enumerate(sobol_total_order):
            df[f'sobol_total_order_{d}']= [sto]
        df.to_csv(os.path.join(cycle_dir,'cycle_info.csv'))
        
        self.all_cycle_info.append(df)  
        
        self.expectation_by_cycle.append(expectation), self.double_sigma_by_cycle.append(double_sigma), self.sobol_first_order_by_cycle.append(sobol_first_order)
        self.sobol_total_order_by_cycle.append(sobol_total_order), self.entropy_diff_by_cycle.append(entropy_diff)
        del predictions
    
    def write_summary(self, base_dir):
        if len(self.bounds) == 2:
            x0 = []
            x1 = []
            x0_inner = []
            x1_inner = []
            for i in range(self.gridStorage.getSize()):
                gp = self.gridStorage.getPoint(i)
                
                if gp.isInnerPoint():
                    x0_inner.append(gp.getStandardCoordinate(0))
                    x1_inner.append(gp.getStandardCoordinate(1))
                
                x0.append(gp.getStandardCoordinate(0))
                x1.append(gp.getStandardCoordinate(1))    
            fig, ax = plt.subplots(1,1, figsize=(9,9))
            ax.set_aspect(1)
            # self.mmg.plot_2d_gaussians(ax)
            ax.scatter(x0, x1)
            ax.scatter(x0_inner, x1_inner, color='black')
            fig.savefig(fname=os.path.join(base_dir,'grid'), dpi=300)
        
        # with open(os.path.join(base_dir, 'all_sampler_cycle_info'), 'w') as file:
        #     for cycle_info in self.all_cycle_info:
        #         file.write(cycle_info)
        df = pd.concat(self.all_cycle_info, ignore_index=True)
        df.to_csv(os.path.join(base_dir, 'all_cycle_info.csv'))
                        
        predictions = self.surrogate_predict(self.check_sampler.samples)
        fig = plt.figure()
        bars = plt.hist(predictions, bins=200, density=True)
        heights, edges = bars[0], bars[1]
        plt.vlines([self.expectation_approx(predictions)],0, max(heights),color='red', label=r'Expectation and 95% confidance interval')
        lower_interval, upper_interval = self.confidance_interval(predictions,percentage=95)
        plt.hlines(0, lower_interval, upper_interval, color='red', linewidth=3)
        plt.xlabel('function output')
        plt.ylabel('probability density')
        plt.title(f'Output Distribution, {self.check_sampler.num_samples} samples')
        plt.legend()
        fig.savefig(os.path.join(base_dir, 'output_distribution'), dpi=400)
        hist_data_heights = pd.DataFrame({'heights': heights})
        hist_data_heights.to_csv(os.path.join(base_dir, 'output_distribution_hist_data_heights.csv'), index=False)
        hist_data_edges = pd.DataFrame({'edges': edges})
        hist_data_edges.to_csv(os.path.join(base_dir, 'output_distribution_hist_data_edges.csv'), index=False)
        
        fig = plt.figure()
        plt.plot(np.array(self.num_samples_by_cycle), np.array(self.expectation_by_cycle))
        plt.fill_between(self.num_samples_by_cycle, np.array(self.expectation_by_cycle)-np.array(self.double_sigma_by_cycle),np.array(self.expectation_by_cycle)+np.array(self.double_sigma_by_cycle), color='grey', label='2sigma')
        plt.ylabel('approx_expectation')
        plt.xlabel('Number of Parent Function Evaluations')
        plt.legend()
        fig.savefig(os.path.join(base_dir,'approx_expectation'), dpi=300)
        expectation_data = pd.DataFrame({'num_samples': self.num_samples_by_cycle, 'expectation': self.expectation_by_cycle, 'double_sigma': self.double_sigma_by_cycle})
        expectation_data.to_csv(os.path.join(base_dir, 'approx_expectation_data.csv'), index=False)
        
        fig = plt.figure()
        self.sobol_first_order_by_cycle = np.array(self.sobol_first_order_by_cycle)
        sobol_first_order_data = pd.DataFrame({'num_samples': self.num_samples_by_cycle})
        for i, dimension in enumerate(self.sobol_first_order_by_cycle.T):
            sobol_first_order_data[f'sobol_first_order_{i}'] = dimension
            plt.plot(self.num_samples_by_cycle, dimension)
            plt.ylabel('First order sobol indicie')
            plt.xlabel('Number of Parent Function Evaluations')
            fig.savefig(os.path.join(base_dir, 'approx_sobol_first_order'),dpi=300)
        sobol_first_order_data.to_csv(os.path.join(base_dir, 'approx_sobol_first_order_data.csv'), index=False)
        
        fig = plt.figure()
        self.sobol_total_order_by_cycle = np.array(self.sobol_total_order_by_cycle)
        sobol_total_order_data = pd.DataFrame({'num_samples': self.num_samples_by_cycle})
        for i, dimension in enumerate(self.sobol_total_order_by_cycle.T):
            sobol_total_order_data[f'sobol_total_order_{i}'] = dimension
            plt.plot(self.num_samples_by_cycle, dimension)
            plt.ylabel('Total order sobol indicie')
            plt.xlabel('Number of Parent Function Evaluations')
            fig.savefig(os.path.join(base_dir, 'approx_sobol_total_order'),dpi=300)
        sobol_total_order_data.to_csv(os.path.join(base_dir, 'approx_sobol_total_order_data.csv'), index=False)
        
        fig = plt.figure()
        self.entropy_diff_by_cycle = np.array(self.entropy_diff_by_cycle)
        entropy_diff_data = pd.DataFrame({'num_samples': self.num_samples_by_cycle, 'entropy_diff': self.entropy_diff_by_cycle})
        plt.plot(self.num_samples_by_cycle, self.entropy_diff_by_cycle)
        plt.ylabel('Entropy Difference')
        plt.xlabel('Number of Parent Function Evaluations')
        fig.savefig(os.path.join(base_dir, 'approx_entropy_diff'),dpi=300)
        entropy_diff_data.to_csv(os.path.join(base_dir, 'approx_entropy_diff_data.csv'), index=False)
                
    def check_UQ_convergence():
        None
# # import pysgpp library
# import pysgpp
# import inspect
# f = lambda x0, x1: mmg.evaluate([x0, x1])
# # f = lambda x0, x1: 16.0 * (x0-1)*x0 * (x1-1)*x1*x1

# poly_basis_degree = 3
# grid = pysgpp.Grid.createPolyBoundaryGrid(dim, poly_basis_degree)
# # grid = pysgpp.Grid.createLinearBoundaryGrid(dim)

# gridStorage = grid.getStorage()

# # create regular sparse grid, level 3
# initial_level = 3
# gridGen = grid.getGenerator()

# gridGen.regular(initial_level)

# print("number of initial grid points:    {}".format(gridStorage.getSize()))

# alpha = pysgpp.DataVector(gridStorage.getSize())
# print("length of alpha vector:           {}".format(alpha.getSize()))
# # Obtain function values and refine adaptively 5 times
# num_refinement_steps =100

# x0, x1, x0_leaf, x1_leaf = [], [], [], []

# #We don't want to run the function for every point so a wrapper function should check to see if the point has been ran and if it has return that value
# def dummy_runner(f, samples, labeled_samples=None):
#     if type(labeled_samples) != type(None):
#         # print('LABELED SAMPLES',labeled_samples.items())
#         for k, v in samples.items():
#             if k in labeled_samples.keys():
#                 samples[k] = labeled_samples[k]
#             else:
#                 samples[k] = f(*k)
#     else:
#         for k, v in samples.items():
#             samples[k] = f(*k)
#     return samples
            
# labeled_samples=None
# samples = {}
# # I want to know if there are any samples that have never been a leaf
# was_leaf = []

# samples_s = []
# is_leaf_s = []
# for refnum in range(num_refinement_steps):
#     print('REFINEMENT STEP', refnum+1)
#     # make samples dict
#     for i in range(gridStorage.getSize()):
#         gp = gridStorage.getPoint(i)
#         x0 = gp.getStandardCoordinate(0)
#         x1 = gp.getStandardCoordinate(1)
#         samples[(x0,x1)] = None
#         if i > len(was_leaf):
#             # print('APPENDING')
#             was_leaf.append(False)
#     was_leaf.append(False)
    
#     # label samples
#     samples = dummy_runner(f, samples, labeled_samples)
#     samples_s.append(samples.copy())
#     labeled_samples = samples.copy()
    
#     # set function values in alpha
#     is_leaf = []
#     for i in range(gridStorage.getSize()):
#         gp = gridStorage.getPoint(i)
#         # print(dir(gp))
#         # break
#         x0 = gp.getStandardCoordinate(0)
#         x1 = gp.getStandardCoordinate(1)
#         # print('function value',f(x0,x1), type(f(x0,x1)))
#         # print('dict value',samples[(x0,x1)], type(samples[(x0,x1)]))
#         alpha[i] = samples[(x0,x1)]
#         if gp.isLeaf():
#             was_leaf[i] = True
#             is_leaf.append(True)
#         else:
#             is_leaf.append(False)     
#     is_leaf_s.append(is_leaf)
#     # break
#     pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)
#     gridGen.refine(pysgpp.SurplusRefinementFunctor(alpha, 1))
#     print("refinement step {}, new grid size: {}".format(refnum+1, gridStorage.getSize()))    
#     alpha.resizeZero(gridStorage.getSize())
    



##!!!!!!!!!!!EVALUATING AN INTEGRAL USING ANALYTICALLY CALCULATED COEFFICIENTS THAT ASSUME THE HEIRACIAL BASIS FUNCTION IS AN ACCURATE INTERPOLATOR!!!!!!!!!!!

# import pysgpp
# from pysgpp import Grid, DataVector, DataMatrix

# # Define the dimensionality of the problem
# dim = 2

# # Create a piecewise linear grid
# grid = Grid.createLinearGrid(dim)
# grid_gen = grid.getGenerator()
# level = 3
# grid_gen.regular(level)

# # Get the grid points
# grid_storage = grid.getStorage()
# num_points = grid_storage.getSize()

# # Create a DataVector to store the function values at the grid points
# alpha = DataVector(num_points)

# # Define a function to integrate
# def f(x):
#     return x[0] * x[1]

# # Evaluate the function at the grid points
# for i in range(num_points):
#     gp = grid_storage.getPoint(i)
#     coords = [gp.getStandardCoordinate(d) for d in range(dim)]
#     alpha[i] = f(coords)

# # Create an operation to compute the quadrature weights
# op_quad = pysgpp.createOperationQuadrature(grid)

# # Compute the integral (quadrature)
# integral = op_quad.doQuadrature(alpha)

# # Print the result
# print("Integral:", integral)

# # To get the quadrature weights for each point, you can use the following:
# weights = DataVector(num_points)
# op_quad.getQuadratureWeights(weights)

# # Print the quadrature weights
# for i in range(num_points):
#     gp = grid_storage.getPoint(i)
#     coords = [gp.getStandardCoordinate(d) for d in range(dim)]
#     print(f"Point: {coords}, Weight: {weights[i]}")