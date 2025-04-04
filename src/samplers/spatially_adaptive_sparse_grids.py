import numpy as np
import warnings
import os
from common import S
import pysgpp
import samplers
from pysgpp import BoundingBox1D
from scipy.stats import sobol_indices, uniform, entropy
import matplotlib.pyplot as plt

class SpatiallyAdaptiveSparseGrids:
    def __init__(self, bounds, parameters, poly_basis_degree=3, initial_level=3, **kwargs):
        self.sampler_interface = S.ACTIVE
        self.bounds = bounds
        self.parameters = parameters
        if type(parameters[0]) == type([]):
            self.parameters = [tuple(pa) for pa in parameters]
        self.poly_basis_degree= poly_basis_degree
        self.initial_level = initial_level
        
        self.custom_limit = np.inf
        self.custom_limit_value=0 
        
        self.dim=len(parameters)
        
        # The test sampler is used to fill the space and run the surrogate model so that we can calcualte the integral and sobel indicies to check for convergence in UQ situations.
        check_sampler_kwargs = kwargs['test_sampler']
        self.check_sampler = getattr(samplers, check_sampler_kwargs['type'])(**check_sampler_kwargs)
        self.old_check_predictions = None
        self.num_active_cycles = 0
        self.all_cycle_info = []
        self.num_samples_by_cycle = []
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
        self.num_samples_by_cycle.append(self.gridStorage.getSize())
        return batch_samples
    
    def get_next_parameters(
        self,
        train: dict,
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
                        
            self.alpha[i] = train[box_point]
        
        
        pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(self.alpha)
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

            if box_point not in train.keys():
                param_dict = {}
                for j, param in enumerate(self.parameters):
                    param_dict[param]=box_point[j]
                batch_samples.append(param_dict)
        self.num_active_cycles += 1
        self.num_samples_by_cycle.append(len(train))
        return batch_samples
    
    def update_custom_limit_value(self): # Necessary
        NotImplemented
    
    def surrogate_predict(self, positions):
        positions = np.array(positions)
        positions_dm = pysgpp.DataMatrix(positions)
        opEval = pysgpp.createOperationMultipleEval(self.grid, positions_dm)
        results = pysgpp.DataVector(len(positions))
        opEval.eval(self.alpha, results)
        ans = np.array(results.array()) # sometimes this line breaks python. 
        # ans = np.array([results.get(i) for i in range(len(positions))])
        return ans
    
    def integral_approx(self, function_values):
        bounds_array = np.array(self.bounds).T
        space_volume = np.prod(bounds_array[1]-bounds_array[0])
        integral_estimate = np.mean(function_values) * space_volume
        return integral_estimate
    
    def expectation_2sigma_approx(self, function_values):
        expectation_estimate = np.mean(function_values)
        double_sigma_estimate = 2*np.sqrt(np.mean(function_values**2) - np.mean(function_values)**2)
        return expectation_estimate, double_sigma_estimate
    
    def sobel_indicies_approx(self):
        func = lambda positions: self.surrogate_predict(positions.T)
        sobol = sobol_indices(func=func,  n=2**np.log2(self.check_sampler.num_samples), dists=self.dists)
        return sobol.first_order, sobol.total_order
    
    def relative_entropy(samples_p, samples_q, num_bins):
        """
        Calculate the relative entropy (Kullback-Leibler divergence) between two distributions.

        Parameters:
        samples_p (list or np.array): Samples from the first distribution.
        samples_q (list or np.array): Samples from the second distribution.

        Returns:
        float: The relative entropy between the two distributions.
        """
        if type(samples_p) == None or type(samples_q) == None:
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
        headder = "approx_expectation, approx_2sigma, approx_sobol_1st_order, approx_sobol_total_order, approx_entropy_diff, num_samples"
        cycle_info_path = os.path.join(cycle_dir, 'sampler_cycle_info')
        if not os.path.exists(cycle_info_path):
            with open(cycle_info_path, 'w') as file:
                file.write(headder)
        predictions = self.surrogate_predict(self.check_sampler.samples)
        expectation, double_sigma = self.expectation_2sigma_approx(predictions)
        sobol_first_order, sobol_total_order = self.sobel_indicies_approx(predictions)
        entropy_diff = self.relative_entropy(self.old_check_predictions, predictions)
        self.old_check_predictions=predictions
        cycle_info = f"{expectation},{double_sigma},{sobol_first_order},{sobol_total_order},{entropy_diff},{self.num_samples_by_cycle[-1]}"
        with open(cycle_info_path, 'a') as file:
            file.write(cycle_info)
        self.all_cycle_info.append(cycle_info)
    
    def write_summary(self, base_dir):
        with open(os.path.join(base_dir, 'all_sampler_cycle_info'), 'w') as file:
            for cycle_info in self.all_cycle_info:
                file.write(cycle_info)
                
        predictions = self.surrogate_predict(self.check_sampler.samples)
        fig = plt.figure()
        bars = plt.hist(predictions, bins=200, density=True)
        heights, widths = bars[0], bars[1]
        plt.vlines([np.percentile(predictions,2.5),np.mean(predictions), np.percentile(predictions,97.5)],0, max(heights),color='red', label=r'mean and 95% confidance interval')
        plt.xlabel('function output')
        plt.ylabel('probability density')
        plt.title(f'Output Distribution, {self.check_sampler.num_samples} samples')
        fig.savefig(os.path.join(base_dir, 'output_distribution'), dpi=400)
        
        approx_expectation, approx_2sigma, approx_sobol_1st_order, approx_sobol_total_order, approx_entropy_diff = [],[],[],[],[]
        for cycle_info in self.all_cycle_info:
            approx_expectation_, approx_2sigma_, approx_sobol_1st_order_, approx_sobol_total_order_, approx_entropy_diff_ = cycle_info.split(',')
            approx_expectation.append(approx_expectation_), approx_2sigma.append(approx_2sigma_), approx_sobol_1st_order.append(approx_sobol_1st_order_), approx_sobol_total_order.append(approx_sobol_total_order_), approx_entropy_diff.append(approx_entropy_diff_)
        fig = plt.figure()
        plt.plot(self.num_samples_by_cycle, approx_expectation)
        plt.fill_between(self.num_samples_by_cycle, np.array(approx_expectation)-np.array(approx_2sigma),np.array(approx_expectation)+np.array(approx_2sigma), color='grey', label='2sigma')
        plt.ylabel('approx_expectation')
        plt.xlabel('Number of samples in cycle')
        plt.legend()
        fig.savefig(os.path.join(base_dir,'approx_expectation'), dpi=300)
        cycle_attributes = [approx_sobol_1st_order, approx_sobol_total_order, approx_entropy_diff]
        for att in cycle_attributes:
            fig = plt.figure()
            plt.ylabel(att.__qualname__)
            plt.xlabel('Number of samples in cycle')
            fig.savefig(os.path.join(base_dir, att.__qualname__),dpi=300)
        
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
    
