import numpy as np
import warnings
import os
import pysgpp
from scipy.stats import sobol_indices, uniform, entropy
from scipy.stats import norm, truncnorm
import pandas as pd
import shutil
import pickle
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
from time import sleep
from copy import deepcopy

import multiprocessing
multiprocessing.set_start_method("spawn")
import traceback
import functools

import multiprocessing
import traceback

from joblib import Parallel, delayed, cpu_count

# from enchanted_surrogates.runners.gene_single_monitor import calculate_decay_coefficient

# plugins = load_plugins()
# gene_monitor = plugins['gene_single_monitor']
# calculate_decay_coefficient = gene_monitor.calculate_decay_coefficient

#pysgpp common errors
# using self.alpha.array() in the same line of code causes an error
# np.array(self.alpha) causes an error.
# tring to fill an pysgpp vector outside of its size causes error
# trying to put a double inside a vector causes an error, should be float

class SgppSampler(Sampler):
    def __init__(self, bounds, parameters, test_dir=None, do_brute_force_sobol_indicies = False, brute_force_sobol_indicies_num_samples=1e6, **kwargs):
        self.base_run_dir = kwargs.get('base_run_dir')
        self.budget = kwargs.get('budget', 100)
        self.do_write_batch_info = kwargs.get('do_write_batch_info', True)
        self.write_batch_info_every_x_samples = kwargs.get('write_batch_info_every_x_samples', 1)
        self.do_quad_sobol = kwargs.get('do_quad_sobol', False)
        self.gaussian_input_uncertanties = kwargs.get('gaussian_input_uncertanties', False)
        self.test_dir = test_dir
        self.bounds = np.array(bounds)
        self.parameters = parameters
        self.num_samples_at_last_write = 0
        
        self.max_boundary_tree_points = kwargs.get('max_boundary_tree_points')
        self.boundary_threshold_quantile = kwargs.get('boundary_threshold_quantile')
        self.do_boundary_tree = kwargs.get('do_boundary_tree')
        self.do_add_initial_boundary = kwargs.get('do_add_initial_boundary', False)
        self.exploit = kwargs.get('exploit', 0.5)
        if type(parameters[0]) == type([]):
            print('CONVERTING LIST TO TUPLE')
            self.parameters = [tuple(pa) for pa in parameters]
            print(self.parameters, type(self.parameters[0]))
        self.do_brute_force_sobol_indicies = do_brute_force_sobol_indicies
        self.point_not_in_train_count = 0
        if self.do_brute_force_sobol_indicies:
            self.brute_force_sobol_indicies_num_samples = float(brute_force_sobol_indicies_num_samples)
                    
        self.custom_limit = np.inf
        self.custom_limit_value = 0
        self.dim=len(parameters)
        self.n_jobs = kwargs.get('n_jobs', 0)
        self.alpha = None
        
        adaptive_strategy = kwargs.get('adaptive_strategy',None)
        self.guide_dataset_size = 0
        # needed for active sampling but useful to opt out when using the samplers model for a predictor in post processing
        self.do_surplus_based=True
        self.adaptive_strategy = adaptive_strategy
        if adaptive_strategy:
            basis_type = adaptive_strategy['basis'].pop('type')
            self.grid = getattr(pysgpp.Grid, basis_type)(self.dim, **adaptive_strategy['basis'])
            self.gridStorage = self.grid.getStorage()
            self.gridGen = self.grid.getGenerator()
            self.initial_level = adaptive_strategy.get('initial_level', 2)
            self.gridGen.regular(self.initial_level)

            self.point_refinements_per_batch = adaptive_strategy.get('point_refinements_per_batch', 2)
            self.refinement_type = adaptive_strategy.get('refinement_type', 'surplus')
            if self.refinement_type == 'static_grid':
                self.level = self.initial_level
                self.final_level = self.adaptive_strategy.kwargs.get('final_level')
            
            self.guide_dataset_path = None
            self.guide_dataset = None
            if self.refinement_type == 'anova_spatially_dimensionally':
                self.guide_dataset_path = adaptive_strategy.get('guide_dataset_path', None)
                self.guide_sampler = import_sampler(adaptive_strategy['guide_sampler_config']['type'],adaptive_strategy['guide_sampler_config'])
            self.infer_bounds = adaptive_strategy.get('infer_bounds', False)
            self.infer_parents = adaptive_strategy.get('infer_parents', False)
            self.mean_recent_surplus_threshold = adaptive_strategy.get('mean_recent_surplus_threshold')
        
        self.mean_recent_surplus = np.inf
         
        # The test sampler is used to fill the space and run the surrogate model so that we can calcualte the integral and sobel indicies to brute_check for convergence in UQ situations.
        brute_check_sampler_config = kwargs.get('brute_check_sampler')
        if brute_check_sampler_config:
            self.do_brute_check=True
            brute_check_sampler_config['parameters'] = self.parameters
            brute_check_sampler_config['bounds'] = self.bounds
            self.brute_check_sampler = import_sampler(brute_check_sampler_config['type'], brute_check_sampler_config)
            self.old_brute_check_predictions = None
        else:
            self.do_brute_check=False
        self.batch_number = 0
        self.num_samples_by_batch = [0] 
        
        self.train = {}
        self.code_acquisition = {}
        self.run_dirs = {}
        # self.S.Act = {}
        self.bounds_points = {}
        self.parent_points = {}
        self.virtual_boundary_points = {}
        self.anchor_boundary_points = {}
        
        self.grid_increase = None
        self.custom_submitted = 0
    
    def point_transform_box2unit(self, box_point):
        # min max normalisation
        unit_point = tuple()
        for i, bco in enumerate(box_point):
            min_ = self.bounds[i][0]
            max_ = self.bounds[i][1]
            assert max_ > min_
            uco = (bco - min_) / (max_ - min_)
            unit_point = unit_point + (uco,)
        return unit_point

    def point_transform_unit2box(self, unit_point):
        # min max normalisation
        box_point = tuple()
        for i, uco in enumerate(unit_point):
            min_ = self.bounds[i][0]
            max_ = self.bounds[i][1]
            assert max_ > min_
            bco = uco*(max_ - min_) + min_
            box_point = box_point + (bco,)
        return box_point
    
    def points_transform_unit2box(self, unit_points):
        unit_points = np.array(unit_points)
        return unit_points * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    
    def points_transform_box2unit(self, box_points):
        return (box_points - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def get_initial_samples(self):
        # returns the initial set of parameters for initial trianing of surrogate model
        # self.grid = pysgpp.Grid.createPolyBoundaryGrid(self.dim, self.poly_basis_degree)
        print('INITIAL LEVEL:',self.initial_level)
        # self.gridGen.regular(self.initial_level)
        print('INITIAL GRID SIZE:', self.gridStorage.getSize())
        # array containing labels in order they come from;- for i in range(gridStorage.getSize()): gp = gridStorage.getPoint(i)
        self.alpha = pysgpp.DataVector(self.gridStorage.getSize())
        # self.Act_aquisition = pysgpp.DataVector(self.gridStorage.getSize())
        
        
        batch_samples = []
        boundary_counter = 0
        print('initial samples',self.gridStorage.getSize())
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = tuple()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
                
            if 1.0 in unit_point or 0.0 in unit_point:
                pass
                boundary_counter += 1
                
            box_point = self.point_transform_unit2box(unit_point)
        
            param_dict = {}
            for j, param in enumerate(self.parameters):
                param_dict[param]=box_point[j]
        
            batch_samples.append(param_dict)
        
        print('boundary counter',boundary_counter)
        self.grid_increase = self.gridStorage.getSize()
        self.custom_submitted += len(batch_samples)
        self.batch_number += 1
        return batch_samples
    
    def safe_run(self, function, *args, **kwargs):
        """
        Safely executes a function in a separate process to isolate potential crashes.

        This method wraps the given `function` to run in a subprocess with automatic exception handling.
        It ensures that any result or error is communicated back via a multiprocessing queue.
        If the subprocess crashes (e.g., due to a segmentation fault), the main process remains unaffected.

        Args:
            function (callable): The function to execute safely. It does NOT need to handle queues or exceptions.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result returned by the function if successful, or None if the function crashes or fails.
        """
        
        # def wrapper(queue, *args, **kwargs):
        #     try:
        #         result = function(*args, **kwargs)
        #         queue.put(("success", result))
        #     except Exception:
        #         queue.put(("error", traceback.format_exc()))

        # queue = multiprocessing.Queue()
        # p = multiprocessing.Process(target=wrapper, args=(queue, *args), kwargs=kwargs)
        # p.start()
        # p.join()

        # if not queue.empty():
        #     status, payload = queue.get()
        #     if status == "success":
        #         return payload
        #     else:
        #         print("safe run of function failed with error:")
        #         print(payload)
        # else:
        #     print("Function crashed or was killed. Potentially because of a segmentation fault")
        # return None

        # I could not find a safe run that works well in a container in HPC without pickeling. I should probably look at full isolation with subprocesses, seperate scripts and saving variables to files.
        return function(*args, **kwargs)

    
    def get_guide_data(self):
        if self.guide_dataset_path:
            if self.base_run_dir:                    
                if not os.path.exists(os.path.join(self.base_run_dir, 'batch_00')):
                    os.mkdir(os.path.join(self.base_run_dir, 'batch_00'))                        
                shutil.copy(self.guide_dataset_path, os.path.join(self.base_run_dir, 'batch_00', 'enchanted_dataset.csv'))
                with open(os.path.join(self.base_run_dir, 'batch_00', 'readme.txt'), 'w') as readme:
                    readme.write(f"""
        This data set was copyed here from, {self.guide_dataset_path}.
        It is the initial dataset that was used to guide the anova spatially dimensioanlly adaptive sparse grid refinement:
        https://sgpp.sparsegrids.org/docs/example_predictiveANOVARefinement_py.html
        """)
            self.guide_dataset_df = pd.read_csv(self.guide_dataset_path, index_col=False)
            print('debug df', self.guide_dataset_df.columns, self.parameters)
            self.guide_dataset_df_inputs = self.guide_dataset_df[self.parameters]
            output_col = [col for col in self.guide_dataset_df.columns if 'output' in col]
            assert len(output_col) == 1
            self.guide_dataset_df_output = self.guide_dataset_df[output_col]
            self.guide_dataset_inputs_box_points = self.guide_dataset_df_inputs.to_numpy()
            self.guide_dataset_inputs_unit_points = np.array(self.points_transform_box2unit(self.guide_dataset_inputs_box_points))
            assert len(self.guide_dataset_inputs_unit_points) == len(self.guide_dataset_df)
            self.guide_dataset_size = len(self.guide_dataset_df)
            self.guide_dataset = pysgpp.DataMatrix(self.guide_dataset_inputs_unit_points.tolist())
        elif self.guide_sampler is not None:
            samples = self.guide_sampler.get_next_samples()
            self.custom_submitted += len(samples)
            self.batch_number += 1
            return samples
        
    
    def compute_basis_function_volumes(self):
        # These volumes only apply to the unit space
        basis = self.grid.getBasis()
        volumes = pysgpp.DataVector(self.gridStorage.getSize())
        volumes_ = []
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            volume = 1.0
            for d in range(self.gridStorage.getDimension()):
                level = gp.getLevel(d)
                index = gp.getIndex(d)
                volume *= basis.getIntegral(level, index)
            volumes[i] = volume
            volumes_.append(volume)
        return volumes, np.array(volumes_)
    
    def get_adapted_samples(self):
        
        previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
        new_data_df = pd.read_csv(os.path.join(previous_batch_dir, f'enchanted_dataset.csv'))
        output_col = [col for col in new_data_df.columns if 'output' in col]
        if len(output_col) > 1:
            raise RuntimeError('StaticSparseGrid SAMPLER REQUIRES EXACTLY ONE OUTPUT VARIABLE. THE single_code_run IN THE RUNNER SHOULD RETURN A DICTIONARY OF OUTPUTS WHERE ONLY ONE HAS output IN THE KEY, eg \{growthrate_output\:5\}. THIS IS THE ONE THAT WILL BE USED FOR ACTIVE LEARNING PUTPOSES')
        train_df = new_data_df[self.parameters + output_col]
        new_train = {
            tuple(row[col] for col in self.parameters): float(row[output_col].iloc[0])
            for _, row in train_df.iterrows()
        }

        acquisition_col = [col for col in new_data_df.columns if 'acquisition' in col]
        if len(acquisition_col) > 0:
            new_acquisition = {
                tuple(row[col] for col in self.parameters): float(row[acquisition_col].iloc[0])
                for _, row in new_data_df.iterrows()
            }
            self.code_acquisition.update(new_acquisition)

        self.train.update(new_train)
        
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = ()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
            box_point = self.point_transform_unit2box(unit_point)

            self.alpha[i] = self.lookup_function(box_point, self.train)
            # if not self.approx_in(box_point, self.train): # and box_point not in self.bounds_points and box_point not in self.parent_points:
            #     if self.infer_bounds and not gp.isInnerPoint() or self.infer_parents and not gp.isLeaf():
            #         # If it is both then default to boundary point
            #         if self.infer_bounds and not gp.isInnerPoint():
            #             print('appending bounds points')
            #             self.bounds_points[box_point] = np.mean(list(self.train.values()))#self.surrogate_predict([box_point])[0]
            #         # TODO: infer parents WOn't work as surrogate predict has no alpha value, put alpha set up into surrogate predict
                    # elif self.infer_parents and not gp.isLeaf():
                    #     self.parent_points[box_point] = self.surrogate_predict([box_point])[0]
                    #     print('appending parent points')
            # if box_point not in self.decay.keys():
            #     self.decay[box_point] = calculate_decay_coefficient(self.run_dirs[box_point])
            # self.alpha[i] = self.parent_value_scaled(box_point)
            # self.decay_aquisition[i] = -(self.decay[box_point]**2)
            # if box_point in self.train.keys():
            #     self.alpha[i] = self.train[box_point]
            # elif box_point in self.bounds_points.keys():
            #     self.alpha[i] = self.bounds_points[box_point]
            # elif box_point in self.parent_points.keys():
            #     self.alpha[i] = self.parent_points[box_point]
                
        
        
        # This changes alpha from the training point values into the surpluses
        # pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(self.alpha)
        self.heirarchisation(self.grid, self.alpha)
        # Now the surrogate is trained we can write batch info
        self.num_samples = len(self.train)+self.guide_dataset_size
        if self.do_write_batch_info:
            if self.num_samples - self.num_samples_at_last_write >= self.write_batch_info_every_x_samples or self.batch_number in [0,1,2,3]:
                # Now the surrogate is trained we can write batch info
                self.write_batch_info(previous_batch_dir)
                self.num_samples_at_last_write = self.num_samples
        else:
            self.save_grid(previous_batch_dir)
                        
        grid_size_before_refinement = self.gridStorage.getSize()
        print('grid size before refinement:', grid_size_before_refinement)
        
        print(f'CURRENT REFINEMENT STRATEGY, DO-SURPLUS:{self.do_surplus_based}')
        self.refine()
        
        self.update_mean_recent_surplus()
        
        grid_size_after_refinement = self.gridStorage.getSize()
        print('grid size after refinement:', grid_size_after_refinement)
        self.grid_increase = grid_size_after_refinement - grid_size_before_refinement
        # print("refinement step {}, new grid size: {}".format(refnum+1, gridStorage.getSize()))  
        self.alpha.resizeZero(self.gridStorage.getSize())
        # self.decay_aquisition.resizeZero(self.gridStorage.getSize())
        
        batch_samples = []
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = ()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
            
            box_point = self.point_transform_unit2box(unit_point)

            if not self.approx_in(box_point, self.train): # and box_point not in self.bounds_points and box_point not in self.parent_points:
                print('NEW POINT',box_point, unit_point)
                print('IS BOUNDARY', not gp.isInnerPoint())
                print('is parent', not gp.isLeaf())
                
                if self.infer_bounds and self.is_boundary_point(unit_point):
                    NotImplemented # do nothing
                if self.infer_parents and not gp.isLeaf():
                    NotImplemented # do nothing
                else:
                    param_dict = {}
                    for j, param in enumerate(self.parameters):
                        param_dict[param]=box_point[j]                    
                    print('adding to batch samples')
                    batch_samples.append(param_dict)
        
        self.batch_number += 1
        print('len batch_samples',len(batch_samples))        
        self.custom_submitted += len(batch_samples)
        
        if self.custom_submitted > self.budget:
            # self.budget = 0 # this will stop the executor from asking for more samples because base_sampler.has_budget() will return false 
            # using return None to stop sampling as I need sampler to request samples for one batch more than the config allows so that I can write the batch info on the last batch
            # self.merge_batch_info()
            return None # this needs to be allowed by the executor
        
        if self.refinement_type=='static_grid':
            if self.level > self.final_level:
                self.merge_batch_info()
                return None
        return batch_samples

    def _heirarchisation(self, grid, alpha):
        pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)
    
    def heirarchisation(self, *args, **kwargs):
        return self.safe_run(self._heirarchisation, *args, **kwargs)
    
    def get_next_samples(
        self,
        *args,
        **kwargs) -> list[dict[str, float]]:
        # returns a list of parameters that are the next batch to be labeled, based on the training samples, models used and selection criteria.
        """
        args
        train, dict: A dictionary where the keys are a tuple of inputs (x0,x1,x2) (*in the order of the origional parameters) and the value is a label
        """
        #        
        if not self.base_run_dir:
            raise RuntimeError('base_run_dir IS NOT SET IN SAMPLER. THIS IS PASSED IN THE CONFIG FOR THE EXECUTOR. THE EXECUTOR MUST THEN PASS IT TO THE SAMPLER SO IT CAN GRAB DATA FOR TRAINING. ENSURE THE EXECUTOR HAS THIS LINE IN start_runs --> self.sampler.base_run_dir = self.base_run_dir ')
        
        if self.refinement_type == 'anova_spatially_dimensionally':
            if self.batch_number == 0:
                print('debug anova_spatiall_dimensionally, batch 0')
                guide_samples = self.get_guide_data()
                if guide_samples:
                    next_samples = guide_samples
                else:
                    next_samples = self.get_initial_samples()
            elif self.batch_number == 1 and not self.guide_dataset:
                print('debug anova_spatiall_dimensionally, batch 1')
                self.guide_dataset_path = os.path.join(self.base_run_dir, 'batch_0','enchanted_dataset.csv')
                self.guide_dataset_df = pd.read_csv(self.guide_dataset_path, index_col=False)
                print('debug df', self.guide_dataset_df.columns, self.parameters)
                self.guide_dataset_df_inputs = self.guide_dataset_df[self.parameters]
                output_col = [col for col in self.guide_dataset_df.columns if 'output' in col]
                assert len(output_col) == 1
                self.guide_dataset_df_output = self.guide_dataset_df[output_col]
                self.guide_dataset_inputs_box_points = self.guide_dataset_df_inputs.to_numpy()
                self.guide_dataset_inputs_unit_points = np.array(self.points_transform_box2unit(self.guide_dataset_inputs_box_points))
                assert len(self.guide_dataset_inputs_unit_points) == len(self.guide_dataset_df)
                self.guide_dataset = pysgpp.DataMatrix(self.guide_dataset_inputs_unit_points.tolist())
                next_samples = self.get_initial_samples()
            else:
                next_samples = self.get_adapted_samples()
        elif self.batch_number == 0:
            next_samples = self.get_initial_samples()
        else:
            next_samples = self.get_adapted_samples()
        self.num_samples_by_batch.append(len(self.train)+self.guide_dataset_size)
        return next_samples
    
    def update_mean_recent_surplus(self):
        if self.grid_increase != None:
            self.mean_recent_surplus = np.mean( np.abs(np.array(self.alpha.array())[-self.grid_increase : ]) )
        else:
            self.mean_recent_surplus = np.mean( np.abs(np.array(self.alpha.array())) )
    
    def variance_refinement(self):
        alpha = pysgpp.DataVector(self.gridStorage.getSize())
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = ()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
            
            box_point = self.point_transform_unit2box(unit_point)
            alpha[i] = float(self.lookup_function(box_point)**2)
        # pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(self.alpha)
        self.heirarchisation(self.grid, self.alpha)
        self.gridGen.refine(pysgpp.SurplusRefinementFunctor(alpha, self.point_refinements_per_batch))
    
    def approx_in(self, query_key, dictionary, tol=1e-9):
        return any(all(abs(a - b) < tol for a, b in zip(query_key, key)) for key in dictionary)
    
    def refine(self):
        if self.refinement_type == 'surplus':
            self.gridGen.refine(pysgpp.SurplusRefinementFunctor(self.alpha, self.point_refinements_per_batch))
        elif self.refinement_type == 'volume':
            volumes, _ = self.compute_basis_function_volumes()
            self.gridGen.refine(pysgpp.SurplusRefinementFunctor(volumes, self.point_refinements_per_batch))
        elif self.refinement_type == 'surplus_volume':
            # this does surplus times volume and takes the max
            self.gridGen.refine(pysgpp.SurplusVolumeRefinementFunctor(self.alpha, self.point_refinements_per_batch))
        elif self.refinement_type == 'variance':
            self.variance_refinement()        
        elif self.refinement_type == 'expectation':
            # only applies to uniform probability distirbutions of inputs:
            # DOI: 10.1007/978-3-319-06898-5_7
            # This refinement criterion includes for each collocation node the volume of the basis function’s support and describes therefore a local estimate for the quadrature error.
            
            alpha = pysgpp.DataVector(self.gridStorage.getSize())
            for i in range(self.gridStorage.getSize()):
                gp = self.gridStorage.getPoint(i)
                levels = []
                for dimension in range(self.dim):
                    dimension = dimension + 1
                    levels.append(gp.getLevel(dimension))
                lev = np.max(levels)
                alpha[i] = float(np.abs(self.alpha[i])*2.0**(-lev))
            # pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(self.alpha)
            self.heirarchisation(self.grid, self.alpha)
            self.gridGen.refine(pysgpp.SurplusRefinementFunctor(alpha, self.point_refinements_per_batch))
        elif self.refinement_type == 'variance_then_surplus':
            print('mean_recent_surplus', self.mean_recent_surplus, 'threshold', self.mean_recent_surplus_threshold)
            
            if self.mean_recent_surplus < self.mean_recent_surplus_threshold or self.do_surplus_based:
                self.do_surplus_based = True
                print('doing surplus refinement')
                self.gridGen.refine(pysgpp.SurplusRefinementFunctor(self.alpha, self.point_refinements_per_batch))
            else:
                print('doing variance refinement')
                self.do_surplus_based = False
                self.variance_refinement()
        elif self.refinement_type == 'code_acquisition':
            volumes, volumes_ = self.compute_basis_function_volumes()
            # Min-Max Normalization
            volumes_norm = (volumes_ - volumes_.min()) / (volumes_.max() - volumes_.min())
            code_acquisition = np.array(list(self.code_acquisition.values()))
            code_acquisition_norm = {key: (ca - np.nanmin(code_acquisition)) / (np.nanmax(code_acquisition) - np.nanmin(code_acquisition)) for key, ca in self.code_acquisition.items()}
            
            code_aquisition_volume = pysgpp.DataVector(self.gridStorage.getSize())
            for i in range(self.gridStorage.getSize()):
                gp = self.gridStorage.getPoint(i)
                unit_point = ()
                for j in range(self.dim):
                    unit_point = unit_point + (gp.getStandardCoordinate(j),)
                
                box_point = self.point_transform_unit2box(unit_point)
                if np.isnan(code_acquisition_norm[box_point]):
                    code_aquisition_volume[i] = 0.0
                else:
                    code_aquisition_volume[i] = float(self.exploit*code_acquisition_norm[box_point] + (1-self.exploit)* volumes_norm )
                    # decay_aquisition_.append(-(self.decay[box_point]**2))
            
            self.gridGen.refine(pysgpp.SurplusRefinementFunctor(code_aquisition_volume, self.point_refinements_per_batch))
        elif self.refinement_type == 'anova_spatially_dimensionally':
            # num_initial_datapoints = len(self.guide_dataset_df)
            # errorVector = DataVector(num_initial_datapoints)
            pred = np.array(self.surrogate_predict(self.guide_dataset_inputs_box_points)).flatten()
            assert len(pred) == len(self.guide_dataset_df)
            true = self.guide_dataset_df_output.to_numpy().flatten()
            assert len(pred) == len(self.guide_dataset_df)
            error = np.array((pred - true)**2).flatten().tolist()
            errorVector = pysgpp.DataVector(error)            
            #refinement  stuff
            refinement = pysgpp.ANOVAHashRefinement()
            decorator = pysgpp.PredictiveRefinement(refinement)
            # refine a single grid point each time
            # print("Error over all = %s" % errorVector.sum())
            indicator = pysgpp.PredictiveRefinementIndicator(self.grid, self.guide_dataset, errorVector, self.point_refinements_per_batch)
            decorator.free_refine(self.gridStorage, indicator)
            # print("Refinement step %d, new grid size: %d" % (refnum+1, HashGridStorage.getSize()))
        elif self.refinement_type == "static_grid":
            self.level += 1
            basis_type = self.adaptive_strategy['basis'].pop('type')
            self.grid = getattr(pysgpp.Grid, basis_type)(self.dim, **self.adaptive_strategy['basis'])
            self.gridStorage = self.grid.getStorage()
            self.gridGen = self.grid.getGenerator()
            self.gridGen.regular(self.level)     
    
    def parent_value_scaled(self, box_point):
        # if self.gaussian_input_uncertanties:
        #     unit_point = self.point_transform_box2unit(box_point)
        #     return self.unit_truncnorm_pdf(unit_point) * self.lookup_function(box_point)
        # else:
        return self.lookup_function(box_point)
        
    
    def update_custom_limit_value(self): # Necessary
        NotImplemented
    
    def is_boundary_point(self, unit_point):
        if 0 in unit_point or 1 in unit_point:
            return True
        else:
            return False
    
    # currently only works with point in the unit cube.     
    def surrogate_predict(self, points, n_jobs=0, space='box_space'):
        if space == 'box_space':
            unit_points = self.points_transform_box2unit(points)
        else:
            unit_points = points

        self.gridStorage = self.grid.getStorage()
        alpha = pysgpp.DataVector(self.gridStorage.getSize())
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = ()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
            box_point = self.point_transform_unit2box(unit_point)
            alpha[i] = float(self.lookup_function(box_point))
        
        # pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(alpha)
        self.heirarchisation(self.grid, self.alpha)
        
        def batch_predict(unit_points):
            import pysgpp as sg
            unit_points = np.array(unit_points).tolist()
            unit_points_dm = sg.DataMatrix(unit_points)
            opEval = sg.createOperationMultipleEval(self.grid, unit_points_dm)
            results = sg.DataVector(len(unit_points))
            opEval.eval(alpha, results)
            ans = np.array(results.array()) # sometimes this line breaks python. 
            # ans = np.array([results.get(i) for i in range(len(positions))])
            return ans
        if n_jobs==0:
            return batch_predict(unit_points)
        else:
            if n_jobs == -1:
                n_jobs = cpu_count()
            split_box_points = np.array_split(unit_points, n_jobs)
            ans_list = Parallel(n_jobs=n_jobs)(delayed(batch_predict)(unit_points) for unit_points in split_box_points)
            return np.concatenate(ans_list, axis=0)
    
    def _quadrature_integral(self):
        op_quad = pysgpp.createOperationQuadrature(self.grid)
        print('debug performing quadrature integral')
        unit_integral = op_quad.doQuadrature(self.alpha)
        return unit_integral
    
    def quadrature_integral(self, *args, **kwargs):
        return self.safe_run(self._quadrature_integral, *args, **kwargs)
    
    # def combi_grid_analysis(self):
    #     THE ISSUE WITH THIS METHOD IS THAT IT REQUIRED MORE GROUND TRUTH FUNCTION EVALUATIONS THAN THE PARENT SPARSE GRID.
    #     "This converts the heirarchial grid function into a combi-grid global function that can be used for expectation, var and sobol indicies calculation"
    #     treeStorage_all = pysgpp.convertHierarchicalSparseGridToCombigrid(self.grid.getStorage(),
    #                                                            pysgpp.GridConversionTypes_ALLSUBSPACES)
    #     surrogate_predict_unit = lambda points: self.surrogate_predict(points, n_jobs=0, space='unit_space')
    #     func = pysgpp.MultiFunc(surrogate_predict_unit)    
    #     opt_all = pysgpp.CombigridMultiOperation.createExpUniformLinearInterpolation(self.dim, func)
    #     opt_all.getLevelManager().addLevelsFromStructure(treeStorage_all)
        
    #     # create polynomial basis
    #     config = pysgpp.OrthogonalPolynomialBasis1DConfiguration()
    #     config.polyParameters.type_ = pysgpp.OrthogonalPolynomialBasisType_LEGENDRE
    #     basisFunction = pysgpp.OrthogonalPolynomialBasis1D(config)
        
    #     # create polynomial chaos surrogate from sparse grid
    #     surrogateConfig = pysgpp.CombigridSurrogateModelConfiguration()
    #     surrogateConfig.type = pysgpp.CombigridSurrogateModelsType_POLYNOMIAL_CHAOS_EXPANSION
    #     surrogateConfig.loadFromCombigridOperation(opt_all)
    #     surrogateConfig.basisFunction = basisFunction
    #     pce = pysgpp.createCombigridSurrogateModel(surrogateConfig)
    #     # compute sobol indices
    #     sobol_indices = pysgpp.DataVector(1)
    #     total_indices = pysgpp.DataVector(1)
    #     pce.getComponentSobolIndices(sobol_indices)
    #     pce.getTotalSobolIndices(total_indices)
    #     # print results
    #     print('COMBIGRID ANALYSIS')
    #     print("Mean: {} Variance: {}".format(pce.mean(), pce.variance()))
    #     print("Sobol indices {}".format(sobol_indices.toString()))
    #     print("Total Sobol indices {}".format(total_indices.toString()))
    #     print("Sum {}\n".format(sobol_indices.sum()))
        
    #     out_dict = {'mean':pce.mean, 'std': np.sqrt(pce.variance), 'sobolF':sobol_indices, 'sobolT':total_indices}
        
    #     return out_dict
    
    def _quadrature_function_integral(self, function, acted_on='box_point'):
        # this would be useful for calculating expectation values, variances and sobol indicies, even with non uniform probability distirbutions
        # The function can be a data look up table that is multiplied by a prob density for expectation or squared for variance etc
        # self.gridStorage = self.grid.getStorage()
        self.gridStorage = self.grid.getStorage()
        alpha = pysgpp.DataVector(self.gridStorage.getSize())
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = ()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
                box_point = self.point_transform_unit2box(unit_point)
                try:
                    val = float(function(unit_point if acted_on == 'unit_point' else box_point))
                    if not np.isfinite(val):
                        warnings.warn(f'quadrature function integral got a non finite value from the function: {val}')
                        val = 0.0
                except Exception as e:
                    warnings.warn(f'quadrature function integral got an error when evaluating function value from the function: {e}')
                    val = 0.0
                alpha[i] = val
        # compute surpluses
        # pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(alpha)
        self.heirarchisation(self.grid, alpha)
        print('debug performing quadrature in quadrature function integral')
        op_quad = pysgpp.createOperationQuadrature(self.grid)
        unit_integral = op_quad.doQuadrature(alpha)
        return unit_integral
    
    def quadrature_function_integral(self, *args, **kwargs):
        return self.safe_run(self._quadrature_function_integral, *args, **kwargs)
    
    def approx_lookup(self, query_key, dictionary, tol=1e-9, default=None):
        return float(next(
            (dictionary[key]
            for key in dictionary
            if all(abs(a - b) < tol for a, b in zip(query_key, key))),
            default
        ))
    
    def lookup_function(self, point, unit_or_box='box'):
        if unit_or_box == 'box':
            box_point = point
        elif unit_or_box == 'unit':
            box_point = self.point_transform_unit2box(point)
        else:
            box_point = point
        box_point = tuple(box_point)
        if self.approx_in(box_point, self.train):
            return self.approx_lookup(box_point, self.train)
        
        elif self.approx_in(box_point, self.bounds_points):
            return self.approx_lookup(box_point, self.bounds_points)
        
        elif self.approx_in(box_point, self.parent_points):
            return self.approx_lookup(box_point, self.parent_points)
        
        elif self.approx_in(box_point, self.virtual_boundary_points):
            return self.approx_lookup(box_point, self.virtual_boundary_points)
        
        elif self.approx_in(box_point, self.anchor_boundary_points):
            return self.approx_lookup(box_point, self.anchor_boundary_points)
        else: 
            self.point_not_in_train_count += 1
            # get closeest point,
            all_points = list(self.train.keys()) + list(self.parent_points.keys()) + list(self.bounds_points.keys())
            # all_points_dict = self.train | self.parent_points | self.bounds_points
            closest_point = all_points[np.argmin(np.sum(np.abs(np.array(all_points) - box_point)**2))]
            messages=[f'this point {box_point} was not in train, bounds_points or parent_points',
                      f'len train: {len(self.train)}',
                      f'grid size: {self.gridStorage.getSize()}',
                      f'closeest point: {closest_point}',
                      f'number of times this issue has occured: {self.point_not_in_train_count}', 
                      'returning simulation value for closest point']
                    #   f'train:{self.train}']
            message = '\n'.join(messages)
            # warnings.warning(message)
            # return self.lookup_function(closest_point)
            
            raise KeyError(message)
            
    
    def unit_truncnorm_pdf(self, unit_point):
        sigma_multiplyer = 3
        trunc_gaussian = truncnorm(a=-sigma_multiplyer, b=sigma_multiplyer, loc=0.5, scale=0.5/sigma_multiplyer)
        return np.prod(trunc_gaussian.pdf(unit_point))
    
    def quadrature_expectation(self):
        # assumes uniform input distributions
        # if self.gaussian_input_uncertanties:
        #     # Pfx = lambda unit_point: self.unit_truncnorm_pdf(unit_point)*self.lookup_function(unit_point, unit_or_box='unit')
        #     return self.quadrature_function_integral(function=self.gaussian_Pfx, acted_on='unit_point')        
        return self.quadrature_integral()
    
    def gaussian_Pfx(self, unit_point):
        Px = self.unit_truncnorm_pdf(unit_point)
        fx = self.lookup_function(unit_point, unit_or_box='unit')
        return Px*fx
    
    def _quadrature_variance(self):
        # assumes uniform probability distirbutions for inputs
        # VAR(f(x)) = EXP[f(x)**2] - EXP[f(x)]**2
        # VAR(f(x)) = EXP[({f(x)-EXP[f(x)]}**2]
        # if self.gaussian_input_uncertanties:
        #     fxsquared = lambda unit_point: self.unit_truncnorm_pdf(unit_point)*self.lookup_function(unit_point, unit_or_box='unit')**2
        EXPfx = self.quadrature_expectation()
        fx_neg_mean_squared = lambda unit_point: (self.lookup_function(unit_point, unit_or_box='unit') - EXPfx)**2
        VARfx = self.quadrature_function_integral(fx_neg_mean_squared, acted_on='unit_point')

        # fxsquared = lambda unit_point: self.lookup_function(unit_point, unit_or_box='unit')**2
        # print('after fx sq')
        # EXPfxsquared = self.quadrature_function_integral(fxsquared, acted_on='unit_point')
        # print('after exp fx sq')
        # EXPfx = self.quadrature_expectation()
        # print('after exp fx')
        # VARfx = EXPfxsquared - EXPfx**2
        return VARfx
    
    def quadrature_variance(self, *args, **kwargs):
        return self.safe_run(self._quadrature_variance, *args, **kwargs)
    
    def integral_approx(self, function_values):
        bounds_array = np.array(self.bounds).T
        space_volume = np.prod(bounds_array[1]-bounds_array[0])
        integral_estimate = np.mean(function_values) * space_volume
        return integral_estimate
    
    # Assumes the inputs used to get the function values were sampled from their uncertainty distributions.
    # def expectation_approx(self, function_values, num_bins=200):
    #     # assumes uniform input distributions
    #     heights, edges = np.histogram(function_values, bins=num_bins)
    #     weights = heights/sum(heights)
    #     centers = (edges[:-1] + edges[1:]) / 2
    #     expectation = np.average(centers,weights=weights)
    #     return expectation
    
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
        
    # def double_sigma_approx(self, function_values):
    #     double_sigma_estimate = 2*(np.sqrt(self.expectation_approx(function_values**2) - self.expectation_approx(function_values)**2))
    #     return double_sigma_estimate
    
    def sobel_indicies_approx(self):
        self.dists = []
        for b in self.bounds:
            assert b[1] > b[0]
            self.dists.append(uniform(loc=b[0], scale=b[1]-b[0]))
        
        func = lambda positions: self.surrogate_predict(positions.T, n_jobs=self.n_jobs)
        sobol = sobol_indices(func=func,  n=2**int(np.log2(self.brute_force_sobol_indicies_num_samples)), dists=self.dists)
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
    
        
        # ##save new grid with boundary tree values        
        # print('pickeling boundary tree points')        
        # with open(os.path.join(batch_dir,'virtual_boundary_points.pkl'), 'wb') as file:
        #     pickle.dump(self.virtual_boundary_points, file)

        # print('pickeling anchor boundary points')        
        # with open(os.path.join(batch_dir,'anchor_boundary_points.pkl'), 'wb') as file:
        #     print('debug len anchor points 10', len(self.anchor_boundary_points))
        #     pickle.dump(self.anchor_boundary_points, file)
        
        # print('SAVING GRID')
        # with open(os.path.join(batch_dir,'pysgpp_grid_virtual_boundary.txt'), 'w') as file:
        #     file.write(self.grid.serialize())
        # print('SAVING SURPLUSES')
        # self.alpha.toFile(os.path.join(batch_dir, 'surpluses_virtual_boundary.mat'))
        
        # del grid
        # del gridStorage
        # del gridGen
        # del alpha        

    def copy_grid(self):
        new_grid = self.grid.createGridOfEquivalentType(len(self.parameters))
        new_grid_storage = new_grid.getStorage()
        # Transfer points from old grid to new grid
        # new_alpha = pysgpp.DataVector(self.gridStorage.getSize())
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            new_grid_storage.insert(pysgpp.HashGridPoint(gp)) #Hash maybe not needed
            # unit_point = ()
            # for j in range(self.dim):
            #     unit_point = unit_point + (gp.getStandardCoordinate(j),)
            # box_point = self.point_transform_unit2box(unit_point) 
            # new_alpha[i] = self.train[box_point]
        # pysgpp.createOperationHierarchisation(new_grid).doHierarchisation(new_alpha)
        # sasg.grid = new_grid
        # sasg.alpha = new_alpha
        # print('sasg degree',sasg.grid.getDegree())
        return new_grid
    
    def write_cycle_info(self, *args, **kwargs):
        return self.write_batch_info(*args, **kwargs)       
    
    def write_batch_info(self, batch_dir, name='', save_grid=True): #Necessary
        fname = name+'batch_info.csv'
        # print('debug do boundary tree:', self.do_boundary_tree)
        # if self.do_boundary_tree:
        #     self.add_boundary_tree(batch_dir)
        if self.do_write_batch_info:
            print(f'+++ \n Write ACTIVE batch: {self.batch_number}')
            print('+++ \n NUM INNER LEAF POINTS', len(self.train))            
            print('+++ \n NUM PARENT POINTS', len(self.parent_points))
            print('+++ \n NUM bounds POINTS', len(self.bounds_points))
            # df = pd.DataFrame({"num_samples":[self.num_samples_by_batch[-1]], "num_parents":[len(self.parent_points)], "num_bounds":[len(self.bounds_points)], "quad_expectation":[self.quadrature_expectation()],"quad_double_sigma":[2*np.sqrt(self.quadrature_variance())]})
            print('before quad exp')
            quad_exp = self.quadrature_expectation()
            print('before quad var')
            quad_std = np.sqrt(self.quadrature_variance())
            print('after quad var')
            df = pd.DataFrame({"num_samples":[len(self.train)+self.guide_dataset_size], "num_parents":[len(self.parent_points)], "num_bounds":[len(self.bounds_points)], "quad_mean":[quad_exp],"quad_std":[quad_std],"num_anchor_points":[len(self.anchor_boundary_points)]})
            if self.grid_increase != None:
                df['mean_recent_surplus'] = [np.mean( np.abs(np.array(self.alpha.array())[-self.grid_increase : ]) )]
            else:
                df['mean_recent_surplus'] = [np.mean( np.abs(np.array(self.alpha.array())) )]
            df['mean_surplus'] = [np.mean(np.abs(np.array(self.alpha.array())))]
            df['max_surplus'] = [np.max(np.abs(np.array(self.alpha.array())))]
            df['do_surplus_based'] = self.do_surplus_based

            data_dir = os.path.join(self.base_run_dir, 'enchanted_dataset.csv')
            data_df = pd.read_csv(data_dir)
            output_col = [col for col in data_df.columns if 'output' in col]
            if len(output_col)>1:
                warnings.warn(f'MORE THAN ONE OUTPUT COL {output_col}. Taking the first')
            output_col = output_col[0]
            weighted_mean, weighted_var, weights = self.kde_weighted_mean_var(data_df, value_col=output_col, parameters=self.parameters)
            df['weighted_mean'] = [weighted_mean]
            df['weighted_std'] = [np.sqrt(weighted_var)]
            
            sobol_indices, anchor_fraction = self.anchored_anova_firstorder_sobol(df_or_path=data_df, parameters=self.parameters, total_var=weighted_var, value_col=output_col, weight_col=None, tol=1e-6, min_slice_size=3)
            for param, si, af in zip(self.parameters,sobol_indices, anchor_fraction):
                df[f'{param}_anchorAnova_sobolF'] = [si]
                df[f'{param}_anchorAnova_anchorFrac'] = [af]

            if self.do_quad_sobol:
                quad_first_order_sobol = self.quadrature_first_order_sobol(num_points=20)
                for param, si in zip(self.parameters, quad_first_order_sobol):
                    df[f'{param}_quad_sobolF'] = [si]
            
            # if self.test_dir != None:
            #     x_test, y_test = self.get_test_set(self.test_dir)
            #     y_pred = self.surrogate_predict(x_test, n_jobs=self.n_jobs)
            #     residuals = y_test - y_pred
            #     me = np.mean(np.abs(residuals))
            #     df["mean_error"]=[me]
            #     df["expectation_error"] = [np.abs(np.mean(y_test)-quad_exp)]
            #     df["std_error"] = [np.abs(np.sqrt(np.var(y_test))-quad_std)]
            # if self.do_brute_check:
            #     print('DOING BRUTE CHECK')
            #     predictions = self.surrogate_predict(self.brute_check_sampler.samples, n_jobs=self.n_jobs)
            #     expectation = self.expectation_approx(predictions)
            #     double_sigma = self.double_sigma_approx(predictions)
            #     entropy_diff = self.relative_entropy(self.old_brute_check_predictions, predictions, num_bins=200)
            #     self.old_brute_check_predictions=predictions
            #     print('EXPECTATION', expectation)
            #     df["brute_expectation"]=[expectation]
            #     df["brute_double_sigma"]=[double_sigma]
            #     df["brute_entropy_diff"]=[entropy_diff]
            #     del predictions
                
            # if self.do_brute_force_sobol_indicies:
            #     print('doing brute sobol indicies')
            #     sobol_first_order, sobol_total_order = self.sobel_indicies_approx()
            #     for d, sfo in enumerate(sobol_first_order):
            #         df[f'brute_sobol_first_order_{d}'] = [sfo]
            #     for d, sto in enumerate(sobol_total_order):
            #         df[f'brute_sobol_total_order_{d}']= [sto]
            df.to_csv(os.path.join(batch_dir,fname), header=True, index=False)
            all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
            if os.path.exists(all_batch_info_path):
                df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
            else:
                df.to_csv(all_batch_info_path, mode='w', header=True, index=False)            
            del df
        
        if save_grid:
            self.save_grid(batch_dir, name=name)
        
    def merge_batch_info(self):
        assert self.base_run_dir
        dfs = []
        for dir_ in os.listdir(self.base_run_dir):
            if 'batch_' in dir_:
                batch_dir = os.path.join(self.base_run_dir, dir_)
                if os.path.exists(os.path.join(batch_dir, 'batch_info.csv')):
                    dfi = pd.read_csv(os.path.join(batch_dir, 'batch_info.csv'))
                    dfs.append(dfi)
        df = pd.concat(dfs)
        df.to_csv(os.path.join(self.base_run_dir,'batch_info.csv'), header=True, index=False)

        
    def save_grid(self,batch_dir, name=''):
        print('pickeling points')        
        with open(os.path.join(batch_dir,name+'train_points.pkl'), 'wb') as file:
            pickle.dump(self.train, file)
            
        with open(os.path.join(batch_dir,name+'virtual_boundary_points.pkl'), 'wb') as file:
            pickle.dump(self.virtual_boundary_points, file)
    
        with open(os.path.join(batch_dir,name+'anchor_boundary_points.pkl'), 'wb') as file:
            pickle.dump(self.anchor_boundary_points, file)

        with open(os.path.join(batch_dir,name+'infered_bound_points.pkl'), 'wb') as file:
            pickle.dump(self.bounds_points, file)
            
        with open(os.path.join(batch_dir,name+'infered_parent_points.pkl'), 'wb') as file:
            pickle.dump(self.parent_points, file)
        
        print('SAVING GRID')
        with open(os.path.join(batch_dir,name+'pysgpp_grid.txt'), 'w') as file:
            file.write(self.grid.serialize())
        print('SAVING SURPLUSES')
        self.alpha.toFile(os.path.join(batch_dir, name+'surpluses.mat'))

    def anchored_anova_firstorder_sobol(self, df_or_path, parameters, total_var, value_col='f', weight_col='w', tol=1e-6, min_slice_size=10):
        from scipy.spatial import KDTree
        """
        Estimate first-order Sobol indices using anchored ANOVA from sparse grid data.
        Uses normalized slice weights if weight column is present, otherwise estimates weights via nearest-neighbor spacing.

        Parameters:
        - df_or_path: pandas DataFrame or path to CSV file
        - parameters: list of column names representing input dimensions (e.g. ['x0', 'x1', 'x2'])
        - value_col: name of the column with function values
        - weight_col: name of the column with quadrature weights (optional)
        - tol: tolerance for matching anchor values in other dimensions
        - min_slice_size: minimum number of points required to compute a slice variance

        Returns:
        - sobol_indices: array of shape (len(parameters),), estimated first-order Sobol indices
        - anchor_fraction: array of shape (len(parameters),), fraction of the anchors that contributed to each index
        """
        df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
        d = len(parameters)
        sobol_matrix = []
        anchor_counts = np.zeros(d, dtype=int)

        for _, anchor in df.iterrows():
            sobol_per_dim = []

            for i in range(d):
                dim_i = parameters[i]
                other_dims = [p for j, p in enumerate(parameters) if j != i]

                # Select points where all other dimensions match the anchor (within tol)
                mask = np.ones(len(df), dtype=bool)
                for dim_j in other_dims:
                    mask &= np.abs(df[dim_j] - anchor[dim_j]) < tol

                slice_df = df[mask]
                if len(slice_df) < min_slice_size:
                    sobol_per_dim.append(np.nan)
                    continue

                fi_vals = slice_df[value_col].values

                if weight_col in df.columns:
                    wi_vals = slice_df[weight_col].values
                    wi_norm = wi_vals / np.sum(wi_vals)
                else:
                    xi_vals = slice_df[dim_i].values.reshape(-1, 1)
                    xi_tree = KDTree(xi_vals)
                    dists, _ = xi_tree.query(xi_vals, k=2)
                    wi_est = dists[:, 1]  # distance to nearest neighbor
                    wi_norm = wi_est / np.sum(wi_est)

                fi_mean = np.sum(wi_norm * fi_vals)
                fi_var = np.sum(wi_norm * (fi_vals - fi_mean)**2)
                sobol_per_dim.append(fi_var / total_var)
                anchor_counts[i] += 1

            sobol_matrix.append(sobol_per_dim)

        sobol_matrix = np.array(sobol_matrix)
        sobol_indices = np.nanmean(sobol_matrix, axis=0)

        print("Anchor contributions per dimension:", anchor_counts)
        return sobol_indices, np.array(anchor_counts)/len(df)


    def kde_weighted_mean_var(self, df_or_path, value_col='f', parameters=None, bandwidth=0.1, kernel='gaussian'):
        from sklearn.neighbors import KernelDensity

        """
        Compute weighted mean and variance using KDE-derived weights based on sampling density.

        Parameters:
        - df_or_path: pandas DataFrame or path to CSV file
        - value_col: name of the column with function values
        - parameters: list of column names representing input dimensions (optional; defaults to all except value_col)
        - bandwidth: bandwidth for KDE smoothing
        - kernel: KDE kernel type (e.g. 'gaussian', 'tophat', 'epanechnikov')

        Returns:
        - weighted_mean: KDE-weighted mean of the function values
        - weighted_var: KDE-weighted variance of the function values
        - weights: array of normalized KDE weights
        """
        df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
        if parameters is None:
            parameters = [col for col in df.columns if col != value_col]

        X = df[parameters].values
        f_vals = df[value_col].values

        # Fit KDE to input locations
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        kde.fit(X)
        log_density = kde.score_samples(X)
        density = np.exp(log_density)

        # Inverse density as weight (sparser regions get higher weight)
        raw_weights = 1.0 / (density + 1e-12)
        weights = raw_weights / np.sum(raw_weights)

        # Weighted mean and variance
        weighted_mean = np.sum(weights * f_vals)
        weighted_var = np.sum(weights * (f_vals - weighted_mean)**2)

        return weighted_mean, weighted_var, weights


    def quadrature_first_order_sobol(self, num_points=20):
        """
        Compute first-order Sobol index for target_dim using SG++ quadrature.

        Parameters:
        - grid: SG++ Grid object
        - alpha: SG++ DataVector of surpluses
        - dim: total number of dimensions
        - target_dim: index of variable to compute Sobol index for
        - num_points: number of fixed values for target_dim

        Returns:
        - S_i: first-order Sobol index for target_dim
        """
        # This method is artificially increasing with number of samples and it also often causes a segmentation fault that kills the main python process
        # It should be ran in a seperate process to isolate the process death. Most methods require pickeling. Subprocess python3 script is probably the best way
        # then the grid and alphas should be saved to file...
        # Total variance
        total_var = self.quadrature_variance()
        first_order_sobol = []
        for target_dim in range(self.dim):
            # Sweep over values of x_i
            xi_vals = np.linspace(0, 1, num_points)
            fi_vals = []

            for xi in xi_vals:
                # Create a function that fixes x_i and integrates over x_{-i}
                def f_fixed(x_rest):
                    x_full = np.zeros(self.dim)
                    x_full[target_dim] = xi
                    x_full[np.arange(self.dim) != target_dim] = x_rest
                    # p = pysgpp.DataVector(x_full.tolist())
                    return self.surrogate_predict([np.array(x_full)], space='unit_space') #op_eval.eval(self.alpha, p)

                # Create a grid over x_{-i}
                sub_dim = self.dim - 1
                sub_grid = pysgpp.Grid.createLinearGrid(sub_dim)
                level = int(np.ceil(np.log2(self.dim) + 2))
                # print(f'performing quad first order sobol with level {level} for {self.dim} dimensions')
                sub_grid.getGenerator().regular(level)
                sub_storage = sub_grid.getStorage()
                sub_alpha = pysgpp.DataVector(sub_storage.getSize())

                # Fill sub_alpha with evaluations of f_fixed
                for i in range(sub_storage.getSize()):
                    p_rest = [sub_storage.getPoint(i).getStandardCoordinate(di) for di in range(sub_dim)]
                    sub_alpha[i] = float(f_fixed(np.array(p_rest)))

                self.heirarchisation(sub_grid, sub_alpha)
                # pysgpp.createOperationHierarchisation(sub_grid).doHierarchisation(sub_alpha)
            
                # Integrate over x_{-i}
                print('debug, doing quadrature in quadrature_first order sobol')
                fi_vals.append(self.do_quadrature(sub_grid, sub_alpha))

            # Compute variance of f_i(x_i)
            # fi_mean = np.mean(fi_vals)
            fi_var = np.var(fi_vals, ddof=1)

            # Normalize
            S_i = fi_var / total_var
            first_order_sobol.append(S_i)
        return first_order_sobol

    def _do_quadrature(self, grid, alpha):
        quad = pysgpp.createOperationQuadrature(grid)
        print('debug, doing quadrature in quadrature_first order sobol')
        return quad.doQuadrature(alpha)
    
    def do_quadrature(self, *args, **kwargs):
        return self.safe_run(self._do_quadrature, *args, **kwargs)
    # def get_test_set(self, test_dir):
    #     print('RETRIVING TEST SET FROM', test_dir)        
    #     if os.path.exists(os.path.join(test_dir,'merged_runner_return.csv')):        
    #         df_test = pd.read_csv(os.path.join(test_dir,'merged_runner_return.csv'))
    #         print('got runner_return.csv')
    #     elif os.path.exists(os.path.join(test_dir, 'merged_runner_return.txt')):
    #         df_test = pd.read_csv(os.path.join(test_dir, 'merged_runner_return.txt'))
    #         print('got runner_return.txt')
    #     # elif os.path.exists(os.path.join(test_dir, 'runner_return.txt')):
    #     #     df_test = pd.read_csv(os.path.join(test_dir, 'runner_return.txt'))
    #     #     print('got runner_return.txt')    
    #     else:
    #         print('NO RUNNER RETURN FOUND, BEGINNIGN PARSING')
    #         finished_result = find_files(test_dir, 'GENE.finished')
    #         stopped_result = find_files(test_dir, 'stopped_by_monitor')
    #         result = finished_result + stopped_result
    #         run_dirs = [os.path.dirname(path) for path in result]
    #         if len(result) == 0:        
    #             raise FileNotFoundError('NO RUNNER RETURN PATH WAS FOUND IN:',test_dir,'\n ALSO THERE SEEM TO BE NO FINNISHED OR EARLY STOPPED GENE RUNS IN:',test_dir)
    #         else:
    #             outputs = []
    #             for i, run_dir in enumerate(run_dirs):
    #                 if i % 10 == 0:
    #                     print('NUMBER OF RUN_DIR PARSED:',i)
    #                 outputs.append(parse_run_dir(run_dir, parameters))
    #             with open(os.path.join(test_dir, 'merged_runner_return.txt'), 'w') as file:
    #                 lines = [runner_return_headder] + outputs
    #                 lines = [line+'\n' for line in lines]
    #                 file.writelines(lines)
    #             df_test = pd.read_csv(os.path.join(test_dir, 'merged_runner_return.txt'))
                
    #     test_x = np.array(df_test.iloc[:,0:-1].astype('float'))
    #     # print('debug l tx', len(test_x))
    #     test_y = np.array(df_test.iloc[:,-1].astype('float'))
    #     return test_x, test_y
    
        
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None


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

# #We don't want to run the function for every point so a wrapper function should brute_check to see if the point has been ran and if it has return that value
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