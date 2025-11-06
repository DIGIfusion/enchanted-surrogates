import os
import numpy as np
import pandas as pd
from enchanted_surrogates.samplers.base_sampler import Sampler
from sg_lib.grid.grid import Grid
from sg_lib.algebraic.multiindex import Multiindex
from sg_lib.operation.interpolation_to_spectral import InterpolationToSpectral
from sg_lib.operation.spectral_projection import SpectralProjection
from sg_lib.adaptivity.spectral_scores import SpectralScores
import time
import pickle
import warnings

class SensitivityDrivenSparseGrid(Sampler):
    def __init__(self, parameters, bounds, tol=1e-6, max_level=20, *args, **kwargs):
        """
        """
        print('INITALIZING SENTITIVITY DRIVEN SPARSE GRID SAMPLER')
        self.base_run_dir = kwargs.get('base_run_dir',None)
        self.parameters = parameters
        self.bounds = bounds
        self.dim = len(parameters)
        self.budget = kwargs.get('budget',1)
        self.do_write_batch_info = kwargs.get('do_write_batch_info', True)
        self.write_batch_info_every_x_samples = kwargs.get('write_batch_info_every_x_samples', 1)
        self.num_samples_at_last_write = 0
        # sparse grid setup
        # here, we consider a uniform input distribution, thus the bounds are [0, 1]^dim
        self.left_bounds     = np.zeros(self.dim)
        self.right_bounds     = np.ones(self.dim)
        # sparse grids always begin at level 1
        self.grid_level         = 1
        # not important; keep it 1 for interpolation
        self.level_to_nodes     = 1
        # probability density function; in this case, we have a uniform density in [0, 1]^dim
        # since the stochastic inputs are independent, it has a product structure
        self.weights         = [lambda x: 1. for i in range(self.dim)]

        # tolerance used in the adaptive algorithm
        if tol == 0:
            self.ignore_finish_criteria = True
        else:
            self.ignore_finish_criteria = False
        self.tols         = float(tol)*np.ones(self.dim + 1)
        # maximum level reachable by the sparse grid
        self.max_level     = max_level

        # first multiindex in K is (1, 1, ... , 1)
        self.init_multiindex = np.ones(self.dim, dtype=int)
        self.current_multiindices = [self.init_multiindex]
        # create objects to do sensitivity-driven adaptve sparse grid interpolation
        # grid object
        print('MAKING THE GRID OBJECT')
        self.Grid_obj = Grid(self.dim, self.grid_level, self.level_to_nodes, self.left_bounds, self.right_bounds, self.weights)
        # multiindex object
        print('MAKING THE MULTIINDEX OBJECT')
        self.Multiindex_obj = Multiindex(self.dim)

        # interpolation object
        print('MAKING THE INTERPOLATION OBJECT')
        self.InterpToSpectral_obj = InterpolationToSpectral(self.dim, self.level_to_nodes, self.left_bounds, self.right_bounds, self.weights, self.max_level, self.Grid_obj)
        print('MAKING THE SPECTRAL PROJECTION OBJECT')
        self.SpectralProjection_obj = SpectralProjection(self.dim, self.level_to_nodes, self.left_bounds, self.right_bounds, self.weights, self.max_level, self.Grid_obj)
        
        # adaptivity object; the most important thing is the refinement indicator; see the paper
        # see also the implementation
        print('MAKING THE ADAPTIVITY OBJECT')
        self.Adaptivity_obj = SpectralScores(self.dim, self.tols, self.init_multiindex, self.max_level, self.level_to_nodes, self.InterpToSpectral_obj)

        # add initial multiindex to the multiindex set, aka, K = {(1, 1, ..., 1)}
        self.init_multiindex_set = self.Multiindex_obj.get_std_total_degree_mindex(self.grid_level)
        self.current_multiindex_set = self.init_multiindex_set
        
        self.global_index = 0
        
        self.current_grid_points = None
        
        self.batch_number = 0
        print('FINISHED INITALIZING SENTITIVITY DRIVEN SPARSE GRID SAMPLER')

    def get_initial_samples(self):
        init_grid_points     = self.Grid_obj.get_std_sg_surplus_points(self.init_multiindex_set)
        # init_no_points         = self.Grid_obj.get_no_fg_grid_points(self.current_multiindex_set)
        self.current_grid_points = [init_grid_points]
        self.InterpToSpectral_obj.get_local_global_basis(self.Adaptivity_obj)
        
        samples = [{key: value for key, value in zip(self.parameters, sample)} for sample in init_grid_points]
        
        samples = [{**samp, 'index': ind} for samp, ind in zip(samples, range(self.submitted, self.submitted + len(samples)))]
        self.batch_number += 1
        self.submitted += len(samples)
        print('debug init samples', type(samples), samples)
        return samples
    
    def get_data(self):
        data_path = os.path.join(self.base_run_dir, 'enchanted_dataset.csv')
        df = pd.read_csv(data_path)
        output_col = [col for col in df.columns if 'output' in col]
        if len(output_col)>1:
            warnings.warn(f'There is more than one output column, using the first one: {output_col}')
        data = df.set_index('index')[output_col[0]].to_dict()
        print('debug data', type(data), data)
        return data
    
    def _scale(self, samples):
        """Scale samples from [0,1] to bounds."""
        scaled = np.empty_like(samples)
        for i, (low, high) in enumerate(self.bounds):
            scaled[:, i] = samples[:, i] * (high - low) + low
        return scaled

    def get_next_samples(self):
        """
        Returns the next batch of samples as a list of dicts.
        Each dict includes 'source', 'index', and parameter values.
        """
        if self.batch_number == 0:
            print('GETTING INITIAL SAMPLES')
            return self.get_initial_samples()
        elif self.batch_number == 1:
            print('GETTING BATCH 1')
            data = self.get_data()
            print('debug sg_point_set', self.current_grid_points)
            print('debug multiindices', self.current_multiindices)
            for sg_point_set, multiindex in zip(self.current_grid_points, self.current_multiindices):
                for sg_point in sg_point_set:
                    print('debug global index', self.global_index, data[self.global_index])
                    sg_val = data[self.global_index]
                    print('debug sg_point sg_val', sg_point, sg_val)
                    self.global_index += 1
                    self.InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

                self.InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, self.Grid_obj)
            
            # batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
            # self.write_batch_info(batch_dir=batch_dir)
            
            print('debug batch number', self.batch_number)
            
            print('INITALISING ADAPTATION')
            self.Adaptivity_obj.init_adaption()
            
            self.current_multiindices = self.Adaptivity_obj.do_one_adaption_step_preproc()
            
            if len(self.current_multiindices) == 0:
                while len(self.current_multiindices) == 0:
                    if not self.do_one_adaption_step_postproc():
                        return None
                    # self.Adaptivity_obj.do_one_adaption_step_postproc(self.current_multiindices)
                    self.current_multiindices = self.Adaptivity_obj.do_one_adaption_step_preproc()
            
            print('debug current_multiindices', self.current_multiindices)
            samples = []
            self.current_grid_points = []
            for multiindex in self.current_multiindices:
                grid_points_i = self.Grid_obj.get_sg_surplus_points_multiindex(multiindex)
                print('debug grid_points_i', type(grid_points_i), grid_points_i)
                self.current_grid_points.append(grid_points_i)
                samples_i = [{key: value for key, value in zip(self.parameters, sample)} for sample in grid_points_i]
                print('debug samples_i', samples_i)
                samples_i = [{**samp, 'index': ind} for samp, ind in zip(samples_i, range(self.submitted, self.submitted + len(samples_i)))]
                print('debug samples_i', samples_i)
                self.submitted += len(samples_i)
                
                samples.extend(samples_i)
            
            self.batch_number += 1
            print('debug samples','type', type(samples), samples)
            return samples
            
        else:
            print(f'GETTING BATCH {self.batch_number}')
            batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')            
            data = self.get_data()
            for sg_point_set, multiindex in zip(self.current_grid_points, self.current_multiindices):
                for sg_point in sg_point_set:
                    sg_val = data[self.global_index]
                    print('debug sg_point sg_val', sg_point, sg_val)
                    self.global_index += 1
                    self.InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

                self.InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, self.Grid_obj)
            
            self.current_multiindex_set = self.Adaptivity_obj.multiindex_set          
            if self.do_write_batch_info:
                if self.submitted - self.num_samples_at_last_write >= self.write_batch_info_every_x_samples or self.batch_number in [0,1,2,3]:
                    # Now the surrogate is trained we can write batch info
                    self.write_batch_info(batch_dir=batch_dir)
                    self.num_samples_at_last_write = self.submitted    
            
            if not self.do_one_adaption_step_postproc():
                return None
            # self.Adaptivity_obj.do_one_adaption_step_postproc(self.current_multiindices)
            #**********DO NOT ADD ANY MORE FINISHED CHECKS, IT CAN CAUSE PREMATURE FINISH**************
            self.Adaptivity_obj.check_termination_criterion()
            finished_adapt = self.Adaptivity_obj.stop_adaption
            print('finished_adapt?',finished_adapt)
            print('******************')
            if finished_adapt and not self.ignore_finish_criteria:
                self.write_batch_info(batch_dir=batch_dir)
                return None
            
            self.current_multiindices = self.Adaptivity_obj.do_one_adaption_step_preproc()
            if len(self.current_multiindices) == 0:
                while len(self.current_multiindices) == 0:
                    if not self.do_one_adaption_step_postproc():
                        return None
                    self.Adaptivity_obj.check_termination_criterion()
                    finished_adapt = self.Adaptivity_obj.stop_adaption                    
                    print('finished_adapt?',finished_adapt)
                    print('******************')
                    if finished_adapt and not self.ignore_finish_criteria:
                        self.write_batch_info(batch_dir=batch_dir)
                        return None
                    # self.Adaptivity_obj.do_one_adaption_step_postproc(self.current_multiindices)
                    self.current_multiindices = self.Adaptivity_obj.do_one_adaption_step_preproc()
                    
            samples = []
            self.current_grid_points = []
            for multiindex in self.current_multiindices:
                grid_points_i = self.Grid_obj.get_sg_surplus_points_multiindex(multiindex)
                self.current_grid_points.append(grid_points_i)
                samples_i = [{key: value for key, value in zip(self.parameters, sample)} for sample in grid_points_i]
                samples_i = [{**samp, 'index': ind} for samp, ind in zip(samples_i, range(self.submitted, self.submitted + len(samples_i)))]
                samples.extend(samples_i)
                self.submitted += len(samples_i)
        
            self.batch_number += 1
            print('debug samples','type', type(samples), samples)
            return samples

    def get_batch_info(self):
        spectral_coeff, orth_poly_basis_global = self.InterpToSpectral_obj.get_spectral_coeff_sg(self.current_multiindex_set)
        mean = self.SpectralProjection_obj.get_mean(spectral_coeff)
        var = self.SpectralProjection_obj.get_variance(spectral_coeff)
        
        batch_info = {
            'num_samples': self.submitted,
            'mean': mean,
            'std': np.sqrt(var)
        }
        
        total_order = self.SpectralProjection_obj.get_total_sobol_indices(spectral_coeff, self.current_multiindex_set)
        first_order = self.SpectralProjection_obj.get_first_order_sobol_indices(spectral_coeff, self.current_multiindex_set)
        for i, param in enumerate(self.parameters):
            batch_info[f'{param}_sobolF'] = first_order[i]
            batch_info[f'{param}_sobolT'] = total_order[i]
        return batch_info

    def write_batch_info(self, batch_dir):
        start = time.time()
        print('WRITING BATCH INFO')
        batch_info = self.get_batch_info()
        
        df = pd.DataFrame({k:[v] for k,v in batch_info.items()})
        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
                
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)
        df.to_csv(os.path.join(batch_dir,'batch_info.csv'), index=False)         
        
        with open(os.path.join(batch_dir,'multiindex_set.pkl'), 'wb') as file:
            pickle.dump(self.Adaptivity_obj.multiindex_set,file)
        
        with open(os.path.join(batch_dir,'spectral_coeff.pkl'), 'wb') as file:
            pickle.dump(self.Adaptivity_obj.multiindex_set,file)
        
        print('WRITING BATCH INFO TOOK:', time.time()-start, 'sec')
        return batch_info
    
    def do_one_adaption_step_postproc(self): 
        try:
            self.Adaptivity_obj.do_one_adaption_step_postproc(self.current_multiindices)
            return True
        except Exception as e:
            print('ADAPTIVITY POSTPROC FAILED. STOPPING SAMPLING. \n',e)
            return False
            
    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None

# # discrete approximation of the Pearson correlation coefficient
# # see the Peherstorfer, Willcox, Gunzburger 2016 paper for the formula
# def compute_corr_coeff(hi_fi_evals, lo_fi_evals):

#     mean_hi_fi = np.mean(hi_fi_evals)
#     mean_lo_fi = np.mean(lo_fi_evals)

#     std_hi_fi = np.std(hi_fi_evals, ddof=1)
#     std_lo_fi = np.std(lo_fi_evals, ddof=1)

#     rho_12 = np.sum(np.array([(hi_fi_eval - mean_hi_fi)*(lo_fi_eval - mean_lo_fi) \
#                 for hi_fi_eval, lo_fi_eval in zip(hi_fi_evals, lo_fi_evals)]))/(std_hi_fi*std_lo_fi*(len(hi_fi_evals) - 1.))

#     return rho_12

# # high-fidelity model for this test
# def hi_fi_model(x):
#     test = np.cos(np.pi + 1.0*x[0] + 0.55*x[1] + 0.8*x[2] + 0.1*x[3]) + 1.0
#     return test
    
# if __name__ == '__main__':

#     # take the grid points corresponding to the first multiindex
#     # init_no_points         = Grid_obj.get_no_fg_grid_points(init_multiindex_set)

#     # begin the adaptive process
#     # sg_evals_all = np.load('data/sg_evals_all.npy')

#     global_index = 0

#     # first step, do the initial subspace which contains 1 point
    
#     # adaptivity begins here; see paper, especially the algorithms, for more details
#     Adaptivity_obj.init_adaption()

#     prev_len         = len(init_no_points)
#     total_len         = 1

#     total_no_adapt_steps = 20

#     for n in range(total_no_adapt_steps):
        
#         # new_multiindices = Adaptivity_obj.do_one_adaption_step_preproc()

#         # for multiindex in new_multiindices:
#         #     new_grid_points = Grid_obj.get_sg_surplus_points_multiindex(multiindex)
#         #     total_len         += len(new_grid_points)

#         #     for sg_point in new_grid_points:
#         #         global_index += 1

#         #         sg_val = sg_evals_all[global_index]
            
#         #         InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

#         #     InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, Grid_obj)
            
#         # Adaptivity_obj.do_one_adaption_step_postproc(new_multiindices)
#         # Adaptivity_obj.check_termination_criterion()

#         # finished_adapt = Adaptivity_obj.stop_adaption

#         # print(n + 1, finished_adapt)
#         # print(n + 1, total_len)
#         # print('******************')

#     one_more_ref_step = 1
#     for i in range(one_more_ref_step):

#         print('new adapt step')
#         new_multiindices = Adaptivity_obj.do_one_adaption_step_preproc()

#         if len(new_multiindices):
#             one_more_ref_step += 1

#         print('multiindices adaptivity')
#         print(new_multiindices)

#         for multiindex in new_multiindices:
#             new_grid_points = Grid_obj.get_sg_surplus_points_multiindex(multiindex)
#             total_len         += len(new_grid_points)

#             mapped_sg_points = Grid_obj.map_std_sg_surplus_points(new_grid_points, left_bounds, right_bounds)

#             for sg_point, mapped_sg_point in zip(new_grid_points, mapped_sg_points):

#                 print('new simulation')
#                 print(mapped_sg_point)

#     exit(0)

#     print('adaptivity done after', str(no_adapt_steps), 'steps')
#     print('grid size =', total_len, 'sparse grid points')

#     InterpToSpectral_obj.get_local_global_basis(Adaptivity_obj)

#     adapt_sg_lo_fi_model = lambda x: InterpToSpectral_obj.eval_operation_sg(Adaptivity_obj.multiindex_set, x)    

#     # here, we compute the Pearson correlation coefficient between the high- and low-fidelity model, which will be relevant for doing  MFMC
#     np.random.seed(9812788)
#     corr_coeff_no_samples     = 100
#     corr_coeff_samples         = np.random.uniform(0, 1, size=(corr_coeff_no_samples, dim))

#     f_eval = [hi_fi_model(sample) for sample in corr_coeff_samples]

#     f_approx = np.zeros(corr_coeff_no_samples)
#     for i, sample in enumerate(corr_coeff_samples):
#         f_approx[i] = adapt_sg_lo_fi_model(sample)

#     corr_coeff = compute_corr_coeff(f_eval, f_approx)

#     print('corr coeff(hi-fi model, lo-fi surrogate) = ',corr_coeff)