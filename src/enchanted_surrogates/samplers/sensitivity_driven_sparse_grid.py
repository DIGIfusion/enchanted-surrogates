import os
import numpy as np
import pandas as pd
from enchanted_surrogates.samplers.base_sampler import Sampler
from sg_lib.grid.grid import Grid
from sg_lib.algebraic.multiindex import Multiindex
from sg_lib.operation.interpolation_to_spectral import InterpolationToSpectral
from sg_lib.adaptivity.spectral_scores import SpectralScores

import warnings

class SensitivityDrivenSparseGrid(Sampler):
    def __init__(self, parameters, bounds,*args, **kwargs):
        """
        """
        print('INALIZING SOBOL INDICES SAMPLER')
        self.parameters = parameters
        self.bounds = bounds
        self.dim = len(parameters)
   
        # sparse grid setup
        # here, we consider a uniform input distribution, thus the bounds are [0, 1]^dim
        self.left_bounds 	= np.zeros(self.dim)
        self.right_bounds 	= np.ones(self.dim)
        # sparse grids always begin at level 1
        self.grid_level 		= 1
        # not important; keep it 1 for interpolation
        self.level_to_nodes 	= 1
        # probability density function; in this case, we have a uniform density in [0, 1]^dim
        # since the stochastic inputs are independent, it has a product structure
        self.weights 		= [lambda x: 1. for i in range(dim)]

        # tolerance used in the adaptive algorithm
        self.tols 		= 1e-6*np.ones(self.dim + 1)
        # maximum level reachable by the sparse grid
        self.max_level 	= 20

        # first multiindex in K is (1, 1, ... , 1)
        self.init_multiindex = np.ones(self.dim, dtype=int)
        
        # create objects to do sensitivity-driven adaptve sparse grid interpolation
        # grid object
        self.Grid_obj = Grid(self.dim, self.grid_level, self.level_to_nodes, self.left_bounds, self.right_bounds, self.weights)
        # multiindex object
        self.Multiindex_obj = Multiindex(self.dim)

        # interpolation object
        self.InterpToSpectral_obj = InterpolationToSpectral(self.dim, self.level_to_nodes, self.left_bounds, self.right_bounds, self.weights, self.max_level, self.Grid_obj)

        # adaptivity object; the most important thing is the refinement indicator; see the paper
        # see also the implementation
        self.Adaptivity_obj = SpectralScores(self.dim, self.tols, self.init_multiindex, self.max_level, self.level_to_nodes, self.InterpToSpectral_obj)

        # add initial multiindex to the multiindex set, aka, K = {(1, 1, ..., 1)}
        self.init_multiindex_set = self.Multiindex_obj.get_std_total_degree_mindex(self.grid_level)
        
    def get_initial_samples(self):
        init_grid_points 	= self.Grid_obj.get_std_sg_surplus_points(self.init_multiindex_set)
    	init_no_points 		= self.Grid_obj.get_no_fg_grid_points(self.init_multiindex_set)
        
	    self.InterpToSpectral_obj.get_local_global_basis(self.Adaptivity_obj)

    	init_grid_points 	= self.Grid_obj.get_std_sg_surplus_points(self.init_multiindex_set)
     
        samples = [{key: value for key, value in zip(self.parameters, sample)} for sample in init_grid_points]
        return samples
    
    def get_data(self):
        data_path = os.path.join(self.base_run_dir, 'enchanted_dataset.csv')
        df = pd.read_csv(data_path)
        output_col = [col for col in df.columns if 'output' in col]
        if len(output_col)>1:
            warnings.warn(f'There is more than one output column, using the first one: {output_col}')
        data = df.set_index('index')[output_col[0]].to_dict()
        return data
    
    def _scale(self, samples):
        """Scale samples from [0,1] to bounds."""
        scaled = np.empty_like(samples)
        for i, (low, high) in enumerate(self.bounds):
            scaled[:, i] = samples[:, i] * (high - low) + low
        return scaled

    def get_next_samples(self, data_df=None):
        """
        Returns the next batch of samples as a list of dicts.
        Each dict includes 'source', 'index', and parameter values.
        """
        if self.batch_number == 0:
            samples = self.get_initial_samples(self)
        else:
            None
        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def write_batch_info(self, batch_dir, data_df=None):
        df = pd.DataFrame({k:[v] for k,v in batch_info.items()})
        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
                
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)
        df.to_csv(os.path.join(batch_dir,'batch_info.csv'), index=False)         
        return batch_info
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None




# discrete approximation of the Pearson correlation coefficient
# see the Peherstorfer, Willcox, Gunzburger 2016 paper for the formula
def compute_corr_coeff(hi_fi_evals, lo_fi_evals):

	mean_hi_fi = np.mean(hi_fi_evals)
	mean_lo_fi = np.mean(lo_fi_evals)

	std_hi_fi = np.std(hi_fi_evals, ddof=1)
	std_lo_fi = np.std(lo_fi_evals, ddof=1)

	rho_12 = np.sum(np.array([(hi_fi_eval - mean_hi_fi)*(lo_fi_eval - mean_lo_fi) \
				for hi_fi_eval, lo_fi_eval in zip(hi_fi_evals, lo_fi_evals)]))/(std_hi_fi*std_lo_fi*(len(hi_fi_evals) - 1.))

	return rho_12

# high-fidelity model for this test
def hi_fi_model(x):
	test = np.cos(np.pi + 1.0*x[0] + 0.55*x[1] + 0.8*x[2] + 0.1*x[3]) + 1.0
	return test
    
if __name__ == '__main__':

	# take the grid points corresponding to the first multiindex
	# init_no_points 		= Grid_obj.get_no_fg_grid_points(init_multiindex_set)

	# begin the adaptive process
	# sg_evals_all = np.load('data/sg_evals_all.npy')

	global_index = 0

	# first step, do the initial subspace which contains 1 point
	for sg_point in init_grid_points:
		sg_val = sg_evals_all[global_index]
		InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

	InterpToSpectral_obj.update_sg_evals_multiindex_lut(init_multiindex, Grid_obj)
	
	# adaptivity begins here; see paper, especially the algorithms, for more details
	Adaptivity_obj.init_adaption()

	prev_len 		= len(init_no_points)
	total_len 		= 1

	total_no_adapt_steps = 20

	for n in range(total_no_adapt_steps):
		
		new_multiindices = Adaptivity_obj.do_one_adaption_step_preproc()

		for multiindex in new_multiindices:
			new_grid_points = Grid_obj.get_sg_surplus_points_multiindex(multiindex)
			total_len 		+= len(new_grid_points)

			for sg_point in new_grid_points:
				global_index += 1

				sg_val = sg_evals_all[global_index]
			
				InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

			InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, Grid_obj)
			
		Adaptivity_obj.do_one_adaption_step_postproc(new_multiindices)
		Adaptivity_obj.check_termination_criterion()

		finished_adapt = Adaptivity_obj.stop_adaption

		print(n + 1, finished_adapt)
		print(n + 1, total_len)
		print('******************')

	one_more_ref_step = 1
	for i in range(one_more_ref_step):

		print('new adapt step')
		new_multiindices = Adaptivity_obj.do_one_adaption_step_preproc()

		if len(new_multiindices):
			one_more_ref_step += 1

		print('multiindices adaptivity')
		print(new_multiindices)

		for multiindex in new_multiindices:
			new_grid_points = Grid_obj.get_sg_surplus_points_multiindex(multiindex)
			total_len 		+= len(new_grid_points)

			mapped_sg_points = Grid_obj.map_std_sg_surplus_points(new_grid_points, left_bounds, right_bounds)

			for sg_point, mapped_sg_point in zip(new_grid_points, mapped_sg_points):

				print('new simulation')
				print(mapped_sg_point)

	exit(0)

	print('adaptivity done after', str(no_adapt_steps), 'steps')
	print('grid size =', total_len, 'sparse grid points')

	InterpToSpectral_obj.get_local_global_basis(Adaptivity_obj)

	adapt_sg_lo_fi_model = lambda x: InterpToSpectral_obj.eval_operation_sg(Adaptivity_obj.multiindex_set, x)	

	# here, we compute the Pearson correlation coefficient between the high- and low-fidelity model, which will be relevant for doing  MFMC
	np.random.seed(9812788)
	corr_coeff_no_samples 	= 100
	corr_coeff_samples 		= np.random.uniform(0, 1, size=(corr_coeff_no_samples, dim))

	f_eval = [hi_fi_model(sample) for sample in corr_coeff_samples]

	f_approx = np.zeros(corr_coeff_no_samples)
	for i, sample in enumerate(corr_coeff_samples):
		f_approx[i] = adapt_sg_lo_fi_model(sample)

	corr_coeff = compute_corr_coeff(f_eval, f_approx)

	print('corr coeff(hi-fi model, lo-fi surrogate) = ',corr_coeff)