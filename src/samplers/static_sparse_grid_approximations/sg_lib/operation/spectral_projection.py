from .onedim import *
from .abstract_operation import *
from ast import literal_eval

class SpectralProjection(AbstractOperation):
	def __init__(self, dim, linear_growth_factor, left_bounds, right_bounds, weights, max_level, grid_obj):
		
		self._dim 			= dim
		self.__linear_growth_factor = linear_growth_factor
		self.left_bounds 	= left_bounds
		self.right_bounds 	= right_bounds
		self.weights 		= weights
		self.__grid_obj 	= grid_obj
		
		self._sg_func_evals_all_lut 		= OrderedDict()
		self._fg_func_evals_multiindex_lut 	= OrderedDict()

		self._global_indices_dict   = OrderedDict()
		self._no_fg_grid_points     = 0

		self.__local_basis 	= None
		self.__global_basis = None

		self._all_grid_points_1D, surplus_points_1D = grid_obj.get_1D_points(max_level, left_bounds[0], right_bounds[0], weights[0])


	def __get_no_1D_grid_points(self, level):

		no_points = 0

		if self.__linear_growth_factor == 'sym':
			no_points = 2*level - 1
		else:
			if self.__linear_growth_factor == 1:
				no_points = level
			elif self.__linear_growth_factor >= 2:
				if level == 1:
					no_points = 1
				else:
					no_points = self.__linear_growth_factor*level - 1
			else:
				raise NotImplementedError

		return no_points

	def __get_orth_poly_basis_local(self, multiindex):

		degrees_all = []

		for d in range(self._dim):
			grid_1D_len = self.__get_no_1D_grid_points(multiindex[d])
			pl 			= (grid_1D_len - 1)/2
			
			degrees__dim_d = []
			for p in range(pl + 1):
				degrees__dim_d.append(p)
				
			degrees_all.append(degrees__dim_d)

		tensorized_degrees = list(product(*degrees_all))

		return tensorized_degrees


	def __get_orth_poly_basis_global(self, multiindex_set):

		max_level_deg 			= (self.__get_no_1D_grid_points(np.max(multiindex_set)) - 1)/2
	 	orth_poly_basis_global 	= Multiindex(self._dim).get_poly_mindex(max_level_deg)

	 	return orth_poly_basis_global

	def __get_orth_poly_basis_active_set(self, active_set):

		active_set_basis = []

		for multiindex in active_set:
			degs = self.__get_orth_poly_basis_local(multiindex)
			
			for deg in degs:
				if deg not in active_set_basis:
					active_set_basis.append(deg)

		active_set_basis = np.array(active_set_basis, dtype=int)

		return active_set_basis

	def __get_spectral_coeff_local(self, func_evals, multiindex):

		orth_poly_all 		= []
		quad_weights_all 	= []

		for d in range(self._dim):
			index = multiindex[d] 

			curr_no_points 		= self.__get_no_1D_grid_points(index)
			grid_1D 			= self._all_grid_points_1D[0:curr_no_points]
			barycentric_weights	= get_1D_barycentric_weights(grid_1D)

			quad_weights = compute_1D_quad_weights(grid_1D, self.left_bounds[d], self.right_bounds[d], self.weights[d])
			quad_weights_all.append(quad_weights)

			pl = (len(grid_1D) - 1)/2
			
			orth_poly__dim_d_all = []
			for p in range(pl + 1):

				orth_poly__dim_d = []
				for q in range(len(grid_1D)):
					orth_poly = eval_1D_orth_poly(p, self.left_bounds[d], self.right_bounds[d], self.weights[d], grid_1D[q])
					orth_poly__dim_d.append(orth_poly)

				orth_poly__dim_d_all.append(orth_poly__dim_d)			
			orth_poly_all.append(orth_poly__dim_d_all)

		tensorized_weights 		= [np.prod(weights_pair) for weights_pair in list(product(*quad_weights_all))]
		tensorized_poly_bases 	= [[np.prod(poly_pair) for poly_pair in list(product(*poly_pairs))] for poly_pairs in list(product(*orth_poly_all))]

		spectral_coeff_fg = np.zeros(len(tensorized_poly_bases))
		for p, tensorized_poly_basis in enumerate(tensorized_poly_bases):
			for q in range(len(tensorized_poly_basis)):
				spectral_coeff_fg[p] += func_evals[q]*tensorized_poly_basis[q]*tensorized_weights[q]

		return spectral_coeff_fg

	def __get_spectral_coeff_local_dict(self, func_evals, multiindex, tensorized_degrees):

		spectral_coeff_fg = self.__get_spectral_coeff_local(func_evals, multiindex)

		spectral_coeff_dict = OrderedDict()
		for degrees, coeff_fg in zip(tensorized_degrees, spectral_coeff_fg):
			spectral_coeff_dict[repr(degrees)] = coeff_fg

		return spectral_coeff_dict

	def __get_spectral_coeff_global(self, func_evals, multiindex, orth_poly_basis):

		spectral_coeff 			= np.zeros(len(orth_poly_basis))
		tensorized_degrees 		= self.__get_orth_poly_basis_local(multiindex)
		curr_spectral_coeff 	= self.__get_spectral_coeff_local_dict(func_evals, multiindex, tensorized_degrees)

		if type(orth_poly_basis) is np.ndarray:
			orth_poly_basis = orth_poly_basis.tolist()

		for local_pos in list(curr_spectral_coeff.keys()):
			local_pos_list = list(literal_eval(local_pos))

			if local_pos_list in orth_poly_basis:
				index 					= orth_poly_basis.index(local_pos_list)
				spectral_coeff[index] 	= curr_spectral_coeff[local_pos]

		return spectral_coeff

	def __get_spectral_coeff_delta_dict(self, curr_multiindex, multiindex_set):
		
		spectral_coeff_delta 	= self.get_spectral_coeff_delta(curr_multiindex, multiindex_set)
		orth_poly_basis_local 	= self.__get_orth_poly_basis_local(curr_multiindex)

		spectral_coeff_dict = OrderedDict()
		for degrees, coeff_fg in zip(orth_poly_basis_local, spectral_coeff_delta):
			spectral_coeff_dict[repr(degrees)] = coeff_fg

		return spectral_coeff_dict

	def __get_spectral_coeff_active_set(self, curr_multiindex, multiindex_set, orth_poly_basis):

		spectral_coeff 			= np.zeros(len(orth_poly_basis))
		curr_spectral_coeff 	= self.__get_spectral_coeff_delta_dict(curr_multiindex, multiindex_set)
		orth_poly_basis 		= orth_poly_basis.tolist()

		for local_pos in list(curr_spectral_coeff.keys()):
			local_pos_list = list(literal_eval(local_pos))

			if local_pos_list in orth_poly_basis:
				index 					= orth_poly_basis.index(local_pos_list)
				spectral_coeff[index] 	= curr_spectral_coeff[local_pos]

		return spectral_coeff

	def _eval_operation_fg(self, curr_func_evals, multiindex, x):
		
		spectral_fg = 0.

		tensorized_degrees 		= self.__get_orth_poly_basis_local(multiindex)
		spectral_coeff_fg 		= self.__get_spectral_coeff_local(curr_func_evals, multiindex)
		poly_eval_all 			= []

		for p in tensorized_degrees:

			poly_eval__dim_d = []
			for d in range(self._dim):
				poly_eval = eval_1D_orth_poly(p[d], self.left_bounds[d], self.right_bounds[d], self.weights[d], x[d])
				poly_eval__dim_d.append(poly_eval)

			poly_eval_all.append(poly_eval__dim_d)

		tensorized_poly_eval = np.array([np.prod(poly_pair) for poly_pair in poly_eval_all])

		for i, basis_val in enumerate(tensorized_poly_eval):
			spectral_fg += spectral_coeff_fg[i] * basis_val

		return spectral_fg

	def get_local_global_basis(self, adaptivity_obj):

		self.__local_basis 	= adaptivity_obj.local_basis_local
		self.__global_basis = adaptivity_obj.local_basis_global

	def get_spectral_coeff_delta(self, curr_multiindex):
		
		orth_poly_basis_local 	= np.array(self.__get_orth_poly_basis_local(curr_multiindex), dtype=int)
		no_spectral_coeff 		= len(orth_poly_basis_local)
		spectral_coeff_delta 	= np.zeros(no_spectral_coeff)

		differences_indices, differences_signs 	= self._get_differences_sign(curr_multiindex)

		keys_differences 	= list(differences_indices.keys())
		keys_signs 			= list(differences_signs.keys()) 
		
		for key in keys_differences:
			differences 	= differences_indices[key]
			curr_func_evals = self._fg_func_evals_multiindex_lut[repr(differences.tolist())]
			
			sign 					= differences_signs[key] 
			spectral_coeff_delta 	+= self.__get_spectral_coeff_global(curr_func_evals, \
																	differences, orth_poly_basis_local)*sign
			
		return spectral_coeff_delta			

	def get_spectral_coeff_sg(self, multiindex_set):

		orth_poly_basis_global 	= self.__get_orth_poly_basis_global(multiindex_set)
		no_spectral_coeff 		= len(orth_poly_basis_global)
		spectral_coeff 			= np.zeros(no_spectral_coeff)

		for index, multiindex in enumerate(multiindex_set):
			differences_indices, differences_signs = self._get_differences_sign(multiindex)

			keys_differences 	= list(differences_indices.keys())
			keys_signs 			= list(differences_signs.keys()) 

			for key in keys_differences:
				differences 	= differences_indices[key]
				curr_func_evals = self._fg_func_evals_multiindex_lut[repr(differences.tolist())]
				sign 			= differences_signs[key] 

				curr_spectral_coeff = self.__get_spectral_coeff_global(curr_func_evals, differences, orth_poly_basis_global)
				spectral_coeff 		+= sign*curr_spectral_coeff

		return spectral_coeff
		
	def eval_operation_delta(self, curr_multiindex, multiindex_set, x):
		
		spectral_delta = 0.

		differences_indices, differences_signs 	= self._get_differences_sign(curr_multiindex)

		keys_differences 	= list(differences_indices.keys())
		keys_signs 			= list(differences_signs.keys()) 

		for key in keys_differences:
			differences 	= differences_indices[key]
			curr_func_evals = self._fg_func_evals_multiindex_lut[repr(differences.tolist())]
			
			sign 			= differences_signs[key] 
			spectral_delta 	+= self._eval_operation_fg(curr_func_evals, differences, x)*sign

		return spectral_delta

	def eval_operation_sg(self, multiindex_set, x):

		spectral_sg = 0.

		for multiindex in multiindex_set:
			spectral_delta 	= self.eval_operation_delta(multiindex, multiindex_set, x) 
			spectral_sg 	+= spectral_delta

		return spectral_sg

	def get_mean(self, spectral_coeff):
		
		mean = spectral_coeff[0]

		return mean

	def get_variance(self, spectral_coeff):
		
		var = np.sum([spectral_coeff[i]**2 for i in range(1, len(spectral_coeff))])

		return var

	def get_multiindex_contrib_all_dir(self, multiindex_bin, multiindex):
		
		multiindex_dna 			= np.zeros(2**self._dim - 1, dtype=int)
		all_dims_contributions 	= OrderedDict()
		orth_poly_basis_local 	= self.__get_orth_poly_basis_local(multiindex)

		for d in range(2**self._dim - 1):

			mindex_local = []
			local_mindex = multiindex_bin[d]

			for p, orth_poly_basis_loc in enumerate(orth_poly_basis_local):

				if np.count_nonzero(orth_poly_basis_loc) == np.count_nonzero(local_mindex) == np.count_nonzero(orth_poly_basis_loc*local_mindex):
					mindex_local.append(p)

			all_dims_contributions[d] = len(mindex_local)

			if len(mindex_local) >= 1:
 				multiindex_dna[d] = 1
 		#	multiindex_dna[d] = len(mindex_local)

		return multiindex_dna

	def get_directional_var_active_set(self, global_multiindex_set, active_set):
		
		directional_var = np.zeros(self._dim)

		orth_poly_basis_active 	= self.__get_orth_poly_basis_active_set(active_set)
		spectral_coeff 			= np.zeros(len(orth_poly_basis_active))

		for multiindex in active_set:
			spectral_coeff_active_set 	= self.__get_spectral_coeff_active_set(multiindex, global_multiindex_set, orth_poly_basis_active)
			spectral_coeff 				+= spectral_coeff_active_set

		for d in range(self._dim):
			
			mindex = []
			for i in range(len(orth_poly_basis_active)):
				if orth_poly_basis_active[i][d] != 0:
					mindex.append(i)

 			directional_var[d] = np.sum([spectral_coeff[j]**2 for j in mindex])

		return directional_var

	def get_local_var_level_active_set(self, global_multiindex_set, multiindex):
		
		directional_var = np.zeros(self._dim + 1)

		orth_poly_basis_local 	= self.__get_orth_poly_basis_local(multiindex)
		spectral_coeff_local 	= self.get_spectral_coeff_delta(multiindex, multiindex_set)

		mindex_interact = []
		for d in range(self._dim):
			
			mindex_main = []
			for i in range(len(orth_poly_basis_local)):
				if orth_poly_basis_local[i][d] != 0 and np.count_nonzero(orth_poly_basis_local[i]) == 1:
					mindex_main.append(i)
				elif orth_poly_basis_local[i][d] != 0 and np.count_nonzero(orth_poly_basis_local[i]) >= 2:
					mindex_interact.append(i)

 			directional_var[d] = np.sum([spectral_coeff_local[j]**2 for j in mindex_main])

 		directional_var[self._dim] = np.sum([spectral_coeff_local[j]**2 for j in mindex_interact])

		return directional_var

	def get_local_total_var_all(self, multiindex):
		
		directional_var_all = np.zeros(self._dim)

		orth_poly_basis_local 	= self.__get_orth_poly_basis_local(multiindex)
		spectral_coeff_local 	= self.get_spectral_coeff_delta(multiindex, None)

		orth_poly_basis_local = np.array(orth_poly_basis_local, dtype=int)

		for d in range(self._dim):
			
			mindex = []
			for i in range(len(orth_poly_basis_local)):
				if orth_poly_basis_local[i][d] != 0:
					mindex.append(i)

 			directional_var_all[d] = np.sum([spectral_coeff_local[j]**2 for j in mindex])

		return directional_var_all

	def get_local_var_level_active_set_all(self, multiindex_bin, global_multiindex_set, multiindex):
		
		directional_var_all = np.zeros(2**self._dim - 1)

		orth_poly_basis_local 	= self.__get_orth_poly_basis_local(multiindex)
		spectral_coeff_local 	= self.get_spectral_coeff_delta(multiindex, global_multiindex_set)

		orth_poly_basis_local = np.array(orth_poly_basis_local, dtype=int)

		for d in range(2**self._dim - 1):

			mindex_local = []
			local_mindex = multiindex_bin[d]

			for p, orth_poly_basis_loc in enumerate(orth_poly_basis_local):

				if np.count_nonzero(orth_poly_basis_loc) == np.count_nonzero(local_mindex) == np.count_nonzero(orth_poly_basis_loc*local_mindex):
					mindex_local.append(p)

			if len(mindex_local) >= 1:
	 			directional_var_all[d] = np.sum([spectral_coeff_local[j]**2 for j in mindex_local])

		return directional_var_all

	def get_total_var_level_active_set(self, global_multiindex_set, multiindex):
		
		spectral_coeff_local = self.get_spectral_coeff_delta(multiindex, global_multiindex_set)

		total_var_level = np.sum(spectral_coeff_local[1:]**2)

		return total_var_level

	def get_all_dir_var_multiindex(self, multiindex_bin, spectral_coeff, multiindex):
		
		all_dir_var_multiindex = np.zeros(2**self._dim - 1)
		orth_poly_basis_local = self.__get_orth_poly_basis_local(multiindex)

		for d in range(2**self._dim - 1):

			mindex_local = []
			local_mindex = multiindex_bin[d]

			for p, orth_poly_basis_loc in enumerate(orth_poly_basis_local):

				if np.count_nonzero(orth_poly_basis_loc) == np.count_nonzero(local_mindex) == np.count_nonzero(orth_poly_basis_loc*local_mindex):
					mindex_local.append(p)

			if len(mindex_local) >= 1:
 				all_dir_var_multiindex[d] = np.sum([spectral_coeff[j]**2 for j in mindex_local])

		return all_dir_var_multiindex

	def get_first_order_sobol_indices(self, spectral_coeff, multiindex_set):
		
		sobol_indices = np.zeros(self._dim)

		orth_poly_basis_global 	= self.__get_orth_poly_basis_global(multiindex_set)
		Var 					= np.sum([spectral_coeff[i]**2 for i in range(1, len(spectral_coeff))])

		for d in range(self._dim):
			
			mindex = []
			for i in range(len(orth_poly_basis_global)):
				if orth_poly_basis_global[i][d] != 0 and np.count_nonzero(orth_poly_basis_global[i]) == 1:
					mindex.append(i)

 			sobol_indices[d] = np.sum([spectral_coeff[j]**2 for j in mindex])/Var

		return sobol_indices

	def get_total_sobol_indices(self, spectral_coeff, multiindex_set):
		
		sobol_indices = np.zeros(self._dim)

		orth_poly_basis_global 	= self.__get_orth_poly_basis_global(multiindex_set)
		Var 					= np.sum([spectral_coeff[i]**2 for i in range(1, len(spectral_coeff))])

		for d in range(self._dim):
			
			mindex = []
			for i in range(len(orth_poly_basis_global)):
				if orth_poly_basis_global[i][d] != 0:
					mindex.append(i)

 			sobol_indices[d] = np.sum([spectral_coeff[j]**2 for j in mindex])/Var

		return sobol_indices

	def get_all_sobol_indices(self, multiindex_bin, spectral_coeff, multiindex_set):
		
		sobol_indices = np.zeros(2**self._dim - 1)

		orth_poly_basis_global 	= self.__get_orth_poly_basis_global(multiindex_set)
		Var 					= np.sum([spectral_coeff[i]**2 for i in range(1, len(spectral_coeff))])

		for d in range(2**self._dim - 1):

			mindex_local = []
			local_mindex = multiindex_bin[d]

			for p, orth_poly_basis_loc in enumerate(orth_poly_basis_global):

				if np.count_nonzero(orth_poly_basis_loc) == np.count_nonzero(local_mindex) == np.count_nonzero(orth_poly_basis_loc*local_mindex):
					mindex_local.append(p)

			if len(mindex_local) >= 1:
 				sobol_indices[d] = np.sum([spectral_coeff[j]**2 for j in mindex_local])/Var

		return sobol_indices

	def serialize_results(self, E, Var, Sobol_indices, serialization_file):
	    
	    with open(serialization_file, "wb") as output_file:
	    	data = [E, Var, Sobol_indices]
	        dump(data, output_file)

	    output_file.close()

	def unserialize_results(self, serialization_file):

	    with open(serialization_file, "rb") as input_file:
	    	E, Var, Sobol_indices = load(input_file)
	           
	    input_file.close() 

	    return E, Var, Sobol_indices