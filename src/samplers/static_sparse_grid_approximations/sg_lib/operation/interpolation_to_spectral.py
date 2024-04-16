from .onedim import *
from .abstract_operation import *
from ast import literal_eval

class InterpolationToSpectral(AbstractOperation):
	def __init__(self, dim, linear_growth_factor, left_bounds, right_bounds, weights, max_level, grid_obj):
		
		self._dim 					= dim
		self.__linear_growth_factor = linear_growth_factor
		self.__left_bounds 			= left_bounds
		self.__right_bounds 		= right_bounds
		self.__weights 				= weights
		self.__grid_obj 			= grid_obj
		
		self._sg_func_evals_all_lut 		= OrderedDict()
		self._fg_func_evals_multiindex_lut 	= OrderedDict()

		self._global_indices_dict   = OrderedDict()
		self._no_fg_grid_points     = 0

		self.__local_basis 	= None
		self.__global_basis = None

		self._all_grid_points_1D = []

		self._all_sg_points_LUT = []

		for d in range(dim):
			temp, _ = grid_obj.get_1D_points(max_level, left_bounds[d], right_bounds[d], weights[d])
			self._all_grid_points_1D.append(temp)

	
	@property
	def local_basis(self):

	    return self.__local_basis

	@property
	def global_basis(self):

	    return self.__global_basis

	@property
	def all_sg_points_LUT(self):

	    return self._all_sg_points_LUT

	def __eval_ND_orth_poly(self, degs, x):

		poly_eval = 1.
		for d in range(self._dim):
			poly_eval *= eval_1D_orth_poly(degs[d], self.__left_bounds[d], self.__right_bounds[d], self.__weights[d], x[d])

		return poly_eval

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
			pl 			= grid_1D_len - 1
			
			degrees__dim_d = []
			for p in range(pl + 1):
				degrees__dim_d.append(p)
				
			degrees_all.append(degrees__dim_d)

		tensorized_degrees = list(product(*degrees_all))

		return tensorized_degrees


	def __get_orth_poly_basis_global(self, multiindex_set):

		max_level_deg 			= self.__get_no_1D_grid_points(np.max(multiindex_set)) - 1
		orth_poly_basis_global 	= Multiindex(self._dim).get_poly_mindex(max_level_deg)

		return orth_poly_basis_global

	def get_global_basis(self, multiindex_set):

		max_level_deg 			= self.__get_no_1D_grid_points(np.max(multiindex_set)) - 1
		orth_poly_basis_global 	= Multiindex(self._dim).get_poly_mindex(max_level_deg)

		return orth_poly_basis_global

	# def __get_orth_poly_basis_local(self, multiindex):

	# 	if np.sum(multiindex) == self._dim:
	# 		tensorized_degrees = [[0 for i in range(self._dim)]]
	# 	elif repr(multiindex.tolist()) in self.__local_basis.keys():
	# 		tensorized_degrees = self.__local_basis[repr(multiindex.tolist())]
	# 	else:
			
	# 		degrees_all = []
	# 		for d in range(self._dim):
	# 			grid_1D_len = self.__get_no_1D_grid_points(multiindex[d])
	# 			pl 			= grid_1D_len - 1
				
	# 			degrees_dim_d = []
	# 			for p in range(pl + 1):
	# 				degrees_dim_d.append(p)
					
	# 			degrees_all.append(degrees_dim_d)

	# 		tensorized_degrees = list(product(*degrees_all))

	# 	return tensorized_degrees

	# def __get_orth_poly_basis_global(self, multiindex_set):

	#  	orth_poly_basis_global = self.__global_basis

	#  	return orth_poly_basis_global

	def __get_orth_poly_basis_active_set(self, active_set):

		active_set_basis = []

		for multiindex in active_set:
			degs = self.__get_orth_poly_basis_local(multiindex)
			
			for deg in degs:
				if deg not in active_set_basis:
					active_set_basis.append(deg)

		active_set_basis = np.array(active_set_basis, dtype=int)

		return active_set_basis

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
		
		spectral_coeff_delta 	= self.get_spectral_coeff_delta(curr_multiindex)
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

	# def _eval_operation_fg(self, curr_func_evals, multiindex, x):

	# 	interp_fg 	 	= 0.
	# 	poly_eval_all 	= []

	# 	# self._all_grid_points_1D, surplus_points_1D = grid_obj.get_1D_points(max_level, left_bounds[0], right_bounds[0], weights[0])

	# 	for d in range(self._dim):
	# 		index = multiindex[d] 

	# 		curr_no_points 		= self.__get_no_1D_grid_points(index)
	# 		#grid_1D, _ 			= self.__grid_obj.get_1D_points(index, self.__left_bounds[d], self.__right_bounds[d], self.__weights[d]) 
	# 		# grid_1D 			= grid_1D[0:curr_no_points]
	# 		barycentric_weights	= get_1D_barycentric_weights(grid_1D)

	# 		poly_eval = []
	# 		for j in range(len(grid_1D)):
	# 			p_eval = eval_1D_barycentric_interpolant(grid_1D, barycentric_weights, j, x[d])
	# 			poly_eval.append(p_eval)

	# 		poly_eval_all.append(poly_eval)

	# 	tensorized_basis_val = np.array([np.prod(interp_pair) for interp_pair in list(product(*poly_eval_all))], dtype=np.float64)

	# 	for i in range(len(tensorized_basis_val)):
	# 		interp_fg += tensorized_basis_val[i]*curr_func_evals[i]

	# 	return interp_fg

	def _eval_operation_fg(self, curr_func_evals, multiindex, x):

		interp_fg 	 	= 0.
		poly_eval_all 	= []

		for d in range(self._dim):
			index = multiindex[d] 

			curr_no_points 		= self.__get_no_1D_grid_points(index)
			grid_1D 			= self._all_grid_points_1D[d][0:curr_no_points]
			barycentric_weights	= get_1D_barycentric_weights(grid_1D)

			poly_eval = []
			for j in range(len(grid_1D)):
				p_eval = eval_1D_barycentric_interpolant(grid_1D, barycentric_weights, j, x[d])
				poly_eval.append(p_eval)

			poly_eval_all.append(poly_eval)

		tensorized_basis_val = np.array([np.prod(interp_pair) for interp_pair in list(product(*poly_eval_all))], dtype=np.float64)

		for i in range(len(tensorized_basis_val)):
			interp_fg += tensorized_basis_val[i]*curr_func_evals[i]

		return interp_fg

	# map interpolation to spectral projection
	def __get_spectral_coeff_local(self, curr_func_evals, multiindex):

		orth_multiindex_degs 	= self.__get_orth_poly_basis_local(multiindex)
		tensorized_leja_points 	= self.__grid_obj.get_fg_points_multiindex(multiindex, self._all_grid_points_1D)
		# tensorized_leja_points 	= self.__grid_obj.get_fg_points_multiindex(multiindex)

		assert len(orth_multiindex_degs) == len(tensorized_leja_points)

		rhs = np.zeros(len(orth_multiindex_degs))
		for i, leja_point in enumerate(tensorized_leja_points):
			rhs[i] = self._eval_operation_fg(curr_func_evals, multiindex, leja_point)

		orth_basis_matrix = np.zeros((len(orth_multiindex_degs), len(orth_multiindex_degs)))
		for i, leja_point in enumerate(tensorized_leja_points):
			for j, poly_degs in enumerate(orth_multiindex_degs):
				orth_basis_matrix[i, j] = self.__eval_ND_orth_poly(poly_degs, leja_point)

		spectral_coeff_fg = np.linalg.solve(orth_basis_matrix, rhs)

		return spectral_coeff_fg

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

			curr_func_evals 		= self._fg_func_evals_multiindex_lut[repr(differences.tolist())]
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

		return spectral_coeff, orth_poly_basis_global

	def eval_operation_delta(self, curr_multiindex, multiindex_set, x):

		interp_delta = 0.
		
		differences_indices, differences_signs 	= self._get_differences_sign(curr_multiindex)

		keys_differences 	= list(differences_indices.keys())
		keys_signs 			= list(differences_signs.keys()) 

		for key in keys_differences:
			differences 	= differences_indices[key]
			curr_func_evals = self._fg_func_evals_multiindex_lut[repr(differences.tolist())]

			sign 			= differences_signs[key] 
			interp_delta 	+= self._eval_operation_fg(curr_func_evals, differences, x)*sign

		return interp_delta

	def eval_operation_sg(self, multiindex_set, x):

		interp_sg = 0.

		for multiindex in multiindex_set:
			interp_delta = self.eval_operation_delta(multiindex, multiindex_set, x) 
			interp_sg 	+= interp_delta

		return interp_sg

	def eval_operation_sg_ct(self, func_evals, signs, multiindex_set, x):

		interp_sg = 0.

		for j, multiindex in enumerate(multiindex_set):
			interp_sg 	+= self._eval_operation_fg(func_evals[j], multiindex, x)*signs[j]

			
		return interp_sg

	def get_mean(self, spectral_coeff):
		
		mean = spectral_coeff[0]

		return mean

	def get_variance(self, spectral_coeff):
		
		var = np.sum([spectral_coeff[i]**2 for i in range(1, len(spectral_coeff))])

		return var

	def get_multiindex_contrib_all_dir(self, multiindex_bin, multiindex):
		
		multiindex_dna 			= np.zeros(2**self._dim - 1, dtype=int)
		all_dims_contributions 	= OrderedDict()

		# differences_indices, differences_signs 	= self._get_differences_sign(multiindex)
		# orth_poly_basis_loc_all 				= []

		# keys_differences 	= differences_indices.keys()
		# keys_signs 			= differences_signs.keys() 

		# first_multiindex 	= differences_indices[0]
		# largest_basis 		= self.__get_orth_poly_basis_local(first_multiindex)
		# largest_basis 		= [list(basis) for basis in largest_basis]

		# orth_poly_basis_loc_all.append(largest_basis)

		# temp = [0 for d in range(self._dim)]
		# for d in range(1, len(keys_differences)):

		# 	curr_basis = self.__get_orth_poly_basis_local(differences_indices[d])
		# 	curr_basis = [list(basis) for basis in curr_basis]

		# 	orth_poly_basis_loc_updated = []
		# 	for basis in largest_basis:

		# 		if basis in curr_basis:
		# 			orth_poly_basis_loc_updated.append(basis)
		# 		else:
		# 			orth_poly_basis_loc_updated.append(temp)

		# 	orth_poly_basis_loc_all.append(orth_poly_basis_loc_updated)

		# orth_poly_basis_local = np.zeros((len(largest_basis), self._dim), dtype=int)

		orth_poly_basis_local = self.__get_orth_poly_basis_local(multiindex)

		# for i, basis in enumerate(orth_poly_basis_loc_all):

		# 	sign 	= differences_signs[i]
		# 	basis 	= np.array(basis, dtype=int)

		# 	orth_poly_basis_local += basis*sign

		for d in range(2**self._dim - 1):

			mindex_local = []
			local_mindex = multiindex_bin[d]

			for p, orth_poly_basis_loc in enumerate(orth_poly_basis_local):

				if np.count_nonzero(orth_poly_basis_loc) == np.count_nonzero(local_mindex) == np.count_nonzero(orth_poly_basis_loc*local_mindex):
					mindex_local.append(p)

			all_dims_contributions[d] = len(mindex_local)

			if len(mindex_local) >= 1:
 		 		multiindex_dna[d] = 1
 			#multiindex_dna[d] = len(mindex_local)

		return multiindex_dna

	def get_directional_var_active_set(self, multiindex_set, active_set):
		
		directional_var = np.zeros(self._dim)

		orth_poly_basis_active 	= self.__get_orth_poly_basis_active_set(active_set)
		spectral_coeff 			= np.zeros(len(orth_poly_basis_active))

		for multiindex in active_set:
			spectral_coeff_active_set 	= self.__get_spectral_coeff_active_set(multiindex, multiindex_set, orth_poly_basis_active)
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
		spectral_coeff_local 	= self.get_spectral_coeff_delta(multiindex)

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

	def get_local_var_level_active_set_all(self, multiindex_bin, global_multiindex_set, multiindex):
		
		directional_var_all = np.zeros(2**self._dim - 1)

		orth_poly_basis_local 	= self.__get_orth_poly_basis_local(multiindex)
		spectral_coeff_local 	= self.get_spectral_coeff_delta(multiindex)

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

	def get_local_total_var_all(self, multiindex):
		
		directional_var_all = np.zeros(self._dim)

		orth_poly_basis_local 	= self.__get_orth_poly_basis_local(multiindex)
		spectral_coeff_local 	= self.get_spectral_coeff_delta(multiindex)

		orth_poly_basis_local = np.array(orth_poly_basis_local, dtype=int)

		for d in range(self._dim):
			
			mindex = []
			for i in range(len(orth_poly_basis_local)):
				if orth_poly_basis_local[i][d] != 0:
					mindex.append(i)

			directional_var_all[d] = np.sum([spectral_coeff_local[j]**2 for j in mindex])

		return directional_var_all

	def get_total_var_level_active_set(self, global_multiindex_set, multiindex):
		
		spectral_coeff_local = self.get_spectral_coeff_delta(multiindex)

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
