from .abstract_adapt_operation import *

class DimensionPriority(DimensionAdaptivity):

	def __init__(self, dim, tol, init_multiindex, max_level, level_to_nodes, spectral_op_obj):
		
		self._dim 				= dim
		self._tol 				= tol
		self._init_multiindex 	= init_multiindex
		self._max_level 		= max_level
		self._level_to_nodes 	= level_to_nodes

		self.__init_level_O = 2
		self.__init_level_B = 3

		self.__spectral_op_obj = spectral_op_obj

		self._multiindices_edge_set_to_del = []

		self._O 				= OrderedDict()
		self._E 				= OrderedDict() 
		self._local_dir_var 	= OrderedDict()
		self._local_dir_var_max = OrderedDict()
		self._key_O 			= 0
		self._key_E 			= 0

		self._curr_max_dir_var 			= 0.
		self._curr_max_dir_var_indices 	= []

		self._multiindex_set 	= []
		self._init_no_points 	= 0

		self._stop_adaption = False

		self._local_basis_global 	= None
		self._local_basis_local 	= OrderedDict()

		self._multiindex_bin = Multiindex(self._dim).get_poly_mindex_binary(self._dim)

	@property
	def E(self):
		
		return self._E

	def __find_keys(self, ordered_dict, value):

		key_of_interest = 0
		for key in list(ordered_dict.keys()):
			if ordered_dict[key].tolist() == value.tolist():
				key_of_interest = key

		return key_of_interest

	def __update_dir_var_max(self):

		unsorted_dir_var_max 	= OrderedDict()
		self._local_dir_var_max = OrderedDict()

		for multiindex in list(self._E.values()):
			curr_dir_var = self._local_dir_var[repr(multiindex.tolist())]

			unsorted_dir_var_max[np.max(curr_dir_var)] = multiindex

		keys = list(unsorted_dir_var_max.keys())
		keys.sort(reverse=True)

		for key in keys:
			self._local_dir_var_max[key] = unsorted_dir_var_max[key] 

	def update_dir_vars(self, multiindex):

		delta_coeff 	= self.__spectral_op_obj.get_spectral_coeff_delta(multiindex)
		curr_dir_var 	= self.__spectral_op_obj.get_all_dir_var_multiindex(self._multiindex_bin, delta_coeff, multiindex)

		print('Multiindex', multiindex)
		print(curr_dir_var)


		self._local_dir_var[repr(multiindex.tolist())] = curr_dir_var

	def init_adaption(self):

		self._key_O = -1
		self._key_E = -1

		init_O = Multiindex(self._dim).get_std_total_degree_mindex(self.__init_level_O)
		init_E = Multiindex(self._dim).get_std_total_degree_mindex_level(self.__init_level_B)

		self._local_basis_local[repr(self._init_multiindex)] 	= self._get_local_hierarchical_basis(self._init_multiindex)
		self._local_basis_global 								= self._get_local_hierarchical_basis(self._init_multiindex)

		for mindex in init_O:
			self._key_O 			+= 1
			self._O[self._key_O] 	= mindex

			self._multiindex_set.append(mindex)

			local_basis_neighbor = np.array([self._get_no_1D_grid_points(n) - 1 for n in mindex], dtype=int)
			self._update_local_basis(mindex.tolist(), local_basis_neighbor)

		for mindex in init_E:
			self._key_E 			+= 1
			self._E[self._key_E] 	= mindex

			self._multiindex_set.append(mindex)

			local_basis_neighbor = np.array([self._get_no_1D_grid_points(n) - 1 for n in mindex], dtype=int)
			self._update_local_basis(mindex.tolist(), local_basis_neighbor)

	def do_one_adaption_step_preproc(self):

		self.__update_dir_var_max()

		max_var_vec 					= np.zeros(2**self._dim - 1, dtype=int)
		self._curr_max_dir_var_indices 	= []

		curr_multiindex_set = [multiindex.tolist() for multiindex in self._multiindex_set]

		for local_dir_var in list(self._local_dir_var_max.keys()):

			self._curr_max_dir_var 			= local_dir_var
			curr_max_multiindex 			= self._local_dir_var_max[self._curr_max_dir_var]
			curr_max_multiindex_local_vars 	= self._local_dir_var[repr(curr_max_multiindex.tolist())]

			print('MAX DIR VAR', self._curr_max_dir_var)
			print('INDEX MAX', np.flipud(np.argsort(list(self._local_dir_var_max.keys()))))

			max_var_index 				= np.argmax(curr_max_multiindex_local_vars)
			max_var_vec[max_var_index] 	= 1

			self._curr_max_dir_var_indices.append(max_var_index)

			if np.count_nonzero(self._local_dir_var[repr(curr_max_multiindex.tolist())]) >= 2 and max_var_index >= self._dim:
				max_indv_var_index 				= np.argmax(curr_max_multiindex_local_vars[0:self._dim])
				max_var_vec[max_indv_var_index] = 1
				self._curr_max_dir_var_indices.append(max_indv_var_index)

			fwd_neighbors_curr_max_multiindex 	= Multiindex(self._dim).get_successors(curr_max_multiindex)
			admissible_fwd_neighbors 			= []

			for multiindex in fwd_neighbors_curr_max_multiindex:
				if Multiindex(self._dim).is_admissible(self._multiindex_set, multiindex) and multiindex.tolist() not in curr_multiindex_set:
					admissible_fwd_neighbors.append(multiindex)

			if admissible_fwd_neighbors:

				activated_dir = np.zeros(len(admissible_fwd_neighbors), dtype=int)
				for i, multiindex in enumerate(admissible_fwd_neighbors):
					activated_dir_multiindex 	= self.__spectral_op_obj.get_multiindex_contrib_all_dir(self._multiindex_bin, multiindex)
					activated_dir[i] 			= np.sum(activated_dir_multiindex * max_var_vec)

				print('curr multiindex set', self._multiindex_set)
				print('max multiindex', curr_max_multiindex)
				print('and its admissible neighbors', admissible_fwd_neighbors)
				print('activated_dir', activated_dir)

				candidate_multiindex_indices 	= np.argwhere(activated_dir == np.amax(activated_dir)).flatten().tolist()
				candidate_multiindex 			= admissible_fwd_neighbors[np.argmax(activated_dir)]

				print('all candidate_multiindex', candidate_multiindex_indices)

				break

		self._key_E += 1
		self._E[self._key_E] = candidate_multiindex 

		local_basis_neighbor = np.array([self._get_no_1D_grid_points(n) - 1 for n in candidate_multiindex], dtype=int)
		self._update_local_basis(candidate_multiindex.tolist(), local_basis_neighbor)

		edge_set 							= [multiindex.tolist() for multiindex in list(self._E.values())]
		existing_keys 						= []
		self._multiindices_edge_set_to_del 	= []

		print('ASSOC VARIANCES BEFORE', self._local_dir_var[repr(curr_max_multiindex.tolist())])

		for index in self._curr_max_dir_var_indices:
			self._local_dir_var[repr(curr_max_multiindex.tolist())][index] = 0.

		if np.sum(self._local_dir_var[repr(curr_max_multiindex.tolist())]) == 0:
			print('TO DEL', curr_max_multiindex)
			print('ASSOC VARIANCES AFTER', self._local_dir_var[repr(curr_max_multiindex.tolist())])

			self._multiindices_edge_set_to_del.append(curr_max_multiindex)
			del self._local_dir_var_max[self._curr_max_dir_var]

		if self._multiindices_edge_set_to_del:
			for multiindex in self._multiindices_edge_set_to_del:

				key = self.__find_keys(self._E, multiindex)		

				self._key_O += 1
				self._O[self._key_O] = self._E[key]

				del self._E[key]
				del self._local_dir_var[repr(multiindex.tolist())]

		self._multiindex_set.append(candidate_multiindex)

		print('****************************************')

		return candidate_multiindex

	def do_one_adaption_step_postproc(self, candidate_multiindex):

		delta_coeff 	= self.__spectral_op_obj.get_spectral_coeff_delta(candidate_multiindex)
		curr_dir_var 	= self.__spectral_op_obj.get_all_dir_var_multiindex(self._multiindex_bin, delta_coeff, candidate_multiindex)

		self._local_dir_var[repr(candidate_multiindex.tolist())] = curr_dir_var

		# self.__update_dir_var_max()

	def check_termination_criterion(self):

		max_level = np.max(self._multiindex_set)
		if len(list(self._E.values())) == 0 or max_level >= self._max_level or self._curr_max_dir_var <= self._tol:
			self._stop_adaption = True

	def serialize_data(self, serialization_file):
		
		with open(serialization_file, "wb") as output_file:
			data = [self._key_O, self._O, self._key_E, self._E, self._key_local_error, self._local_error, self._multiindex_set, \
							self._local_basis_local, self._local_basis_global]
			dump(data, output_file)

		output_file.close()

	def unserialize_data(self, serialization_file):

		with open(serialization_file, "rb") as input_file:
			self._key_O, self._O, self._key_E, self._E, self._key_local_error, self._local_error, self._multiindex_set, \
							self._local_basis_local, self._local_basis_global = load(input_file)

		input_file.close()