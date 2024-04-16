from .abstract_adapt_operation import *

class DimensionPriority(DimensionAdaptivity):

	def __init__(self, dim, tols, init_multiindex, max_level, level_to_nodes, spectral_op_obj):
		
		self._dim 				= dim
		self._tols 				= tols
		self._init_multiindex 	= init_multiindex
		self._max_level 		= max_level
		self._level_to_nodes 	= level_to_nodes

		self.__init_level_O = 2
		self.__init_level_B = 3

		self.__spectral_op_obj = spectral_op_obj

		self._edge_set_dna = np.zeros(2**dim - 1, dtype=int)

		self._multiindices_edge_set_to_del = []

		self._O 				= OrderedDict()
		self._E 				= OrderedDict() 
		self._local_dir_var 	= OrderedDict()
		self._key_O 			= 0
		self._key_E 			= 0

		self._max_dir_vars = np.zeros(2**dim - 1)

		self._multiindex_set 	= []
		self._init_no_points 	= 0

		self._stop_adaption = False

		self._local_basis_global 	= None
		self._local_basis_local 	= OrderedDict()

		self._multiindex_bin = Multiindex(self._dim).get_poly_mindex_binary(self._dim)

	@property
	def E(self):
		
		return self._E

	@property
	def edge_set_dna(self):
		
		return self._edge_set_dna

	def __find_keys(self, ordered_dict, value):

		key_of_interest = 0
		for key in list(ordered_dict.keys()):
			if ordered_dict[key].tolist() == value:
				key_of_interest = key

		return key_of_interest

	def update_dir_vars(self, multiindex):

		delta_coeff 	= self.__spectral_op_obj.get_spectral_coeff_delta(multiindex)
		curr_dir_var 	= self.__spectral_op_obj.get_all_dir_var_multiindex(self._multiindex_bin, delta_coeff, multiindex)

		self._local_dir_var[repr(multiindex.tolist())] = curr_dir_var

		print('MULTIINDEX', multiindex)
		print('curr_dir_var', curr_dir_var)

		for multiindex in self._multiindices_edge_set_to_del:
			del self._local_dir_var[repr(multiindex)]

	def get_edge_set_dna(self):

		local_dir_vars 	= np.array([local_dir_var for local_dir_var in list(self._local_dir_var.values())])

		for i, local_dir_var in enumerate(local_dir_vars.T):
			max_dir_d = np.max(local_dir_var)

			self._max_dir_vars[i] = max_dir_d
			self._edge_set_dna[i] = max_dir_d > self._tols[i]

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

		self.get_edge_set_dna()

		print('in preproc')

		print('local vars')
		print(self._local_dir_var)
		print('MAX VARS')
		print(self._max_dir_vars)
		print('DNA CURR EDGE SET')
		print(self._edge_set_dna)

		neighbors_edge_set = Multiindex(self._dim).get_successors_edge_set(self._multiindex_set, list(self._E.values()))

		no_diff_genes = np.zeros(len(neighbors_edge_set), dtype=int)

		for i, multiindex in enumerate(neighbors_edge_set):

			multiindex_dna 		= self.__spectral_op_obj.get_multiindex_contrib_all_dir(self._multiindex_bin, multiindex)
			no_diff_genes[i] 	= np.sum(multiindex_dna * self._edge_set_dna)

		# 	print 'FOR MINDEX', multiindex
		# 	print 'DNA', multiindex_dna
		# 	print 'AND EDGE SET DNA', self._edge_set_dna

		# print 'NO DIFF GENES', no_diff_genes

		candidate_multiindex = neighbors_edge_set[np.argmax(no_diff_genes)]

		self._key_E += 1
		self._E[self._key_E] = candidate_multiindex 

		local_basis_neighbor = np.array([self._get_no_1D_grid_points(n) - 1 for n in candidate_multiindex], dtype=int)
		self._update_local_basis(candidate_multiindex.tolist(), local_basis_neighbor)

		edge_set 							= [multiindex.tolist() for multiindex in list(self._E.values())]
		existing_keys 						= []
		self._multiindices_edge_set_to_del 	= []
		for mindex in edge_set:

			fwd_neighbors = Multiindex(self._dim).get_successors(mindex)

			is_in_edge_set = 1
			for neighbor in fwd_neighbors:
				if neighbor.tolist() not in edge_set:
					is_in_edge_set = 0
					break

			if is_in_edge_set:
				self._multiindices_edge_set_to_del.append(mindex)

		if self._multiindices_edge_set_to_del:
			for multiindex in self._multiindices_edge_set_to_del:

				key = self.__find_keys(self._E, multiindex)		

				self._key_O += 1
				self._O[self._key_O] = self._E[key]

				del self._E[key]

		self._multiindex_set.append(candidate_multiindex)

		print('candidate_multiindex', candidate_multiindex)
		print('MINDEICES TO DEL', self._multiindices_edge_set_to_del)

		return candidate_multiindex

	def do_one_adaption_step_postproc(self, candidate_multiindex):

		delta_coeff 	= self.__spectral_op_obj.get_spectral_coeff_delta(candidate_multiindex)
		curr_dir_var 	= self.__spectral_op_obj.get_all_dir_var_multiindex(self._multiindex_bin, delta_coeff, candidate_multiindex)

		self._local_dir_var[repr(candidate_multiindex.tolist())] = curr_dir_var

		for multiindex in self._multiindices_edge_set_to_del:
			del self._local_dir_var[repr(multiindex)]

	def check_termination_criterion(self):

		max_level = np.max(self._multiindex_set)
		if len(list(self._E.values())) == 0 or max_level >= self._max_level or np.sum(self._edge_set_dna) == 0:
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