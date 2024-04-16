from .abstract_adapt_operation import *

class DimensionPriorityExprapolation(DimensionAdaptivity):

	def __init__(self, dim, tols, init_multiindex, max_level, level_to_nodes):
		
		self._dim 				= dim
		self._tols 				= tols
		self._init_multiindex 	= init_multiindex
		self._max_level 		= max_level
		self._level_to_nodes 	= level_to_nodes

		self.__init_level_O = 2
		self.__init_level_B = 3

		self._multiindices_edge_set_to_del = []

		self._O 				= OrderedDict()
		self._E 				= OrderedDict() 
		self._key_O 			= 0
		self._key_E 			= 0

		self._max_dir_vars = np.zeros(2**dim - 1)

		self._multiindex_set 	= []
		self._init_no_points 	= 0

		self._stop_adaption = False

		self._local_basis_global 	= None
		self._local_basis_local 	= OrderedDict()

	@property
	def E(self):
		
		return self._E

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

		neighbors_edge_set = Multiindex(self._dim).get_successors_edge_set(self._multiindex_set, list(self._E.values()))

		return neighbors_edge_set

	def do_one_adaption_step_postproc(self):

		pass

	def check_termination_criterion(self):

		max_level = np.max(self._multiindex_set)
		if len(list(self._E.values())) == 0 or max_level >= self._max_level:
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