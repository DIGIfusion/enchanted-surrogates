import numpy as np
from abc import ABCMeta, abstractmethod
from itertools import product
from collections import OrderedDict
from pickle import dump, load
from sg_lib.algebraic.multiindex import *

class AbstractOperation(object, metaclass=ABCMeta):
    _dim = 0

    @property
    def dim(self):

        return self._dim

    def _get_differences_sign(self, multiindex):

        differences_indices = OrderedDict()
        differences_signs   = OrderedDict()

        possible_diff_indices = np.zeros((self._dim, 2))

        for d in range(self._dim):
            possible_diff_indices[d][0] = multiindex[d]
            possible_diff_indices[d][1] = multiindex[d] - 1

        differences_indices_temp = np.array(list(product(*possible_diff_indices)), dtype=int)

        key_indices = 0
        for d in differences_indices_temp:

            if all(i >= 1 for i in d):
                differences_indices[key_indices] = d
                key_indices                     += 1                
                
        possible_differences_signs = np.zeros((self._dim, 2))

        for d in range(self._dim):

            if multiindex[d] == 1:
                possible_differences_signs[d][0] = 1
                possible_differences_signs[d][1] = 0
            else:
                possible_differences_signs[d][0] = 1
                possible_differences_signs[d][1] = -1

        differences_signs_temp = np.array(list(product(*possible_differences_signs)), dtype=int)

        key_signs = 0
        for element in differences_signs_temp:

            if np.prod(element) != 0:   
                differences_signs[key_signs] = np.prod(element)
                key_signs                   += 1

        return differences_indices, differences_signs

    def _get_multiindex_dict(self, multiindex_set):

        multiindex_dict = OrderedDict()

        for index, multiindex in enumerate(multiindex_set):
            multiindex_dict[repr(multiindex.tolist())] = index

        return multiindex_dict

    def update_sg_evals_all_lut(self, sg_point, func_eval):

        # sg_point[0] = np.round(sg_point[0], 5)

        self._sg_func_evals_all_lut[repr(sg_point.tolist())] = func_eval

    def update_sg_evals_multiindex_lut(self, multiindex, grid_obj):

        sg_points = grid_obj.get_fg_points_multiindex(multiindex, self._all_grid_points_1D)
        # sg_points = grid_obj.get_fg_points_multiindex(multiindex)

        # print('IN UPDATE')
        # print(sg_points)

        self._all_sg_points_LUT.append(sg_points)

        func_evals = []
        for sg_point in sg_points:
            # print(sg_point)

            #sg_point[0] = np.round(sg_point[0], 5)

            func_evals.append(self._sg_func_evals_all_lut[repr(sg_point.tolist())])

        # print('**************')

        self._fg_func_evals_multiindex_lut[repr(multiindex.tolist())] = func_evals
    
    def reset_datastructures(self):

        self._sg_func_evals_all_lut         = OrderedDict()  
        self._sg_func_evals_multiindex_lut  = OrderedDict()  

    def serialize_data(self, serialization_file):
        
        with open(serialization_file, "wb") as output_file:
            dump(self._sg_func_evals_all_lut, output_file)

        output_file.close()

    def unserialize_data(self, serialization_file):

        data = []
        with open(serialization_file, "rb") as input_file:
            while True:
                try: 
                    data.append(load(input_file))
                except EOFError: 
                    break
        
        input_file.close() 

        self._sg_func_evals_all_lut = data[-1]
        
    @abstractmethod
    def _eval_operation_fg(self):
        
        return

    @abstractmethod
    def eval_operation_delta(self):
        
        return 
 
    @abstractmethod
    def eval_operation_sg(self):
        
        return  