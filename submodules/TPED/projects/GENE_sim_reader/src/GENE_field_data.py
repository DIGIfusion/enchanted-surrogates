import struct
import os
import numpy as np
import xarray as xr
from os.path import getsize

from TPED.projects.GENE_sim_reader.utils.fieldlib import fieldfile
from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC
from TPED.projects.GENE_sim_reader.src.GENE_parameters_data import GeneParameters as GP

# List of field quantities and their corresponding indices
FIELD_QUANTITIES = ['field_phi', 'field_apar', 'field_bpar']
TIME_SAMPLER_METHODS = ['absolute', 'percentage']

class GeneField:
    def __init__(self, field_filepath: str):
        self.field_filepath = field_filepath
        # Convert field filepath to parameter filepath and load parameters
        param_filepath = GFC(self.field_filepath).switch_suffix_file('parameters')
        self.param_dict = GP(param_filepath).parameters_filepath_to_dict()
        self.field = fieldfile(self.field_filepath, self.param_dict)
    




    def field_filepath_to_xarray(
            self, 
            time_criteria: (str|list) = 'last', 
            input_fields: (str|list) = 'all', 
            time_sampler_method: str = 'absolute'):
        
        field_dict = self.field_filepath_to_dict(time_criteria, input_fields, time_sampler_method)
        return self.field_dict_to_xarray(field_dict)

    
    
    def field_dict_to_xarray(self, field_dict: dict):
        """
        Convert field dictionary to xarray dataset.
        
        Args:
        - field_dict (dict): A dictionary containing field data and relevant information.
        
        Returns:
        - xarray.Dataset: An xarray dataset containing field data.
        """
        filepath = field_dict.pop('filepath', None)
        directory = os.path.dirname(filepath)
        suffix = GFC(filepath).suffix_from_filepath()

        # Create a new dataset with trimmed data
        field_dataset = xr.Dataset(
            coords={
                'time': np.array(field_dict['time']),
                'zgrid': np.array(field_dict['zgrid'])
            }
        )

        for field in FIELD_QUANTITIES:
            if (field in field_dict) and (len(field_dict[field]) > 0):
                field_data = np.array(field_dict[field])
                field_dataset[field] = (['time', 'zgrid'], field_data)

        field_dataset.attrs['filepath'] = filepath
        field_dataset.attrs['directory'] = directory
        field_dataset.attrs['suffix'] = suffix

        return field_dataset




    
    def field_filepath_to_dict(
        self, 
        time_criteria: (str|list) = 'last', 
        input_fields: (str|list) = 'all', 
        time_sampler_method: str = 'absolute'
    ) -> dict:
        """
        Reads field data from a binary file and returns a dictionary containing relevant information.

        Args:
        - time_criteria (str): Specifies the time instance(s) to extract. Options:
            - 'last': Extracts data at the last available time.
            - 'first': Extracts data at the first available time.
        - input_fields (str): Specifies the field column(s) to extract. Options:
            - 'all': Extracts all fields.
            - str: Extracts the specified field column (phi OR apar OR bpar).
            - list: Extracts multiple specified field columns (i.e. ['phi', 'bpar']).

        Returns:
        - dict: A dictionary containing field data and relevant information.
        """

        # Initializing the field dictionary with default values
        field_dict = {'filepath': self.field_filepath, 'time': [],
                           'field_phi': [], 'field_apar': [], 
                           'field_bpar': [], 'zgrid': []}

        time_values = self._time_sampler(time_criteria, time_sampler_method)
        field_dict['time'] = time_values
        
        # Sample field names based on input_fields parameter
        filtered_field_names = self._sample_name_ind(FIELD_QUANTITIES, input_fields)

        # Extracting field data
        output_field_dict = self._extract_field_data(field_dict, time_values, filtered_field_names)

        return output_field_dict


    def field_filepath_to_time_trace(self, 
                                     time_criteria: (str|list) = 'last', 
                                     input_fields: (str|list) = 'all', 
                                     time_sampler_method: str = 'absolute'):
        

        return None 

    ##########################################
    ###### Private Helper Functions ##########
    ##########################################

    def _extract_field_data(self, input_field_dict:dict, time_values:list, sampled_field_names:list) -> dict:

        field_dict = input_field_dict.copy()

        dz = float(2.0)/float(self.field.nz)
        ntot = self.field.nz*self.field.nx
        zgrid = np.arange(ntot)/float(ntot-1)*(2*self.field.nx-dz)-self.field.nx
        field_dict['zgrid'] = zgrid

        if 'n0_global' in self.param_dict:
            phase_fac = -np.e**(-2.0*np.pi*(0.0+1.0J)*self.param_dict['n0_global']*self.param_dict['q0'])
        else:
            phase_fac = -1.0

        # extracting field data
        for field_name in sampled_field_names:
            all_field_data = np.zeros((len(time_values),ntot),dtype='complex128')
            
            for time in time_values:
                single_field_array = np.zeros(ntot,dtype='complex128')

                self.field.set_time(time)
                if field_name == 'field_phi':
                    field_obj = self.field.phi()
                elif field_name == 'field_apar':
                    field_obj = self.field.apar()
                elif field_name == 'field_bpar':
                    field_obj = self.field.bpar()

                # Determine the loop range and indexing factor based on the value of pars['shat']
                loop_end = int(self.field.nx/2)
                index_sign = 1 if self.param_dict['shat'] >= 0.0 else -1
                phase_sign = 1 if self.param_dict['shat'] >= 0.0 else -1

                for i in range(loop_end + (1 if self.param_dict['shat'] < 0.0 else 0)):
                    # Abstracted index calculations
                    main_start_idx = (i + loop_end) * self.field.nz
                    main_end_idx = (i + loop_end + 1) * self.field.nz

                    mirror_start_idx = (loop_end - i - 1) * self.field.nz
                    mirror_end_idx = (loop_end - i) * self.field.nz

                    field_main_idx = index_sign * i
                    field_mirror_idx = phase_sign * (i + 1)

                    # Assign values for phi
                    single_field_array[main_start_idx:main_end_idx] = field_obj[:, 0, field_main_idx] * phase_fac**i
                    if i < loop_end:
                        single_field_array[mirror_start_idx:mirror_end_idx] = field_obj[:, 0, field_mirror_idx] * phase_fac**(-(i+1))
                    

                all_field_data[time_values.index(time),:] = single_field_array # append to field data array

            all_field_data = all_field_data/field_obj[loop_end,0,0] # normalize the field
            field_dict[field_name] = all_field_data

        return field_dict


    def _time_sampler(self, time_criteria:(str|list), time_sampler_method:str):
        all_time_values = self.field.tfld

        if isinstance(time_criteria, str):
            if time_criteria not in ['last', 'first', 'all']:
                raise ValueError(f'Choose time_criteria from: {["last", "first", "all"]}')

            # Extracting time values based on chosen time criteria
            if time_criteria == 'last':
                time_values = [all_time_values[-1]]
            elif time_criteria == 'first':
                time_values = [all_time_values[0]]
            elif time_criteria == 'all':
                time_values = all_time_values
            
        elif isinstance(time_criteria, float) or isinstance(time_criteria, int):
            time_ind = np.argmin(abs(np.array(all_time_values) - time_criteria))
            time_values = [all_time_values[time_ind]]

        elif isinstance(time_criteria, list) and (len(time_criteria) == 2):

            if time_criteria[0] > time_criteria[1]:
                raise ValueError(f'Ensure time lower bound: ({time_criteria[0]}) < upper bound: ({time_criteria[1]})')
            
            if time_sampler_method not in TIME_SAMPLER_METHODS:
                raise ValueError(f'Choose time_sampler_method from: {TIME_SAMPLER_METHODS}')

            # Extracting time values based on chosen time criteria
            if time_sampler_method == 'absolute':
                lower_time_ind = np.argmin(abs(np.array(all_time_values) - time_criteria[0]))
                upper_time_ind = np.argmin(abs(np.array(all_time_values) - time_criteria[1]))
                
            elif time_sampler_method == 'percentage':
                norm_time_values = np.array(all_time_values)/max(all_time_values)
                lower_time_ind = np.argmin(abs(np.array(norm_time_values) - time_criteria[0]))
                upper_time_ind = np.argmin(abs(np.array(norm_time_values) - time_criteria[1]))
            
            time_values = all_time_values[lower_time_ind:upper_time_ind + 1]
    
        else:
            raise ValueError(f'Ensure time_critera is given as "last", "first", "all", a float/int, or list of [lower_bound_time, upper_bound_time]')

        return time_values



    def _sample_name_ind(self, full_name_list: list, input_names: str) -> list:
        """Return sampled quantities from a full list.
        If input_names is 'all', return all quantities & indices.
        Raises ValueError if any name in input_names is not found in full_name_ind_list.
        """
        if input_names == 'all':
            input_name_list = full_name_list
        else:
            # Split the input_names into a list if it's a string of names with space, else keep as list
            input_name_list = input_names.split() if isinstance(input_names, str) else input_names

        # Convert input_name_list to a set to remove duplicates and then to list
        input_name_list = list(set(input_name_list))

        if not isinstance(input_name_list, list):
            raise ValueError(f'Input names ({input_name_list}) must be a string or list of strings.')

        filtered_list = []
        # Filter full_name_ind_list to include only items whose names are in input_name_list
        for name in input_name_list:
            n_fields = self.param_dict['n_fields']

            if not name in full_name_list:
                raise ValueError(f"Quantity not found: '{name}'.\n Please chose from the following:\n {[', '.join(full_name_list)]}")
            
            if name == 'field_phi':
                filtered_list.append(name)
            elif name == 'field_apar' and n_fields > 1:
                filtered_list.append(name)
            elif name == 'field_bpar' and n_fields > 2:
                filtered_list.append(name)

        return filtered_list


