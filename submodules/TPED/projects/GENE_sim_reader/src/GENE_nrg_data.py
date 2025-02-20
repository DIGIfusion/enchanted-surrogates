import os
import numpy as np
import xarray as xr

from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC
from TPED.projects.GENE_sim_reader.src.GENE_parameters_data import GeneParameters as GP

# Index mapping for energy quantities
NRG_QUANT_IND = [
    ('n_mag', 0), ('u_par_mag', 1), 
    ('T_par_mag', 2), ('T_perp_mag', 3), 
    ('Gamma_ES', 4), ('Gamma_EM', 5), 
    ('Q_ES', 6), ('Q_EM', 7), 
    ('Pi_ES', 8), ('Pi_EM', 9)
]

TIME_SAMPLER_METHODS = ['absolute', 'percentage']


class GeneNrg:
    def __init__(self, nrg_filepath: str):
        """
        Initializes the GeneNrg class with the given NRG file path.

        Args:
            nrg_filepath (str): Path to the NRG file.
        """
        self.nrg_filepath = nrg_filepath
        param_filepath = GFC(self.nrg_filepath).switch_suffix_file('parameters')
        self.spec_list = GP(param_filepath).spec_name_num_list()
        self.n_spec = len(self.spec_list)


    def nrg_filepath_to_xarray(self, time_criteria: str = 'last', input_nrg_species: str = 'all', input_nrg_quantities: str = 'all') -> xr.Dataset:
        nrg_dict = self.nrg_filepath_to_dict(time_criteria, input_nrg_species, input_nrg_quantities)
        return self.nrg_dict_to_xarray(nrg_dict)

    def nrg_dict_to_xarray(self, nrg_dict: dict):
        """
        Convert nrg dictionary to xarray dataset.
        
        Args:
        - nrg_dict (dict): A dictionary containing field data and relevant information.
        
        Returns:
        - xarray.Dataset: An xarray dataset containing nrg data.
        """
        # Extract 'filepath' from the dictionary
        filepath = nrg_dict.pop('filepath', None)
        directory = os.path.dirname(filepath)
        suffix = GFC(filepath).suffix_from_filepath()


        # Creating xarray dataset from field data
        nrg_dataset = xr.Dataset(
            data_vars={
                'time': np.array(nrg_dict['time']),
            }
        )

        for key in nrg_dict:
            if (key != 'time') and (key != 'filepath') and (len(nrg_dict[key]) > 0):
                nrg_dataset[key] = (['time'], np.array(nrg_dict[key]))

        # Add filepath as an attribute
        nrg_dataset.attrs['filepath'] = filepath
        nrg_dataset.attrs['directory'] = directory
        nrg_dataset.attrs['suffix'] = suffix

        return nrg_dataset

    def nrg_filepath_to_dict(
            self, time_criteria: str = 'last', 
            input_nrg_species: (str|list) = 'all', 
            input_nrg_quantities: (str|list) = 'all',
            time_sampler_method: str = 'absolute') -> dict:
        """
        Converts NRG file data to a dictionary based on the specified criteria.

        Args:
            time_criteria (str): Criteria for selecting time steps. Defaults to 'last'.
            input_nrg_species (str): Specifies which species to include. Defaults to 'all'.
            input_nrg_quantities (str): Specifies which quantities to include. Defaults to 'all'.

        Returns:
            dict: Dictionary with NRG data.
        """
        nrg_dict = self._initialize_nrg_dict(self.spec_list)

        sampled_nrg_spec = self._sample_name_ind(self.spec_list, input_nrg_species)
        sampled_nrg_quant = self._sample_name_ind(NRG_QUANT_IND, input_nrg_quantities)

        time_values = self._time_sampler(time_criteria, time_sampler_method)

        if time_criteria == 'last':
            pass
        elif time_criteria == 'all':
            pass
        else:
            raise ValueError('Invalid time criteria. Please choose either "last" or "all".')
        


        with open(self.nrg_filepath, 'r') as nrg_file:
            data = nrg_file.readlines()

            if time_criteria == 'last':
                # Select the last time step index
                time_ind = -1 - self.n_spec  # Get the last index minus the number of species
                nrg_dict = self._process_nrg_data_at_time(data, nrg_dict, time_ind, sampled_nrg_spec, sampled_nrg_quant)
                return self._convert_lists_to_arrays(nrg_dict)

            # Process all time steps
            for time_ind in range(0, len(data), self.n_spec + 1):
                nrg_dict = self._process_nrg_data_at_time(data, nrg_dict, time_ind, sampled_nrg_spec, sampled_nrg_quant)
            
        return self._convert_lists_to_arrays(nrg_dict)

    ############################################################################################
    ############################## Helper functions  ###########################################
    ############################################################################################

    def _initialize_nrg_dict(self, spec_list: list) -> dict:
        """
        Initializes the NRG dictionary structure.

        Args:
            spec_list (list): List of species names and indices.

        Returns:
            dict: Initialized NRG dictionary.
        """
        nrg_dict = {'filepath': self.nrg_filepath, 'time': []}
        for spec_name, _ in spec_list:
            for nrg_quant, _ in NRG_QUANT_IND:
                nrg_dict[nrg_quant + "_" + spec_name] = []
        return nrg_dict




    def _sample_name_ind(self, full_name_ind_list: list, input_names: str) -> list:
        """Return sampled quantities & indices from a full list.
        If input_names is 'all', return all quantities & indices.
        Raises ValueError if any name in input_names is not found in full_name_ind_list.
        """
        if input_names == 'all':
            return full_name_ind_list

        # Split the input_names into a list if it's a string of names
        input_name_list = input_names.split() if isinstance(input_names, str) else input_names

        # Filter full_name_ind_list to include only items whose names are in input_name_list
        filtered_list = [(name, ind) for name, ind in full_name_ind_list if name in input_name_list]

        # Check if all input_names are found in full_name_ind_list
        if len(filtered_list) != len(input_name_list):
            missing_names = set(input_name_list) - {name for name, ind in filtered_list}
            available_names = ', '.join(name for name, _ in full_name_ind_list)
            raise ValueError(f"Quantity not found: '{', '.join(missing_names)}'.\n Please chose from the following:\n {[available_names]}")

        return filtered_list

    def _process_nrg_data_at_time(self, data: list, nrg_dict: dict, time_ind: int, sampled_nrg_spec: list, sampled_nrg_quant: list) -> dict:
        """
        Processes and appends NRG data at a specific time step in the NRG data.

        Args:
            data (list): List of NRG data lines.
            nrg_dict (dict): Dictionary to store processed NRG data.
            time_ind (int): Index of the time step to process.
            sampled_nrg_spec (list): List of sampled species.
            sampled_nrg_quant (list): List of sampled quantities.

        Returns:
            dict: Updated NRG dictionary.
        """
        line = data[time_ind].strip().split()
        time = float(line[0])
        nrg_dict['time'].append(time)

        for spec_name, spec_num in sampled_nrg_spec:
            time_plus_spec_ind = time_ind + spec_num
            for nrg_quant, nrg_ind in sampled_nrg_quant:
                nrg_value = float(data[time_plus_spec_ind].strip().split()[nrg_ind])
                nrg_dict[nrg_quant + "_" + spec_name].append(nrg_value)

        return nrg_dict

    def _convert_lists_to_arrays(self, nrg_dict: dict) -> dict:
        """
        Converts lists in the NRG dictionary to numpy arrays.

        Args:
            nrg_dict (dict): Dictionary with NRG data.

        Returns:
            dict: Dictionary with lists converted to numpy arrays.
        """
        for key in nrg_dict:
            if isinstance(nrg_dict[key], list) and key != 'filepath':
                nrg_dict[key] = np.array(nrg_dict[key], dtype=float)
        return nrg_dict
