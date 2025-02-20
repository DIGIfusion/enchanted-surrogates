import os
import xarray as xr

from TPED.projects.GENE_sim_reader.utils.ParIO import Parameters
from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC

####################################################################################################
########################## Convert parameters filepath to dict #####################################
####################################################################################################

class GeneParameters:
    def __init__(self, parameters_filepath:str):
        self.parameters_filepath = parameters_filepath
        self.parameters_dict = self.parameters_filepath_to_dict()
    

    def parameters_filepath_to_xarray(self):
        parameter_dict = self.parameters_filepath_to_dict()
        return self.parameters_dict_to_xarray(parameter_dict)
    
    def parameters_dict_to_xarray(self, parameter_dict: dict):

        # Extract 'filepath' from the dictionary
        filepath = parameter_dict.pop('filepath', None)
        directory = os.path.dirname(filepath)
        suffix = GFC(filepath).suffix_from_filepath()

        # Convert OrderedDict to xarray dataset
        param_dataset = xr.Dataset(parameter_dict)

        # Add filepath as an attribute
        param_dataset.attrs['filepath'] = filepath
        param_dataset.attrs['directory'] = directory
        param_dataset.attrs['suffix'] = suffix

        return param_dataset

    def parameters_filepath_to_dict(self):
        # Create a parameter dictionary using the Parameters class
        par = Parameters()
        par.Read_Pars(self.parameters_filepath)  # Read the parameter file
        parameter_dict = par.pardict 

        # Add the filename, filepath, and suffix to the parameter dictionary
        parameter_dict['filepath'] = self.parameters_filepath
        
        for key, value in parameter_dict.items():
            if isinstance(value, str):
                strip_value = value.strip("'")
                strip_value = strip_value.strip('"')
                parameter_dict[key] = strip_value
            
        return parameter_dict


    def spec_name_num_list(self):
        n_spec = self.parameters_dict['n_spec']
        spec_list = [(self.parameters_dict[f'name{spec_num}'].strip("'"), spec_num) for spec_num in range(1, n_spec + 1)]
        return spec_list





