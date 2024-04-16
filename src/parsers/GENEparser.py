from base import Parser # add . for relative import
import subprocess
import os
import numpy as np
import f90nml
from typing import List
from copy import deepcopy

class GENEparser(Parser):
    """An I/O parser for GENE

     Attributes
    ----------

    Methods
    -------
    write_input_file
        Writes the inputfile for a single set of parameters.

    read_output_file
        Reads the output file to python format

    """
    def __init__(self, base_params_dir):
        # in other files the parameters are a list of strings.
        # their format should be box-n_spec for unique &name entries
        # for non unique ones like species it should be
        # species-e-omn for &species
                            #name = 'e'
                            #omn = 
        # where e is the uniquely identifying attribute

        # then they can be parsed into a dictionary and added to the nml

        self.base_nml = f90nml.read(base_params_dir)
        print(self.base_nml.todict)
        # print(type(self.base_params))

        patch_nml = f90nml.namelist.Namelist({'box':{'n_spec':'ii'}})
        self.base_nml.patch(patch_nml)
        print(self.base_nml)
    
    def write_input_file(self, params: dict, run_dir: str):
        self.base_nml.patch()
        
        # give some parameters write to a new input file!
        # TODO: write a standard input file based on somthing?
        print("parser params", params)
        print("Writing to", run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, 'input.gene')
            subprocess.run(['touch', f'{input_fpath}'])
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')

        


        with open(input_fpath, 'w') as file:
            for param_name, val in params.items():
                default_params.pop(param_name)
                file.write(f'{param_name}={val}\n')
            for default_param_name, default_param_dict in default_params.items():
                default_val = default_param_dict['default']
                file.write(f'{default_param_name}={default_val}\n')
        # TODO: check for input comparisons with available inputs for code?

    def read_output_file(self, run_dir: str):
        ky_spectrum_file_path = os.path.join(run_dir, self.ky_spectrum_file)
        growth_rate_freq_file_path = os.path.join(
            run_dir, self.growth_rate_freq_file)
        flux_spectrum_file_path = os.path.join(
            run_dir, self.flux_spectrum_file)

        self.ky_spectrum = np.genfromtxt(
            ky_spectrum_file_path, dtype=None, skip_header=2)
        self.eigenvalue_spectrum = np.genfromtxt(
            growth_rate_freq_file_path, dtype=None, skip_header=2)
        self.flux_spectrums = self.parse_flux_spectrum(flux_spectrum_file_path)
        self.fluxes = []
        for species in self.flux_spectrums:
            energy_flux = species[:, 1].sum()
            particle_flux = species[:, 0].sum()
            self.fluxes.extend([energy_flux, particle_flux])
        # self.fluxes = [flux_spec.sum() for flux_spec in self.flux_spectrums]

    def parse_flux_spectrum(self, file_path) -> List[np.ndarray]:
        data_sets = []
        current_data_set = []
        with open(file_path, 'r') as file:
            for line in file:
                # Check if the line is a species marker indicating
                # a new data set
                if line.startswith(' species ='):
                    # If we already have data collected, convert it to a NumPy
                    # array and reset for the next set
                    if current_data_set:
                        data_sets.append(
                            np.array(current_data_set, dtype=float))
                        current_data_set = []
                    # Skip the next line which contains column names
                    next(file)
                else:
                    # Collect data lines into the current set
                    data_values = line.split()
                    if data_values:  # Ensure it's not an empty line
                        current_data_set.append(
                            [float(value) for value in data_values])
            # Don't forget to add the last set if the file ends without
            # a new marker
            if current_data_set:
                data_sets.append(np.array(current_data_set, dtype=float))
        return data_sets

    def input_dict(self, ) -> dict: 
        parameters_dict = self.default_ga_input()

        # BASED ON ASTRA INTERFACE FILE, ALL SET BEFORE THE RADIAL LOOP 
        parameters_dict["KYGRID_MODEL"]['default'] = "4"
        parameters_dict["SAT_RULE"]['default'] = "2"
        parameters_dict["XNU_MODEL"]['default'] = "3" # SAT_RULE 2
        parameters_dict["ALPHA_ZF"] = {'interface_parameter': 'tglf_alpha_zf_in', 'default': "1"} # SAT_RULE 2
        
        parameters_dict['USE_BPER']['default'] = ".True."
        parameters_dict['USE_MHD_RULE']['default'] = ".Frue."

        parameters_dict['NMODES']['default'] = str(int(parameters_dict['NS']['default']) + 2)
        parameters_dict['NBASIS_MAX']['default'] = 6
        parameters_dict['NKY']['default'] = 19
        parameters_dict['UNITS']['default'] = 'CGYRO'

        parameters_dict['VPAR_SHEAR_MODEL']['default'] = 1
        

   
        return parameters_dict
    
if __name__ == '__main__':
    pa = GENEparser(base_params_dir='/home/djdaniel/Downloads/parameters_miller')