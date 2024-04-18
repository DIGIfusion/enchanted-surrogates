from .base import Parser # add . for relative import
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
        """
        Generates the base f90nml namelist from the GENE parameters file at base_params_dir.

        Parameters
        ----------
            base_params_dir (string or path): The directory pointing to the base GENE parameters file.
            The base GENE parameters file must contain all parameters necessary for GENE to run.
            Any parameters to be sampled will be inserted into the base parameter file before each run.
            Any value of a sampled parameter in the base file will be ignored. 
        Returns
        -------
            Nothing 
        """
        self.base_namelist = f90nml.read(base_params_dir) #odict_keys(['parallelization', 'box', 'in_out', 'general', 'geometry', '_grp_species_0', '_grp_species_1', 'units'])
        
        
    def write_input_file(self, params: dict, run_dir):
        """
        Write the GENE input file to the run directory specified. 
        
        Parameters
        ----------
            params (dict): The keys store strings of the names of the parameters as specified in the enchanted surrogates *_config.yaml configuration file.
            The values stores floats of the parameter values to be ran in GENE.

            run_dir (string or path): A directory in the machine intended for the run of GENE.  

        """
        params_keys = list(params.keys())
        params_values = list(params.values())
        patch = {}

        for key, value in zip(params_keys,params_values):
            group_name, variable_name = key.split('-')
            if list(patch.keys()).count(group_name) > 0:
                patch[group_name][variable_name] = value
            else: patch[group_name] = {variable_name:value}

        namelist = self.base_namelist
        patch = f90nml.namelist.Namelist(patch)
        namelist.patch(patch)
        
        f90nml.write(namelist, os.path.join(run_dir,'parameters'))

    # def read_output_file(self, run_dir: str):
    #     ky_spectrum_file_path = os.path.join(run_dir, self.ky_spectrum_file)
    #     growth_rate_freq_file_path = os.path.join(
    #         run_dir, self.growth_rate_freq_file)
    #     flux_spectrum_file_path = os.path.join(
    #         run_dir, self.flux_spectrum_file)

    #     self.ky_spectrum = np.genfromtxt(
    #         ky_spectrum_file_path, dtype=None, skip_header=2)
    #     self.eigenvalue_spectrum = np.genfromtxt(
    #         growth_rate_freq_file_path, dtype=None, skip_header=2)
    #     self.flux_spectrums = self.parse_flux_spectrum(flux_spectrum_file_path)
    #     self.fluxes = []
    #     for species in self.flux_spectrums:
    #         energy_flux = species[:, 1].sum()
    #         particle_flux = species[:, 0].sum()
    #         self.fluxes.extend([energy_flux, particle_flux])
    #     # self.fluxes = [flux_spec.sum() for flux_spec in self.flux_spectrums]

    
if __name__ == '__main__':
    bounds = [[0.1, 300],[2,3.5],[4,6.8]]
    params = {'box-kymin':100.1, '_grp_species_0-omt': 2.75, '_grp_species_1-omt':5.1, '_grp_species_1-dens':9000}
    parser = GENEparser(base_params_dir='/home/djdaniel/Downloads/parameters')
    parser.write_input_file(params,run_dir=os.getcwd())