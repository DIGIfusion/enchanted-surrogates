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

            rprint('Writing to', run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, 'input.tglf')
            subprocess.run(['touch', f'{input_fpath}'])
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')un_dir (string or path): The file system directory where runs are to be stored

        """
        print('Writing to', run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, 'parameters')
            subprocess.run(['touch', f'{input_fpath}'])
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')

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
        
        f90nml.write(namelist, input_fpath)

    # what is returned here is returned to the runner for a single code run, which goes though the base executor to get to the future 
    def read_output_file(self, run_dir: str):
        raise NotImplementedError
    
if __name__ == '__main__':
    bounds = [[0.1, 300],[2,3.5],[4,6.8]]
    params = {'box-kymin':100.1, '_grp_species_0-omt': 2.75, '_grp_species_1-omt':5.1}
    parser = GENEparser(base_params_dir='/home/djdaniel/Downloads/parameters')
    parser.write_input_file(params,run_dir=os.getcwd())