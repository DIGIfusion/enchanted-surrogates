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
        
class GENE_scan_parser(GENEparser):

class GENE_singles_parser(GENEparser)

if __name__ == '__main__':
    bounds = [[0.1, 300],[2,3.5],[4,6.8]]
    params = {'box-kymin':100.1, '_grp_species_0-omt': 2.75, '_grp_species_1-omt':5.1}
    parser = GENEparser(base_params_dir='/home/djdaniel/Downloads/parameters')
    parser.write_input_file(params,run_dir=os.getcwd())