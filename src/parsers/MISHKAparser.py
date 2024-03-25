from .base import Parser
import subprocess 
import os
import f90nml
import numpy as np

class MISHKAparser(Parser):
    """ An I/O parser for MISHKA """ 
    def __init__(self): 
        self.default_namelist = "/scratch/project_2009007/enchanted-surrogates/tests/mishka/fort.10"
        pass 

    def write_input_file(self, params: dict, run_dir: str):
        print(run_dir, params)
        print('Writing to', run_dir)
        if os.path.exists(run_dir): 
            input_fpath = os.path.join(run_dir, 'fort.10')
        else: 
            raise FileNotFoundError(f'Couldnt find {run_dir}')
        
        # TODO: params should be dict, not list
        namelist = f90nml.read(self.default_namelist)
        namelist['newrun'][0]['ntor'] = params[0]

        f90nml.write(namelist, input_fpath)
        print(input_fpath)
    
    def read_output_file(self, run_dir: str): 
        pass


        