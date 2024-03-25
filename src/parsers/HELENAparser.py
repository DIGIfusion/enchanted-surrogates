from .base import Parser
import subprocess 
import os
import f90nml
import numpy as np

class HELENAparser(Parser):
    """ An I/O parser for HELENA """ 
    def __init__(self): 
        self.default_namelist = "/scratch/project_2009007/enchanted-surrogates/tests/helena/fort.10"
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
        namelist['shape']['tria'] = params[0]
        # namelist['shape']['ellip'] = params[1]
        namelist['shape']['xr2'] = 1 - params[1] / 2.0
        namelist['shape']['sig2'] = params[1]
        namelist['profile']['zjz'] = self.make_init_zjz_profile(pedestal_delta=params[1], npts=namelist['profile']['npts'])

        f90nml.write(namelist, input_fpath)
        print(input_fpath)
    
    def read_output_file(self, run_dir: str): 
        pass

    def make_init_zjz_profile(self, pedestal_delta, npts):
        # Makes the initial ZJZ profile based on the pressure profile
        alpha1, alpha2 = 1.0, 1.5
        x = np.linspace(0, 1, npts)
        
        base = 0.9 * (1 - x ** alpha1) ** alpha2 + 0.1 * (1 + np.tanh((1 - pedestal_delta / 2 - x) / pedestal_delta * 2))
        
        pzjzmultip = 0.5    # TODO: as input?
        max_pres_grad_loc = 0.97    # TODO: add calculation from europed

        pedestal_current = pzjzmultip * np.exp(
            -((max_pres_grad_loc - x) / pedestal_delta * 1.5) ** 2)

        return base + pedestal_current
        