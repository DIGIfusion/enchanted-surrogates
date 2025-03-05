from .base import Parser
import subprocess
import os
import numpy as np
from typing import List
from copy import deepcopy
import numpy as np 

import re

class TGLFparser(Parser):
    def __init__(self):
        self.gb_flux_file = "out.tglf.gbflux"
        self.NS = 2 


    def read_gbflux_file(self, gbflux_fpath, NS=3) -> list[np.ndarray]:
        """ 
        Reads the out.tglf.gbflux file 
        and return pflux, eflux for each species 
        pflux: [0: electron, 1->NS: ions]
        eflux: [0: electron, 1->NS: ions]
        NS includes electrons and ion number of species
        """ 
        gbflux_reader = ff.FortranRecordReader('(32(1pe11.4,1x))')
        with open(gbflux_fpath, 'r') as file: 
            line = file.readline()
            output = gbflux_reader.read(line)

        elec_pflux = output[0]
        ion_pfluxs = output[1:NS]
        
        elec_eflux = output[NS]
        ion_efluxs = output[NS+1:NS+1 + NS]
        pflux      = np.array(output[:NS])
        eflux      = np.array(output[NS:2*NS])
        mflux      = np.array(output[2*NS:3*NS])
        expwd      = np.array(output[3*NS:4*NS])
        return pflux, eflux, mflux, expwd

    def write_input_file(self, params: dict, run_dir): 
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, 'input.tglf')
            subprocess.run(['cp', '/home/akadam/tokamak_samplers/data/helena_asdex_core_scan_idete7_20000/scan_00001_helena/TGLF/local_geometry_0.01/input.tglf', f'{input_fpath}'])
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')

        """ Modify Parameters """

        with open(input_fpath, 'r') as f: 
            content = f.readlines() 

        updated_file = []

        for line in content:
            for param_name, val in params.items(): 
                match_string = f"({param_name}" + r"_\d+)\s*=\s*(-?\d+\.\d+)"
                match_var = re.match(match_string, line)
                if match_var: 
                    key = match_var.group(1)
                    new_value = val 
                    line = f"{key} = {new_value}\n"
                    break 
            updated_file.append(line)
        with open(input_fpath, 'w') as file: 
            file.writelines(updated_file)

    def read_output_file(self, tglf_rundir):
        gbflux_fname = os.path.join(tglf_rundir, 'out.tglf.gbflux')
        if not os.path.exists(gbflux_fname):
            return []
        pflux, eflux, mflux, expwd = read_gbflux_file(gbflux_fname)

        return pflux, eflux, mflux
        # Sets something called self.parser.fluxes 