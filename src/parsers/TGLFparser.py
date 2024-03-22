from .base import Parser
import subprocess 
import os 
import numpy as np 
from typing import List 

class TGLFparser(Parser):
    """ An I/O parser for TGLF """ 
    def __init__(self): 
        self.ky_spectrum_file = 'out.tglf.ky_spectrum'
        self.growth_rate_freq_file = 'out.tglf.eigenvalue_spectrum'
        self.flux_spectrum_file = 'out.tglf.sum_flux_spectrum'

    def write_input_file(self, params: dict, run_dir: str):
        # give some parameters write to a new input file! 
        print('Writing to', run_dir)
        if os.path.exists(run_dir): 
            input_fpath = os.path.join(run_dir, 'input.tglf')
            subprocess.run(['touch', f'{input_fpath}'])
        else: 
            raise FileNotFoundError(f'Couldnt find {run_dir}')

    def read_output_file(self, run_dir: str):         
        ky_spectrum_file_path = os.path.join(run_dir, self.ky_spectrum_file)
        growth_rate_freq_file_path = os.path.join(run_dir, self.growth_rate_freq_file)
        flux_spectrum_file_path = os.path.join(run_dir, self.flux_spectrum_file)
        
        self.ky_spectrum = np.genfromtxt(ky_spectrum_file_path, dtype=None, skip_header=2)
        self.eigenvalue_spectrum = np.genfromtxt(growth_rate_freq_file_path, dtype=None, skip_header=2)
        self.flux_spectrums = self.parse_flux_spectrum(flux_spectrum_file_path)
        self.fluxes = [flux_spec.sum() for flux_spec in self.flux_spectrums]

    def parse_flux_spectrum(self, file_path) -> List[np.ndarray]:
        data_sets = []
        current_data_set = []
        with open(file_path, 'r') as file:
            for line in file:
                # Check if the line is a species marker indicating a new data set
                if line.startswith(' species ='):
                    # If we already have data collected, convert it to a NumPy array and reset for the next set
                    if current_data_set:
                        data_sets.append(np.array(current_data_set, dtype=float))
                        current_data_set = []
                    # Skip the next line which contains column names
                    next(file)
                else:
                    # Collect data lines into the current set
                    data_values = line.split()
                    if data_values:  # Ensure it's not an empty line
                        current_data_set.append([float(value) for value in data_values])
            # Don't forget to add the last set if the file ends without a new marker
            if current_data_set:
                data_sets.append(np.array(current_data_set, dtype=float))
        return data_sets

