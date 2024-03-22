from .base import Parser
import subprocess 
import os 

class TGLFparser(Parser):
    """ An I/O parser for TGLF """ 
    def __init__(self): 
        pass 

    def write_input_file(self, params: dict, run_dir: str):
        # give some parameters write to a new input file! 
        if os.path.exists(run_dir): 
            input_fpath = os.path.join(run_dir, 'input.tglf')
            subprocess.run(['touch', f'{input_fpath}'])
        else: 
            raise FileNotFoundError(f'Couldnt find {run_dir}')
        print('Writing to', run_dir)
        