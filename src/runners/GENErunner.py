# import numpy as np
from .base import Runner
from parsers.GENEparser import GENE_scan_parser, GENE_single_parser
import subprocess
import os


class GENErunner(Runner):
    def __init__(self, executable_path, is_gene_scan: bool, *args, **kwargs):
        if is_gene_scan:
            self.parser = GENE_scan_parser()
        else:
            self.parser = GENE_single_parser()
        self.executable_dir = executable_path
    def single_code_run(self, params: dict, run_dir):
        """Logic to run GENE"""
        print(params)
        
        # write input file
        self.parser.write_input_file(params, run_dir)
        # run code #need to find a way to specify that the parameters file is not in the executable directory but the run directory
        # maybe os.chdir allows the executable to be ran as if in run_dir
        os.chdir(run_dir)
        subprocess.call([f"{self.executable_path}"])

        # process output
        # self.parser.read_output_file(run_dir)

        return True

if __name__ == '__main__':
    runner = GENErunner()
    params = {'box-kymin':100.1, '_grp_species_0-omt': 2.75, '_grp_species_1-omt':5.1}
    run_dir = 'path/to/run_dir' #The run directory is where the worker writes the parameters files and the results.
    runner.single_code_run(params, run_dir)