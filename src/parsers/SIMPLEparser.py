import os
from .base import Parser
import ast 

class SIMPLEparser(Parser):
    """An I/O parser for testing"""

    def __init__(self):
        pass

    def write_input_file(self, params: dict, run_dir: str):
        """
        Writes a sample input file.
        """
        print("Writing to", run_dir)
        file_name = run_dir + "./input.txt"
        with open(file_name, "w") as file:
            file.write("Simple start.")

    def read_output_file(self, params: dict, run_dir: str):
        """
        The output file should contain the input parameters.
        """
        file_name = run_dir + "/output.txt"
        params_out = None
        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                lines = file.readlines()
                params_out = ast.literal_eval(lines[0])
                # numbers_as_strings = lines[0].strip()[1:-1].split(",")
                # params_out = [float(num.strip()) for num in numbers_as_strings if num.strip() != '']
                assert isinstance(params_out, dict)
            return params == params_out
        else:
            raise FileNotFoundError(f'{file_name}')
            return False
