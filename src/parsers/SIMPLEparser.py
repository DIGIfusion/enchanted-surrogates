import os
import ast
from .base import Parser
import json


class SIMPLEparser(Parser):
    """
    An I/O parser for testing.

    Methods:
        __init__()
            Initializes the SIMPLEparser object.
        write_input_file(params: dict, run_dir: str) -> None
            Writes a sample input file.
        read_output_file(params: dict, run_dir: str) -> dict
            Reads the output file containing the input parameters.

    """

    def __init__(self):
        """
        Initializes the SIMPLEparser object.

        """
        pass

    def write_input_file(self, params: dict, run_dir: str):
        """
        Writes a sample input file.

        Args:
            params (dict): Dictionary containing input parameters.
            run_dir (str): Directory where the input file is written.

        Returns:
            None

        """
        file_name = os.path.join(run_dir, "in.json")
        with open(file_name, "w") as file:
            json.dump(params, file)
    
    def read_input_file(self, run_dir: str):
        """
        Reads the input file containing the input parameters.
        Args:
            run_dir (str): Directory where the output file is located.
        Returns:
            dict: Dictionary containing the output parameters read from the file.
        Raises:
            FileNotFoundError: If the output file does not exist.
        """
        
        file_name = os.path.join(run_dir, "in.json")
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"{file_name}")
        with open(file_name, "r") as file:
            params_in = json.load(file)
        return params_in

    def read_output_file(self, run_dir: str):
        """
        Reads the output file containing the input parameters.
        Args:
            run_dir (str): Directory where the output file is located.
        Returns:
            dict: Dictionary containing the output parameters read from the file.
        Raises:
            FileNotFoundError: If the output file does not exist.
        """
        file_name = os.path.join(run_dir, "out.json")
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"{file_name}")

        with open(file_name, "r") as file:
            params_out = json.load(file)
        return params_out
