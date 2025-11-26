import os
import numpy as np
import pandas as pd

from enchanted_surrogates.parsers.base_parser import Parser


class ExampleBayesianOptimizationParser(Parser):
    """
    An example I/O parser for testing the Bayesian Optimization sampler.

    Methods
    -------

        read_output_file(params: dict, run_dir: str) -> dict
            Reads the output file and returs a dictionary with the 
            input settings and the output file.

        write_input_file():
            This is a dummy function that is not actually needed.

        collect_sample_information(run_dir: str) -> dict
            Collects the scoring information needed for Bayesian 
            optimization. 

    """

    def __init__(self):
        """
        Initializes the ExampleBayesianOptimizationParser object.

        """
        pass

    def write_input_file(self):
        """ 
        This is not actually needed.
        """
        pass
          
    def read_output_file(self, run_dir: str):
        """
        Reads the output files from the run directory

        Args:
            run_dir (str): Directory where the output file is located.

        Returns:
            dict: Dictionary containing the settings and the output 
                  dictionaries.

        """
        datavec = pd.read_csv(os.path.join(run_dir, 
                                           "enchanted_datapoint.csv"))
        inputs = float(datavec['x'].iloc[0])
        output = float(datavec['output'].iloc[0])
        outputdict = {'input':inputs, 'output':output}
        return outputdict

    def collect_sample_information(self, run_dir: str, observations: dict):
        """
        Reads the information from the run directory that is needed for the 
        Bayesian optimization sampler.

        Args:
            run_dir (str): Directory where the run has been conducted.
            observations (dict): Dictionary of observations 
                                 (not actually used by this example)

        Returns:
            dict: Dictionary containing the information needed for the BO 
                  routines.
        """
        # Read the sample input vector as dumped in the pkl file
        datavec = pd.read_csv(os.path.join(run_dir, 
                                           "enchanted_datapoint.csv"))
        inputs = {'x':float(datavec['x'].iloc[0])}
        # This computes a distance metric for the current.
        # Mean absolute error is chosen here.
        outputdict = self.read_output_file(run_dir)
        distances = np.array([2.0 - outputdict['output']])
        inputnd = []
        for key in inputs:
            inputnd.append(inputs[key])
        inputnd = np.array(inputnd)
        outputdict = {'run_dir':run_dir, 'inputs':inputnd, 
                      'distances':distances, 'failure':0}
        return outputdict
