"""
# executors/DaskExecutor.py
Contains logic for executing surrogate workflow on Dask.
"""

import time
import os
import parsers
import executors
import numpy as np
import re

def extract_integer(s):
    """Extract the first integer found in the string."""
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None

class DaskExecutorSimulationPipeline():
    """
    Handles the consecutive running of DaskExecutorSimulations. 
    This allows different simulations to be ran in series,
    one output being the input to the next, 
    each can be ran on a different cluster with different resources per worker.
    Args:
    Returns:
    Raises:
    """
    def __init__(self, **kwargs):
        # Making a list of executors in the order of their intergers specified in their key, of config file
        executor_keys = [key for key in kwargs.keys() if 'executor' in key]
        executor_order_index = np.argsort([extract_integer(key) for key in pipeline_parser_keys])
        executor_keys = [executor_keys[index] for index in executor_order_index]
        executor_types = [kwargs[key]['type'] for key in executor_keys]
        
        self.executors = [getattr(executors, executor_type)() for executor_type in executor_types]
        self.num_sub_executors = len(executors)
        
        # Making a list of parser functions in the order of their intergers specified in their key, of config file        
        pipeline_parser_keys = [key for key in kwargs.keys() if 'pipeline_parser' in key]
        pipeline_parser_order_index = np.argsort([extract_integer(key) for key in pipeline_parser_keys])
        pipeline_parser_keys = [pipeline_parser_keys[index] for index in pipeline_parser_order_index]
        pipeline_parser_types = [kwargs[key]['type'] for key in pipeline_parser_keys]
        pipeline_parser_function_strings = [kwargs[key]['function'] for key in pipeline_parser_keys]
        pipeline_parsers = [getattr(parsers, parser_type)() for parser_type in pipeline_parser_types]
        
        self.pipeline_parser_functions = [getattr(pipeline_parser, function_string) for pipeline_parser,function_string in zip(pipeline_parsers, pipeline_parser_function_strings)]
        
        # self.runner = getattr(parsers, runner_args["type"])(**runner_args)
    
    
    
    def start_pipeline(self):
        print(100 * "=")
        print('STARTING PIPELINE')
        for index, executor in enumerate(self.executors):
            executor.start_runs()
            
            