"""
# executors/DaskExecutor.py
Contains logic for executing surrogate workflow on Dask.
"""

import time
import os
import parsers
import uuid
import executors
import numpy as np
import re
from tasks import run_simulation_task

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
    def __init__(self, sampler, **kwargs):
        self.sampler = sampler
        
        # Making a list of executors in the order of their intergers specified in their key, of config file
        executor_keys = [key for key in kwargs.keys() if 'executor' in key]
        executor_order_index = np.argsort([extract_integer(key) for key in pipeline_parser_keys])
        executor_keys = [executor_keys[index] for index in executor_order_index]
        executor_types = [kwargs[key]['type'] for key in executor_keys]
        executor_args_s = [kwargs[key] for key in executor_keys]
        self.executors = [getattr(executors, executor_type)(executor_args) for executor_type, executor_args in zip(executor_types, executor_args_s)]
        self.num_sub_executors = len(executors)
        
        # Making a list of parser functions in the order of their intergers specified in their key, of config file        
        pipeline_parser_keys = [key for key in kwargs.keys() if 'pipeline_parser' in key]
        pipeline_parser_order_index = np.argsort([extract_integer(key) for key in pipeline_parser_keys])
        pipeline_parser_keys = [pipeline_parser_keys[index] for index in pipeline_parser_order_index]
        # pipeline_parser_types = [kwargs[key]['type'] for key in pipeline_parser_keys]
        pipeline_parser_function_strings = [kwargs[key]['function'] for key in pipeline_parser_keys]
        self.pipeline_parser_functions = [getattr(parsers, function_string) for function_string in pipeline_parser_function_strings]
        
        # self.pipeline_parser_functions = [getattr(pipeline_parser, function_string) for pipeline_parser,function_string in zip(pipeline_parsers, pipeline_parser_function_strings)]
        
        
        self.base_run_dir = kwargs.get("base_run_dir")
        if self.base_run_dir==None:
            raise ValueError('''Enchanted surrogates handeles the creation of run directories. 
                             You must supply a base_run_dir in your configs file. Example:
                             executor:
                                type: DaskExecutorSimulation
                                base_run_dir: /project/project_462000451/test-enchanted/trial-dask
                             ...
                             ...
                             ''')
        self.base_out_dir = kwargs.get("base_out_dir")
        # self.runner = getattr(parsers, runner_args["type"])(**runner_args)
    
    def start_runs(self):
        print(100 * "=")
        print('STARTING PIPELINE')
        print('MAKING INITIAL SAMPLES')
        samples = self.sampler.get_initial_parameters()
        num_samples = len(samples)
        print('MAKING RANDOM RUN IDs')
        run_ids = [str(uuid.uuid4()) for sample in samples]
        run_dict = {run_id:sample for run_id, sample in zip(run_ids, samples)}
        
        # Make the base run dir
        # Make the base out dir
        if not os.path.exists(self.base_run_dir):
            os.system(f'mkdir {self.base_run_dir} -p')
        if not os.path.exists(self.base_out_dir):
            os.system(f'mkdir {self.base_out_dir} -p')
            
        print('MAKING ALL NECESSARY DIRECTORIES')
        base_run_dir_simulation_s = []
        for index, executor in enumerate(self.executors):
            base_run_dir_simulation = os.path.join(self.base_run_dir,executor.runner.__class__.__name__+f'_{index}')
            run_dirs = [os.path.join(base_run_dir_simulation, run_id) for run_id in run_ids]
            base_run_dir_simulation_s.append(run_dirs)
        
        # if base_out_dir is in configs file for the pipeline executor then we will make out_dirs other wise we set it to be the same as the run_dir
        if self.base_out_dir!=None:
            base_out_dir_simulation_s = []
            for index, executor in enumerate(self.executors):
                base_out_dir_simulation = os.path.join(self.base_out_dir,executor.runner.__class__.__name__+f'_{index}')
                out_dirs = [os.path.join(base_out_dir_simulation, run_id) for run_id in run_ids]
                base_out_dir_simulation_s.append(out_dirs)
        else:
            base_out_dir_simulation_s = base_run_dir_simulation_s          
        

        print('ENTERING FOR LOOP OVER EACH SIMULATION EXECUTOR')
        previous_parse_futures = [None]*num_samples
        executing_last_simulation = False
        executing_first_simulation = True
        
        for index, executor in enumerate(self.executors):
            if executor.base_run_dir != None or executor.base_out_dir != None:
                raise Warning(f'''The run and out directories are being handeled by the Pipeline Executor.
                              This means that the Simulation Executor run and out directories are being ignored:
                              run: {executor.base_run_dir} out:{executor.base_out_dir}
                              The ExecutorSimulationPileline base run and out directories are being used:
                              run: {self.base_run_dir}, out:{self.base_out_dir}''') 
            
            if index == len(self.executors):
                executing_last_simulation = True
            if index > 0:
                executing_first_simulation = False
            parse_futures = []
            run_dirs = base_run_dir_simulation_s[index]
            out_dirs = base_out_dir_simulation_s[index]
            if not executing_last_simulation:
                next_run_dirs = base_run_dir_simulation_s[index+1]
            
            for run_index, run_dir, out_dir, next_run_dir in zip(range(num_samples),run_dirs, out_dirs, next_run_dirs):
                # if this is the first executor then we need to take samples from the sampler
                # If after first executor then run_dir should be already be set up
                if executing_first_simulation:
                    sample=samples[run_index]
                else:
                    sample=None
                # This send the run_simulation_task to be ran on a worker as soon as possible
                # Dask will wait untill all dependent futures have finished and returned a value
                new_future = executor.client.submit(run_simulation_task, run_dir, out_dir, sample, future=previous_parse_futures[run_index])
                #Make future for parsing
                if not executing_last_simulation:
                    print('PERFORMING PIPELINE PARSE FROM', out_dir, 'TO', next_run_dir)
                    new_parse_future = executor.client.submit(self.pipeline_parser_functions[0], last_out_dir=out_dir, next_run_dir=next_run_dir, future=new_future)
                    parse_futures.append(new_parse_future)
            if not executing_last_simulation:
                previous_parse_futures = parse_futures
                        
            