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
from executors.tasks import run_simulation_task, print_error_wrapper
from dask.distributed import wait, print, as_completed


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
        dynamic_clusters: Boolean. When True the clusters will be created and deleted when needed, including all workers and schedulers
                                    When False the clusters will remain alive untill enchanted surrogates is finnished
    Returns:
    Raises:
    """
    def __init__(self, sampler, **kwargs):
        self.sampler = sampler
        self.dynamic_clusters = kwargs.get('dynamic_clusters', False)
        
        # Making a list of executors in the order of their intergers specified in their key, of config file
        executor_keys = [key for key in kwargs.keys() if 'executor' in key]
        executor_order_index = np.argsort([extract_integer(key) for key in executor_keys])
        executor_keys = [executor_keys[index] for index in executor_order_index]
        executor_types = [kwargs[key]['type'] for key in executor_keys]
        executor_args_s = [kwargs[key] for key in executor_keys]
        self.executors = [getattr(executors, executor_type)(**executor_args) for executor_type, executor_args in zip(executor_types, executor_args_s)]
        self.num_sub_executors = len(self.executors)
        
        # Making a list of parser functions in the order of their intergers specified in their key, of config file        
        pipeline_parser_keys = [key for key in kwargs.keys() if 'pipeline_parser' in key]
        pipeline_parser_order_index = np.argsort([extract_integer(key) for key in pipeline_parser_keys])
        pipeline_parser_keys = [pipeline_parser_keys[index] for index in pipeline_parser_order_index]
        # pipeline_parser_types = [kwargs[key]['type'] for key in pipeline_parser_keys]
        pipeline_parser_function_strings = [kwargs[key]['function'] for key in pipeline_parser_keys]
        self.pipeline_parser_functions = [getattr(parsers, function_string) for function_string in pipeline_parser_function_strings]
        
        # self.pipeline_parser_functions = [getattr(pipeline_parser, function_string) for pipeline_parser,function_string in zip(pipeline_parsers, pipeline_parser_function_strings)]
        
        self.last_runner_return_path=kwargs.get("last_runner_return_path")
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
    
    def initialise_clients(self):
        for executor in self.executors:
            executor.initialize_client()
        # print('SLEEPING 10 BEFORE PRINTING CLUSTER INFO')
        # time.sleep(10)
        # for executor in self.executors:
        #     print('='*100)
        #     print(executor.client)
        #     workers_info = executor.client.scheduler_info()['workers']
        #     for worker, info in workers_info.items():
        #         print(f"Worker: {worker}")
        #         print(f"  Memory: {info['memory_limit'] / 1e9:.2f} GB")
        #         print(f"  Cores: {info['nthreads']}")
        #         print(f"  Host: {info['host']}")
        #         print(f"  Local Directory: {info['local_directory']}")
        #         print()
    
    def clean(self):
        for executor in self.executors:
            executor.clean()
    
    def start_runs(self):
        print(100 * "=")
        print('STARTING PIPELINE')
        
        if not self.dynamic_clusters:
            print('MAKING DASK CLUSTERS')
            self.initialise_clients()
        
        print('MAKING INITIAL SAMPLES')
        samples = self.sampler.get_initial_parameters()
        num_samples = len(samples)
        
        print('MAKING RANDOM RUN IDs')
        run_ids = [str(uuid.uuid4()) for sample in samples]
        run_sample_dict = {run_id:sample for run_id, sample in zip(run_ids, samples)}
        run_dict_s = []
        out_dict_s = []
        
        print('MAKING ALL NECESSARY DIRECTORIES')
        for index, executor in enumerate(self.executors):
            base_run_dir_simulation = os.path.join(self.base_run_dir,executor.runner.__class__.__name__+f'_{index}')
            run_dirs = [os.path.join(base_run_dir_simulation, run_id) for run_id in run_ids]
            run_dict = {run_id:os.path.join(base_run_dir_simulation, run_id) for run_id in run_ids}
            run_dict_s.append(run_dict)
            for run_dir in run_dirs:
                os.system(f'mkdir {run_dir} -p')
        
        # if base_out_dir is in configs file for the pipeline executor then we will make out_dirs other wise we set it to be the same as the run_dir
        if self.base_out_dir!=None:
            print('BASE OUT DIR IS SPECIFIED SO MAKING OUT DIRS')
            for index, executor in enumerate(self.executors):
                base_out_dir_simulation = os.path.join(self.base_out_dir,executor.runner.__class__.__name__+f'_{index}')
                out_dirs = [os.path.join(base_out_dir_simulation, run_id) for run_id in run_ids]
                out_dict = {run_id:os.path.join(base_out_dir_simulation, run_id) for run_id in run_ids}
                out_dict_s.append(out_dict)
                for out_dir in out_dirs:
                    os.system(f'mkdir {out_dir} -p')
        else:
            print('BASE OUT DIR IS NOT SPECIFIED SO SETTING TO RUN DIRs')
            out_dict_s = run_dict_s          
        
        print('PERFORMING FIRST SIMULATION')
        executor = self.executors[0]

        if self.dynamic_clusters:
            print('INITALIZING CLUSTER 1')
            executor.initialize_client() 
        run_dict = run_dict_s[0]
        out_dict = out_dict_s[0]
        
        if executor.base_run_dir != None or executor.base_out_dir != None:
            raise Warning(f'''The run and out directories are being handeled by the Pipeline Executor.
                            This means that the Simulation Executor run and out directories are being ignored:
                            run: {executor.base_run_dir} out:{executor.base_out_dir}
                            The ExecutorSimulationPileline base run and out directories are being used:
                            run: {self.base_run_dir}, out:{self.base_out_dir}''')         
        
        run_futures = []
        parse_futures_s = [] #each item will be a list of parse futures for one simulation
        parse_futures = []
        for run_id in run_ids:
            sample=run_sample_dict[run_id]
            print('SAMPLE:',sample)
            run_dir=run_dict[run_id]
            next_run_dir=run_dict_s[1][run_id] 
            out_dir=out_dict[run_id]
            # This sends the run_simulation_task to be ran on a worker as soon as possible
            print('MAKING RUNNER FUTURES WITH RUNDIR', run_dir, 'OUT DIR', out_dir)
            run_future = executor.client.submit(run_simulation_task, executor.runner, run_dir, out_dir, sample)                    
            run_futures.append(run_future)
            #Make future for parsing
            print('MAKING PIPELINE PARSE FUTURES FROM', out_dir, 'TO', next_run_dir)
            new_parse_future = executor.client.submit(print_error_wrapper,self.pipeline_parser_functions[0], last_out_dir=out_dir, next_run_dir=next_run_dir, run_id=run_id, future=run_future)
            parse_futures.append(new_parse_future)
        parse_futures_s.append(parse_futures)
        
        print('FUTURES FOR FIRST SIMULATION SENT, SENDING THE REST')
        executing_last_simulation=False
        last_futures=[]
        for index, executor, run_dict, out_dict in zip(range(1,len(self.executors)) ,self.executors[1:], run_dict_s[1:], out_dict_s[1:]):
            if self.dynamic_clusters:
                executor.initialize_client()
            
            print('index',index,len(self.executors)-1)
            if index==len(self.executors)-1:
                executing_last_simulation=True
                print('EXECUTING LAST SIMULATION')
            
            parse_futures=[]
            run_futures = []
            for future in as_completed(parse_futures_s[index-1]):
                _, _, run_id = future.result()
                run_dir = run_dict[run_id]
                out_dir = out_dict[run_id]
                sample = None
                run_future = executor.client.submit(run_simulation_task, executor.runner, run_dir, out_dir, sample)
                run_futures.append(run_future)
                if self.dynamic_clusters and len(run_futures) == len(parse_futures_s[index-1]):
                        #we need the previous cluster to provide the future.result()
                        # Once all the new futures are made we can shut down the previous cluster
                        self.executors[index-1].client.shutdown()
                if executing_last_simulation:
                    last_futures.append(run_future)
                if not executing_last_simulation:
                    next_run_dir = out_dict_s[index+1][run_id]
                    new_parse_future = executor.client.submit(print_error_wrapper,self.pipeline_parser_functions[index], last_out_dir=out_dir, next_run_dir=next_run_dir, run_id=run_id, future=run_future)
                    parse_futures.append(new_parse_future)
            if not executing_last_simulation:
                parse_futures_s.append(parse_futures)
    
        print('WAITING UNTILL LAST FUTURES HAVE FINISHED')
        print('NUM LAST FUTURES:',len(last_futures))
        seq = wait(last_futures)
        
        
        outputs = []
        for res in seq.done:
            outputs.append(res.result())
        if self.last_runner_return_path is not None:
            print("SAVING OUTPUT IN:", self.last_runner_return_path)
            with open(self.last_runner_return_path, "w") as out_file:
                for output in outputs:
                    out_file.write(str(output) + "\n\n")
        print("Finished sequential runs")
        
        if self.dynamic_clusters:
            self.executors[-1].client.shutdown()
        
        return outputs
                        
            