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
import warnings


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
        outputs: list. The results that are returned by the last runner in the pipeline.
    Raises:
    """
    def __init__(self, sampler, **kwargs):
        self.all_futures = []
        
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
        
        
        if self.dynamic_clusters:
            warnings.warn('''
                          If dynamic clusters is true then a results report cannot currently be made,
                          as the futures results are stored in the workers that are dynamically closed.
                          Althought the 'complete_report' will still be made.
                          If you need a results report then set dynamic clusters to False.
                          ''')
                
        
        self.base_run_dir = kwargs.get("base_run_dir")
        if self.base_run_dir==None:
            raise ValueError('''
                             Enchanted surrogates handeles the creation of run directories. 
                             You must supply a base_run_dir in your configs file. Example:
                             executor:
                                type: DaskExecutorSimulation
                                base_run_dir: /project/project_462000451/test-enchanted/trial-dask
                             ...
                             ...
                             ''')

        self.runner_return_path=kwargs.get("runner_return_path")
        self.runner_return_headder = kwargs.get('runnner_return_headder', f'{__class__}: no runner_return_headder, was provided in configs file')
        if self.base_run_dir==None and self.runner_return_path==None:
            warnings.warn(f'NO base_run_dir or runner_return_path WAS DEFINED FOR {__class__}')
        elif self.runner_return_path==None and self.base_run_dir!=None:
            self.runner_return_path = os.path.join(self.base_run_dir, 'runner_return.txt')
        
        self.status_report_dir = kwargs.get("status_report_dir", self.base_run_dir)
        os.system(f'mkdir {self.status_report_dir} -p')
        if self.status_report_dir==None:
            warnings.warn('self.status_report_dir IS None')
        # self.runner = getattr(parsers, runner_args["type"])(**runner_args)
        
        # Status Tracking
        # Order of Execution
        self.execution_order = []
        for i, executor in enumerate(self.executors):
            self.execution_order.append(executor.runner.__class__.__name__)
            if i != len(self.executors)-1:
                self.execution_order.append(pipeline_parser_function_strings[i])
        print('EXECUTION ORDER', self.execution_order)
    
    def initialise_client(self):
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
        print(100 * "=")
        if not self.dynamic_clusters:
            print('MAKING DASK CLUSTERS')
            self.initialise_client()
        
        print('MAKING INITIAL SAMPLES')
        params = self.sampler.get_initial_parameters()
        num_params = len(params)
        
        last_futures = self.submit_batch_of_params(params, self.base_run_dir, self.status_report_dir)
        
        print('WAITING UNTILL LAST FUTURES HAVE FINISHED')
        print('NUM LAST FUTURES:',len(last_futures))
        seq = wait(last_futures)
        print('*'*100,"\nFINISHED PIPELINE FOR ALL SAMPLES\n",'*'*100)
        outputs = []
        for res in seq.done:
            outputs.append(res.result())
        if self.runner_return_path is not None:
            print("SAVING OUTPUT IN:", self.runner_return_path)
            with open(self.runner_return_path, "w") as out_file:
                out_file.write(self.runner_return_headder+'\n')
                for output in outputs:
                    out_file.write(str(output)+"\n")
        
        if self.dynamic_clusters:
            self.executors[-1].client.shutdown()
        
        return outputs
    
    
    
    
    def submit_batch_of_params(self, params:dict, base_run_dir:str=None, status_report_dir:str=None):        
        if base_run_dir==None:
            base_run_dir=self.base_run_dir
            
        print('MAKING RANDOM RUN IDs')
        run_ids = [str(uuid.uuid4()) for p in params]
        run_sample_dict = {run_id:sample for run_id, sample in zip(run_ids, params)}
        run_dict_s = []
        
        # Status tracking
        run_id_futures = {run_id:[] for run_id in run_ids}
        all_futures = []
                
        print('MAKING ALL NECESSARY DIRECTORIES')
        for index, executor in enumerate(self.executors):
            base_run_dir_simulation = os.path.join(self.base_run_dir,executor.runner.__class__.__name__+f'_{index}')
            run_dirs = [os.path.join(base_run_dir_simulation, run_id) for run_id in run_ids]
            run_dict = {run_id:os.path.join(base_run_dir_simulation, run_id) for run_id in run_ids}
            run_dict_s.append(run_dict)
            for run_dir in run_dirs:
                os.system(f'mkdir {run_dir} -p')
        
        print('PERFORMING FIRST SIMULATION')
        executor = self.executors[0]

        if self.dynamic_clusters:
            print(f'0: DYNAMICALLY INITALIZING CLUSTER FOR {executor.runner.__class__.__name__}')
            executor.initialize_client() 
        run_dict = run_dict_s[0]
        
        if executor.base_run_dir != None:
            warnings.warn(f'''
                          The run and out directories are being handeled by the Pipeline Executor.
                          This means that the Simulation Executor run directory is being ignored:
                          run: {executor.base_run_dir}
                          The ExecutorSimulationPileline base run directory is being used:
                          run: {self.base_run_dir}
                          ''')         
        
        parse_futures_s = [] #each item will be a list of parse futures for one simulation
        parse_futures = []
        for run_id in run_ids:
            sample=run_sample_dict[run_id]
            run_dir=run_dict[run_id]
            next_run_dir=run_dict_s[1][run_id] 
            # This sends the run_simulation_task to be ran on a worker as soon as possible
            run_future = executor.client.submit(run_simulation_task, executor.runner, run_dir, sample)                    
            run_id_futures[run_id].append(run_future)
            all_futures.append(run_future)
            #Make future for parsing
            new_parse_future = executor.client.submit(print_error_wrapper,self.pipeline_parser_functions[0], last_run_dir=run_dir, next_run_dir=next_run_dir, run_id=run_id, future=run_future)
            run_id_futures[run_id].append(new_parse_future)
            all_futures.append(new_parse_future)
            parse_futures.append(new_parse_future)
        parse_futures_s.append(parse_futures)
        
        print('FUTURES FOR FIRST SIMULATION SENT, SENDING THE REST')
        executing_last_simulation=False
        last_futures=[]
        for index, executor, run_dict in zip(range(1,len(self.executors)) ,self.executors[1:], run_dict_s[1:]):
            if self.dynamic_clusters:
                print(f'{index}: DYNAMICALLY INITIALIZING CLUSTER FOR {executor.runner.__class__.__name__}')
                executor.initialize_client()
            
            if index==len(self.executors)-1:
                executing_last_simulation=True
                print('EXECUTING LAST SIMULATION')
            
            parse_futures=[]
            run_futures = []
            for future in as_completed(parse_futures_s[index-1]):
                _, _, run_id = future.result()
                run_dir = run_dict[run_id]
                sample = None
                run_future = executor.client.submit(run_simulation_task, executor.runner, run_dir, sample)
                run_id_futures[run_id].append(run_future)
                run_futures.append(run_future)
                all_futures.append(run_future)
                if self.dynamic_clusters and len(run_futures) == len(parse_futures_s[index-1]):
                        #we need the previous cluster to provide the future.result()
                        # Once all the new futures are made we can shut down the previous cluster
                        print(f'{index-1}: DYNAMICALLY SHUTTING DOWN CLUSTER FOR {self.executors[index-1].runner.__class__.__name__}')
                        self.executors[index-1].client.shutdown()
                if executing_last_simulation:
                    last_futures.append(run_future)
                if not executing_last_simulation:
                    next_run_dir = run_dict_s[index+1][run_id]
                    new_parse_future = executor.client.submit(print_error_wrapper,self.pipeline_parser_functions[index], last_run_dir=run_dir, next_run_dir=next_run_dir, run_id=run_id, future=run_future)
                    run_id_futures[run_id].append(new_parse_future)
                    all_futures.append(new_parse_future)
                    parse_futures.append(new_parse_future)
            if not executing_last_simulation:
                parse_futures_s.append(parse_futures)
        
        def result_if_ready(future):
            if future.done():
                return future.result()
            else:
                return None
        
        if status_report_dir != None:
            print('MAKING STATUS REPORTS, CONCURRENTLY, IN DIR:', status_report_dir)
            for future in as_completed(all_futures):
                # Making status reports
                with open(os.path.join(status_report_dir, 'completed_report'), 'w') as file:
                    headder = f"run_id,{','.join(self.execution_order)}\n"
                    file.write(headder)
                    for run_id in run_ids:
                        line = f"{run_id}," + ','.join([str(future.done()) for future in run_id_futures[run_id]]) + '\n'
                        file.write(line)
                
                if not self.dynamic_clusters:
                    with open(os.path.join(status_report_dir, 'results_report'), 'w') as file:
                        headder = f"run_id,{','.join(self.execution_order)}\n"
                        file.write(headder)
                        for run_id in run_ids:
                            line = f"{run_id}," + ','.join([str(result_if_ready(future)) for future in run_id_futures[run_id]]) + '\n'
                            file.write(line)
        
        
        return last_futures

        
            
                        
            