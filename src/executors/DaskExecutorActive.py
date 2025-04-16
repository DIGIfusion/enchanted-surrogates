"""
# executors/DaskExecutor.py
Contains logic for executing surrogate workflow on Dask.
"""
import time
from time import sleep
from dask.distributed import Client, as_completed, wait, LocalCluster
from dask_jobqueue import SLURMCluster
import uuid
from common import S
import numpy as np
import importlib
# import runners
from executors.tasks import print_error_wrapper
import os
import traceback
from .tasks import run_simulation_task
import warnings
from common import S
import executors
class DaskExecutorActive():
    """
    Handles the splitting of samples between dask workers for running a simulation.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html

    Attributes:
    """
    
    def __init__(self, sampler=None, total_budget=np.inf, max_cycles=np.inf, time_limit=np.inf, **kwargs):
        if total_budget==np.inf and max_cycles==np.inf and time_limit==np.inf:
            raise ValueError('''NO LIMITATION IS SET. YOU MUST DECLARE AT LEAST ONE OF:
                            total_budget, ie max number of samples,
                            max_cycles, ie max number of active learning cycles, sample , train, sample 
                            time_limit, in seconds''')
        self.total_budget = total_budget
        self.max_cycles = max_cycles
        if self.max_cycles<=1:
            raise ValueError('Max cycles must be above 1. The initial samples are counted as one of the cycles.')
        self.time_limit = time_limit
        
        # This control will be needed by the pipeline and active learning.
        self.sampler = sampler
        assert self.sampler.sampler_interface == S.ACTIVE
        
        self.base_run_dir = kwargs.get("base_run_dir")
        if self.base_run_dir == None:
            raise ValueError("""For the DaskExecutorActive executor a base_run_dir must be specified in the configs file. example:-
                             executor:
                                type: DaskExecutorActive
                                base_run_dir: path/to/base_run_dir
                                ...
                             """)
        if not os.path.isdir(self.base_run_dir):
            print('MAKING BASE RUN DIR')
            os.makedirs(self.base_run_dir, exist_ok=True)
        else:
            if os.path.isfile(os.path.join(self.base_run_dir,'FINNISHED')):
                raise FileExistsError(f'''
                                      The FINNISHED file is in, {self.base_run_dir}, 
                                      this indicates there is already results stored in this directory. 
                                      Please point to an empty or non existent directory''')
        
        self.runner_return_path = kwargs.get("runner_return_path")
        self.runner_return_headder = kwargs.get("runner_return_headder", f'{self.__class__}: no runner_return_headder, was provided in configs file')
        if self.base_run_dir==None and self.runner_return_path==None:
            warnings.warn(f'NO base_run_dir or runner_return_path WAS DEFINED FOR {self.__class__}')
        elif self.runner_return_path==None and self.base_run_dir!=None:
            self.runner_return_path = os.path.join(self.base_run_dir, 'runner_return.txt')
        
        if self.runner_return_path != None and self.runner_return_headder != None:
            print('MAKING RUNNER RETURN FILE')
            with open(self.runner_return_path, 'w') as file:
                file.write(self.runner_return_headder+'\n')
        
        static_executor_kwargs = kwargs['static_executor']
        self.static_executor = getattr(importlib.import_module(f"executors.{static_executor_kwargs['type']}"),static_executor_kwargs['type'])(**static_executor_kwargs) 
        
    def clean(self):
        self.static_executor.clean()

    def initialize_client(self):
        self.static_executor.initialize_client(slurm_out_dir=self.base_run_dir)
            
    def start_runs(self):
        time_start = time.time()
        print(f"STARTING ACTIVE CYCLE, FROM WITHIN A {__class__}")
        
        print('MAKING CLUSTERS')
        self.initialize_client()
                
        print("GENERATING INITIAL SAMPLES:")
        initial_params = self.sampler.get_initial_parameters()
        # initial_params_future = self.static_executor.client.submit(print_error_wrapper,self.sampler.get_initial_parameters)
        # wait(initial_params_future)
        # initial_params = initial_params_future.result()
        
        if self.base_run_dir==None:
            warnings.warn('''
                            No base_run_dir has been provided. It is now assumed that the runner/s being used does not need a run_dir and will be passed None.
                            This could be true if the runner is executing a python function and not a simulation.
                            Otherwise see how to insert a base_run_dir into a config file below:
                            Example
                            
                            executor:
                                type: DaskExecutorActive
                                base_run_dir: /project/path/to/base_run_dir/
                            ...
                            ...
                        ''')
        initial_dir = os.path.join(self.base_run_dir, 'initial_runs')
        os.system(f'mkdir {initial_dir} -p')          
        print("MAKING AND SUBMITING DASK FUTURES FOR INITIAL RUNS")
        initial_futures = self.static_executor.submit_batch_of_params(initial_params, initial_dir)
        
        # train={}
        runner_return=[]
        print('INITIAL FUTURES SUBMITTED WAITING FOR THEM TO COMPLETE')
        seq=wait(initial_futures)
        for future in seq.done:
            out = future.result()
            runner_return.append(out)
            #It is assumed that the runner returns a string in the form:
            # "x0,x1,x3,f(x1 x2 x3)"
            out = out.split(',')
            coordinate = tuple(float(out[i]) for i in range(len(self.sampler.parameters)))
            label = float(out[-1])
            self.sampler.train[coordinate] = label
        runner_return_path = os.path.join(initial_dir,'runner_return.txt')
        print(f'INITIAL FUTURES COMPLETE IN {time.time()-time_start}, WRITING runner_return.txt at:\n', runner_return_path)    
        with open(runner_return_path,'w')as out_file:
            out_file.write(self.runner_return_headder+'\n')
            for out in runner_return:
                out_file.write(str(out)+'\n')
        
        # if 'write_cycle_info' in dir(self.sampler):
        #     time_write = time.time()
        #     print('\nWRITING SAMPLER CYCLE INFO FOR INITIAL CYCLE')
        #     self.sampler.write_cycle_info(initial_dir)
        #     print(f'WRITING SAMPLER CYCLE INFO TOOK {time.time()-time_write} sec')
        
        batch_params = self.sampler.get_next_parameters(initial_dir)
        # batch_params_future = self.static_executor.client.submit(print_error_wrapper,self.sampler.get_next_parameters)
        # wait(batch_params_future)
        # batch_params = batch_params_future.result()
        
            
        num_cycles = 1
        num_samples = len(initial_params)
        time_now=time.time()
        # old_train = train.copy()
        while num_samples<self.total_budget and \
            time_start-time_now<self.time_limit and \
            num_cycles < self.max_cycles and \
            self.sampler.custom_limit_value<self.sampler.custom_limit:
            #---------------------------------------------------------
            cycle_start = time.time()
            cycle_dir = os.path.join(self.base_run_dir, f'active_cycle_{num_cycles}')
            os.system(f'mkdir {cycle_dir} -p')
            print('debug', '(0.8125, 0.5, 0.25)', batch_params)
            futures = self.static_executor.submit_batch_of_params(batch_params, cycle_dir)
            print(f'FUTURES SUBMITTED FOR ACTIVE LEARNING CYCLE {num_cycles}')
            
            # if 'write_cycle_info' in dir(self.sampler) and num_cycles>1:
            #     time_write = time.time()
            #     print(f'WRITING SAMPLER CYCLE INFO FOR CYCLE {num_cycles-1}')
            #     self.sampler.write_cycle_info(cycle_dir)
            #     print(f'WRITING SAMPLER CYCLE INFO TOOK {time.time()-time_write} sec')
            
            print(f'WAITING FOR FUTURES FROM ACTIVE LEARNING CYCLE {num_cycles}')
            runner_return=[]
            seq = wait(futures)
            for future in seq.done:
                out = future.result()
                runner_return.append(out)
                #It is assumed that the runner returns a string in the form:
                # "x0,x1,x3,f(x1 x2 x3)"
                out = out.split(',')
                coordinate = tuple(float(out[i]) for i in range(len(self.sampler.parameters)))
                label = float(out[-1])
                if coordinate in self.sampler.train.keys():
                    print('COORDINATE',coordinate,'in train')
                self.sampler.train[coordinate] = label
            runner_return_path = os.path.join(cycle_dir,'runner_return.txt')
            print('FUTURES COMPLETE')
            print(f'CYCLE {num_cycles} COMPLETE IN {time.time()-cycle_start}, WRITING runner_return.txt at:\n', runner_return_path)
            
            write_runner_start = time.time()
            with open(runner_return_path,'w')as out_file:
                out_file.write(self.runner_return_headder+'\n')
                for out in runner_return:
                    out_file.write(str(out)+'\n')
            print(f'WRITING runner_return.txt TOOK {time.time()-write_runner_start} sec')
            # Update stopping variables
            self.sampler.update_custom_limit_value()            
            num_samples+=len(batch_params)
            time_now = time.time()
            num_cycles+=1
            #--------------------------
            if num_samples>=self.total_budget or \
            time_start-time_now>=self.time_limit or \
            num_cycles >= self.max_cycles or \
            self.sampler.custom_limit_value>=self.sampler.custom_limit:
                break # I can't run get_next_parameters or else parameters will be sampled that never get a label
            #--------------------------
            batch_params = self.sampler.get_next_parameters(cycle_dir)
            # batch_params_future = self.static_executor.client.submit(print_error_wrapper,self.sampler.get_next_parameters)
            # wait(batch_params_future)
            # batch_params = batch_params_future.result()
                    

        if 'write_cycle_info' in dir(self.sampler):
            print(f'WRITING SAMPLER CYCLE INFO FOR CYCLE {num_cycles-1}')
            write_cycle_info_start = time.time()
            self.sampler.write_cycle_info(cycle_dir)
            print(f'WRITING SAMPLER CYCLE INFO TOOK {time.time()-write_cycle_info_start} sec')
                    
        if num_samples>=self.total_budget:
            print('ACTIVE CYCLES FINISHED: num_samples HIT THE total_budget:', self.total_budget)
        if time_start-time_now>=self.time_limit:
            print('ACTIVE CYCLES FINISHED: HIT THE TIME LIMIT:', self.time_limit, 'sec')
        if num_cycles >= self.max_cycles: 
            print('ACTIVE CYCLES FINISHED: num_cycles HIT THE max_cycles:', self.max_cycles)
        if self.sampler.custom_limit_value >= self.sampler.custom_limit:
            print('ACTIVE CYCLES FINISHED: sampler.custom_limit_value HIT THE sampler.custom_limit:', self.sampler.custom_limit)
        
        print("WAITING FOR LAST FUTURES TO COMPLETE")
        wait(futures)
        print("MAKING THE AN OUTPUT FILE CONTAINING ALL DATA:", self.runner_return_path)
        with open(self.runner_return_path, 'w') as file:
            file.write(self.runner_return_headder+'\n')
            for coordinate, label in self.sampler.train.items():
                file.write(f"{','.join([str(co) for co in coordinate])},{label}\n")
        
        if 'post_run' in dir(self.sampler):
            print("WRITING THE SAMPLER SUMMARY FILES IN:", self.base_run_dir)
            self.sampler.post_run(self.base_run_dir)
        
        if 'post_run' in dir(self.static_executor):
            print("WRITING THE STATIC EXECUTOR SUMMARY FILES IN:", self.base_run_dir)
            self.static_executor.post_run(self.base_run_dir, points=np.array(list(self.sampler.train.keys())))
          
        print('ACTIVE CYCLES FINISHED')
        print('num_samples:',num_samples)
        print('num_cycles:',num_cycles)
        time_now = time.time()
        print('wall time sec:', time_now-time_start)
        print('BASE RUN DIR:', self.base_run_dir)
        finished_file_path = os.path.join(self.base_run_dir, 'FINNISHED')
        with open(finished_file_path,'w') as file:
            file.write('FINNISHED')

    