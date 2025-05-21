"""
# executors/DaskExecutor.py
Contains logic for executing surrogate workflow on Dask.
"""
from dask.distributed import Client, as_completed, wait, LocalCluster, Variable
from dask_jobqueue import SLURMCluster
import numpy as np
import importlib
# import runners
import os
from .tasks import run_simulation_task, print_error_wrapper
import warnings
from run import load_configuration
class DaskExecutorSimulation():
    """
    Handles the splitting of samples between dask workers for running a simulation.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html

    Attributes:
    """
    def __init__(self, sampler=None, **kwargs):
        # This control will be needed by the pipeline and active learning.
        self.sampler = sampler
        if type(sampler) == type({}):
            # sampler defined within executor of configs file and so this sampler is actually sampler_args
            sampler_type = sampler.pop("type")
            self.sampler = getattr(importlib.import_module(f'samplers.{sampler_type}'),sampler_type)(**sampler)
        else:
            #sampler defined outside executor in configs file and created in run.py then passed here as a sampler and not sampler_args
            self.sampler = sampler
        
        self.runner_args = kwargs.get("runner")
        if type(self.runner_args) == type(None):
            raise ValueError('''
                             Every ExecutorSimulation needs a runner. 
                             This class defines how the code/simulation should be ran.
                             Here is an example of how it should look in the configs file
                             executor:
                                type: DaskExecutorSimulation
                                runner:
                                    type: SIMPLErunner
                                    executable_path: /path/to/to/simple.sh
                                    other_params: {}
                                    base_run_dir: /path/to/base/run
                                    output_dir: /path/to/base/out
                            ''')
        self.worker_args: dict = kwargs.get("worker_args")
        self.n_jobs: int = kwargs.get("n_jobs", 1)
        
        self.worker_args: dict = kwargs["worker_args"]
        self.n_jobs: int = kwargs.get("n_jobs", 1)
        if self.n_jobs == 1:
            warnings.warn('n_jobs=1 this means there will only be one dask worker. If you want to run samples in paralell please change <executor: n_jobs:> in the config file to be greater than 1.')
        
        self.base_run_dir = kwargs.get("base_run_dir")
        
        self.runner_return_headder = kwargs.get("runner_return_headder", f'{self.__class__}: no runner_return_headder, was provided in configs file')
        if self.base_run_dir==None:
            warnings.warn(f'NO base_run_dir WAS DEFINED FOR {self.__class__}')
            
        self.scale_cluster_to_num_params = kwargs.get('scale_cluster_to_num_params', False)        
        
        self.sub_executor = kwargs.get('sub_executor', None)   
        if type(self.sub_executor) != type(None):
            executor_type = self.sub_executor['type']
            self.sub_executor = getattr(importlib.import_module(f'executors.{executor_type}'),executor_type)(**self.sub_executor)
        
    def clean(self):
        self.client.shutdown()

    def initialize_client(self, slurm_out_dir=None):
        """
        Initializes the client
        Args:
            worker_args (dict): Dictionary of arguments for configuring worker nodes.
            **kwargs: Additional keyword arguments.
        """
        if slurm_out_dir != None:
            jed = self.worker_args.get('job_extra_directives')
            if type(jed) == type(None):
                self.worker_args['job_extra_directives']=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err']
            else:
                self.worker_args['job_extra_directives']+=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err']
        print("Initializing DASK client")
        if self.worker_args.get("local", False):
            # TODO: Increase num parallel workers on local
            print(f"MAKING A LOCAL CLUSTER FOR {self.runner_args['type']}")
            self.cluster = LocalCluster(**self.worker_args)
        else:
            print(f"MAKING A SLURM CLUSTER FOR {self.runner_args['type']}")
            # local_directory = os.path.join(self.base_run_dir, 'scheduler_local_dir')
            # if not os.path.exists(local_directory):
            #     os.makedirs(local_directory)
            # os.environ["DASK_TEMPORARY_DIRECTORY"] = local_directory
            # self.worker_args['local_directory'] = local_directory
            print('TMPDIR', os.environ.get('TMPDIR'))
            self.cluster = SLURMCluster(**self.worker_args)
            self.cluster.job_cls.submit_command = 'ssh -i ~/.ssh/lumi-key uan03 sbatch'
            
            worker_job_script_dir = os.path.join(os.path.expanduser('~'), 'enchanted_dask_worker_job_script_dir')##'/users/danieljordan/dask_tmp'##
            if not os.path.exists(worker_job_script_dir):
                os.makedirs(worker_job_script_dir)
            
            # performing monkey patch of dask jobqueue so that we can set our own file location for the workers submit file
            from dask.utils import tmpfile
            import logging
            from contextlib import contextmanager
            @contextmanager
            def job_file(self):
                """Write job submission script to temporary file"""
                with tmpfile(extension="sh", dir=worker_job_script_dir) as fn:
                    logger = logging.getLogger(__name__)
                    with open(fn, "w") as f:
                        logger.debug("writing job script: \n%s", self.job_script())
                        f.write(self.job_script())
                    yield fn
            self.cluster.job_cls.job_file = job_file
            # monkey patch complete
            self.cluster.scale(self.n_jobs)    
            print('THE JOB SCRIPT FOR A WORKER IS:')
            print(self.cluster.job_script())
            
        self.client = Client(self.cluster ,timeout=180)
        print('DASHBOARD LINK',self.client.dashboard_link)
    
    # This is only ran for the top executor, sub executors will run submit_batch of params
    def start_runs(self):
        if self.base_run_dir==None:
            # Check to see if the base_run_dir was defined in general namelist of config file
            config_path = os.environ.get('ENCHANTED_CONFIG_PATH')
            # The downside of this is that the full config file path needs to be passed for the worker to find it. 
            config = load_configuration(config_path)
            try: 
                self.base_run_dir = config.general['top_executor_base_run_dir']
            except:
                raise ValueError(f'When executing start_runs of {__class__} a self.base_run_dir must be specified.')
        
        if not os.path.exists(self.base_run_dir):
            os.makedirs(self.base_run_dir)

        if os.path.exists(os.path.join(self.base_run_dir, 'ENCHANTED.FINNISHED')):
            raise FileExistsError(f'''The file: {self.base_run_dir}/ENCHANTED.FINNISHED, exists.
                                  This signifies that there is already data in this folder. 
                                  Aborting to avoid accidental data mixing.''' )
                        
        
        print(f"STARTING RUNS FOR RUNNER {self.runner_args['type']}, FROM WITHIN A {__class__}")
        
        print('MAKING CLUSTER')
        self.initialize_client(slurm_out_dir=os.path.join(self.base_run_dir, 'worker_out_base_executor'))
        
        print("GENERATING INITIAL SAMPLES:")
        params = self.sampler.get_initial_parameters(self.base_run_dir)
        futures = self.submit_batch_of_params(params, self.base_run_dir)
        
        print("DASK FUTURES SUBMITTED, WAITING FOR THEM TO COMPLETE")
        sub_executors = [self.sub_executor]
        sub_executor = self.sub_executor
        while True:
            if type(sub_executor) == type(None):
                break
            sub_executor = sub_executor.sub_executor
            
            if type(sub_executor) != type(None):
                sub_executors.append(sub_executor)
        
        all_outputs = {} # key=base_run_dir: value=list of outputs
        if type(self.sub_executor) != type(None):
            all_futures = [futures] + [[]]*len(sub_executors) # A list of lists, each sublist is a list of futures, one for each executor
            base_run_dir_to_runner_return_headder = {}
            for i, sub_executor in enumerate(sub_executors):
                dask_worker_std_out_dir = os.path.join(self.base_run_dir, f'worker_out_sub_executor_{i}')
                print('INITIALIZING SUB EXECUTOR:',i)
                sub_executor.initialize_client(slurm_out_dir=dask_worker_std_out_dir)
                for future in as_completed(all_futures[i]):
                    output, run_dir = future.result() 
                    base_run_dir_tmp = str(os.path.dirname(run_dir))
                    if i == 0:
                        base_run_dir_to_runner_return_headder[base_run_dir_tmp] = self.runner_return_headder
                    else:
                        base_run_dir_to_runner_return_headder[base_run_dir_tmp] = sub_executors[i-1].runner_return_headder
                    if base_run_dir_tmp not in all_outputs.keys():
                        all_outputs[base_run_dir_tmp] = []
                    all_outputs[base_run_dir_tmp].append(output)
                    params = sub_executor.sampler.get_initial_parameters(base_run_dir=run_dir)
                    sub_futures = sub_executor.submit_batch_of_params(params=params, base_run_dir=run_dir)
                    all_futures[i+1] = all_futures[i+1] + sub_futures
            
            # Getting output from futures of last sub_executor
            for future in as_completed(all_futures[-1]):
                    output, run_dir = future.result() 
                    base_run_dir_tmp = str(os.path.dirname(run_dir))
                    if i == 0:
                        base_run_dir_to_runner_return_headder[base_run_dir_tmp] = self.runner_return_headder
                    else:
                        base_run_dir_to_runner_return_headder[base_run_dir_tmp] = sub_executors[i-1].runner_return_headder
                    if base_run_dir_tmp not in all_outputs.keys():
                        all_outputs[base_run_dir_tmp] = []
                    all_outputs[base_run_dir_tmp].append(output)
                    
            wait(all_futures)
                    
            for base_run_dir, outputs in all_outputs.items():
                runner_return_path = os.path.join(base_run_dir,'runner_return.csv')
                print("SAVING OUTPUT IN:", runner_return_path)
                runner_return_headder = base_run_dir_to_runner_return_headder[base_run_dir]
                with open(runner_return_path, "w") as out_file:
                    outputs = [runner_return_headder] + outputs
                    outputs = [line+'\n' for line in outputs]
                    out_file.writelines(outputs)

        else:
            outputs = []
            for future in as_completed(futures):
                output, run_dir = future.result() # Not sure what is going on, return should be (out_string, run_path) what I get is ((out_string, run_path), run_path)
                base_run_dir_tmp = str(os.path.dirname(run_dir))
                if base_run_dir_tmp not in all_outputs.keys():
                    all_outputs[base_run_dir_tmp] = []
                outputs.append(output)            
            runner_return_path = os.path.join(self.base_run_dir,'runner_return.csv')                
            with open(runner_return_path, "w") as out_file:
                    outputs = [self.runner_return_headder] + outputs
                    outputs = [line+'\n' for line in outputs]
                    out_file.writelines(outputs)

            

        print("Finished sequential runs")
        
        print('WRITTING ENCHANTED.FINNISHED FILE, SEE base_run_dir:',self.base_run_dir)
        with open(os.path.join(self.base_run_dir,'ENCHANTED.FINNISHED'), 'w') as file:
            file.write(f'ENCHANTED.FINNISHED, {__class__}')
    
    def submit_batch_of_params(self, params: list, base_run_dir: str, *args, **kwargs): 
        if self.scale_cluster_to_num_params:
            self.n_jobs = len(params)
            self.cluster.scale(self.n_jobs)
        
        print('MAKING TEMPORARY RUNNER')
        runner_type = self.runner_args['type']
        runner = getattr(importlib.import_module(f'runners.{runner_type}'),runner_type)(**self.runner_args)
        if 'pre_run' in dir(runner):
            print('PRERUN IS IN RUNNER')
            runner.pre_run(base_run_dir=base_run_dir, params=params, *args, **kwargs)
            
        print("MAKING AND SUBMITTING DASK FUTURES")      
        futures = []
        
        for index, sample in enumerate(params):
            if type(base_run_dir) == type([]):
                brd = base_run_dir[index]
            else:
                brd = base_run_dir
            print(index+1,'MAKING NEW FUTURE FOR:', self.runner_args['type'])
            new_future = self.client.submit(
                run_simulation_task, runner_args=self.runner_args, 
                base_run_dir=brd, index=index, params=sample, future=None, *args, **kwargs
            )
            futures.append(new_future)
            
        return futures
    
