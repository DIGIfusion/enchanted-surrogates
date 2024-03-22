# executors/DaskExecutor.py

import os 
from dask.distributed import Client, as_completed
from dask_jobqueue import SLURMCluster
import uuid
import runners 
from dask import delayed, compute

def run_simulation_task(runner_args, params_from_sampler, base_run_dir): 
    print("Making Run dir")
    run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
    os.mkdir(run_dir)
    runner = getattr(runners, runner_args['type'])(**runner_args) 
    result = runner.single_code_run(params_from_sampler, run_dir)
    return result, params_from_sampler

class DaskExecutor: 
    def __init__(self, sampler, runner_args, base_run_dir: str): 
        print("Starting Setup")
        self.sampler = sampler 
        self.runner_args = runner_args 
        self.base_run_dir = base_run_dir 
        self.max_samples = sampler.num_samples
        # TODO: this is argument
        self.job_script_prologue = ['module load python-data', 'cd /scratch/project_2007159/cursed-tglf/', 'export PYTHONPATH=$PYTHONPATH:/scratch/project_2007159/cursed-tglf/src']
        os.makedirs(base_run_dir, exist_ok=True)
        
    def start_runs(self): 
        print('Beginning Cluster Generation')
        cluster = SLURMCluster(account="project_2005083",
                                    queue="medium",
                                    cores=1, 
                                    memory="12GB", 
                                    processes=1, 
                                    walltime="00:10:00", 
                                    interface='ib0', 
                                    job_script_prologue=self.job_script_prologue)
        cluster.scale(1)
        client = Client(cluster)
        print('Finished Setup')
        print('Starting Database generation')

        print('Creating initial runs')
        futures = [] 
        for _ in range(5): 
            params = self.sampler.get_next_parameter()
            new_future = client.submit(run_simulation_task, self.runner_args, params, self.base_run_dir)
            futures.append(new_future)

        print('Starting search')
        seq = as_completed(futures)
        completed = 0 
        for future in seq: 
            res = future.result() 
            completed += 1
            print(res, completed)
            # TODO: is this waiting for an open node or are we just pushing to queue? 
            if self.max_samples > completed: 
                params = self.sampler.get_next_parameter()
                new_future = client.submit(run_simulation_task, self.runner_args, params, self.base_run_dir)
                seq.add(new_future)
 