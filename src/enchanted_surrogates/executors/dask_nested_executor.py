import os
from .base_executor import Executor
from enchanted_surrogates.executors import simulation_task
from enchanted_surrogates.utils.precise_imports import import_sampler
import subprocess
import time
import warnings
import uuid
import pandas as pd
from enchanted_surrogates.utils.precise_imports import import_executor
from dask.distributed import print, as_completed, wait
from enchanted_surrogates.utils.precise_imports import import_sampler

class DaskNestedExecutor(Executor):
    """
    TODO: add docstring
    """

    def __init__(self, base_run_dir, executors:dict, sampler_kwargs:dict, *args, **kwargs):
        """
        TODO: add docstring
        """
        print('INITIALISING NESTED EXECUTOR')
        self.type = kwargs.get('type')
        self.base_run_dir=base_run_dir
        self.executors_kwargs = list(executors.values())
        print('THE EXECUTORS WILL BE RAN WITH THESE CODES IN THE FOLLOWING ORDER:\n',
              [executor_kwargs['runner_kwargs']['type'] for executor_kwargs in self.executors_kwargs])
        self.executors = [import_executor(executor_kwargs['type'],executor_kwargs) for executor_kwargs in self.executors_kwargs]
        self.sampler_kwargs = sampler_kwargs#kwargs.get('sampler_kwargs')
        sampler_type = self.sampler_kwargs.pop("type")
        self.sampler = import_sampler(type=sampler_type, sampler_kwargs=self.sampler_kwargs) #getattr(importlib.import_module(f'enchanted_surrogates.samplers'),sampler_type)(**sampler_kwargs)

        self.block_until_cluster_started = kwargs.get('block_until_cluster_started', False)
        for executor in self.executors:
            executor.block_until_cluster_started = self.block_until_cluster_started
        self.start_cluster_when_needed = kwargs.get('start_cluster_when_needed', False)
        self.shutdown_finished_clusters = kwargs.get('shutdown_finished_clusters', False)
        self.current_samples_df = None
    
    def clean(self):
        """
        Cleans up resources
        This method is intended to be called when the executor is no longer needed.
        """
        for executor in self.executors:
            executor.clean()

    def start_runs(self):
        """
        TDO: add docstring
        """
        start = time.time()
        print('BASE RUN DIR:', self.base_run_dir)
        if not os.path.exists(self.base_run_dir):
            print('MAKING BASE RUN DIR:',self.base_run_dir)
            os.makedirs(self.base_run_dir)

        if os.path.exists(os.path.join(self.base_run_dir, 'ENCHANTED.FINNISHED')):
            raise FileExistsError(f'''The file: {self.base_run_dir}/ENCHANTED.FINNISHED, exists.
                                  This signifies that there is already data in this folder. 
                                  Aborting to avoid accidental data mixing.''' )

        all_futures = [[]]*len(self.executors)
        
        num_batches = 0
        batch_dir = os.path.join(self.base_run_dir, f'batch_{num_batches}')
        os.makedirs(batch_dir, exist_ok=True)
        #TODO: make it work for batch sampling / active learning. while all([executor.has_budget for executor in executors])
        # HAVE ONE SAMPLER PER CONFIG, IT CAN CALL A SEPERATE SAMPLER PER CODE IF IT WANTS
        # then the output of one runner is passed as a sample to the next runner
        # each runner should be able to handle extra parameters passed in the samples that are the outputs and inputs of the other codes.
        
        samples = self.sampler.get_next_samples()
        self.current_samples_df = pd.DataFrame(samples)
        print('TOTAL NUMBER OF SAMPLES TO BE RAN:', len(self.current_samples_df))
        sampler_cumulative_params = []
        for i, executor in enumerate(self.executors):
            dask_worker_std_out_dir = os.path.join(self.base_run_dir, f'worker_out_sub_executor_{i}')
            if i == 0:
                print(f"STARTING CLUSTER: {i} FOR {executor.runner_kwargs['type']}")
                executor.start_cluster(slurm_out_dir=dask_worker_std_out_dir)
                sampler_i_params = self.sampler.all_samplers[i].parameters
                sampler_cumulative_params += sampler_i_params
                unique_df = self.current_samples_df.drop_duplicates(subset=sampler_i_params)
                filtered_df = unique_df[sampler_i_params]
                sampler_i_samples = filtered_df.to_dict(orient="records")
                futures = executor.submit_batch(sampler_i_samples, base_run_dir=batch_dir)
                all_futures[i] = futures
                
            else:
                if self.start_cluster_when_needed:
                    # This will wait untill atleast one future is finished of the first sub executor
                    ac = as_completed(all_futures[i-1])
                    print('WAITING FOR ONE FUTURE TO COMPLETE FOR EXECUTOR:',i-1)
                    next(ac)
                    print('ONE FUTURE HAS COMPLETED FOR FUTURE',i-1)
                print(f"STARTING CLUSTER: {i} FOR {executor.runner_kwargs['type']}")
                executor.start_cluster(slurm_out_dir=dask_worker_std_out_dir)
                for future in as_completed(all_futures[i-1]):
                    result = future.result()
                    run_dir = result['run_dir']
                    previous_sampler_params = self.sampler.all_samplers[i-1].parameters
                    previous_sample = {k: result[k] for k in previous_sampler_params if k in result}
                    # first filter for all the samples that have the same parameters as the previous sample
                    mask = pd.Series(True, index=self.current_samples_df.index)
                    for k, v in previous_sample.items():
                        mask &= self.current_samples_df[k] == v
                    filtered_df = self.current_samples_df[mask]
                    
                    # Then filter to remove duplicates in the parameters for the current sampler
                    sampler_i_params = self.sampler.all_samplers[i].parameters
                    sampler_cumulative_params += previous_sampler_params
                    sampler_cumulative_params += sampler_i_params
                    sampler_cumulative_params = list(set(sampler_cumulative_params))
                    # print('debug sampler cumulative params', sampler_cumulative_params)
                    # for params in sampler_cumulative_params:
                    #     if params in sampler_negulative_params:
                    #         sampler_negulative_params.remove(params)
                    # print('debug sampler negulative params', sampler_negulative_params)
                    unique_df = filtered_df.drop_duplicates(subset=sampler_i_params)
                    filtered_df = unique_df[sampler_cumulative_params]
                    # Create a DataFrame with repeated values
                    # result.pop('success')
                    # result.pop('run_dir')
                    # for param in sampler_cumulative_params:
                    #     if param in result:
                    #         result.pop(param)
                    extra_df = pd.DataFrame([result] * len(filtered_df))
                    # Concatenate side-by-side
                    filtered_df = filtered_df.reset_index(drop=True)
                    combined_df = pd.concat([filtered_df, extra_df], axis=1)
                    sampler_i_samples = combined_df.to_dict(orient="records")

                    # base_run_dir_tmp = str(os.path.dirname(run_dir))
                    enchanted_dataset_path = os.path.join(os.path.dirname(run_dir), f'enchanted_dataset_{i-1}.csv')
                    dfi = pd.DataFrame({r:[v] for r,v in result.items()})
                    if os.path.exists(enchanted_dataset_path):
                        dfi.to_csv(enchanted_dataset_path, mode='a', header=False, index=False)
                    else:
                        dfi.to_csv(enchanted_dataset_path, mode='w', header=True, index=False)
                    sub_futures = executor.submit_batch(sampler_i_samples, base_run_dir=run_dir)
                    all_futures[i] = all_futures[i] + sub_futures
                    # this should shut down clusters when they have finished being used. Not to be used when doing active learning
                
        # write the results for the last set of futures.
        base_enchanted_dataset_path = os.path.join(self.base_run_dir, f'enchanted_dataset.csv')
        for future in as_completed(all_futures[-1]):
            result = future.result()
            run_dir = result['run_dir']
            enchanted_dataset_path = os.path.join(os.path.dirname(run_dir), f'enchanted_dataset_{len(self.executors)}.csv')
            dfi = pd.DataFrame({r:[v] for r,v in result.items()})
            if os.path.exists(enchanted_dataset_path):
                dfi.to_csv(enchanted_dataset_path, mode='a', header=False, index=False)
            else:
                dfi.to_csv(enchanted_dataset_path, mode='w', header=True, index=False)
            if os.path.exists(base_enchanted_dataset_path):
                dfi.to_csv(base_enchanted_dataset_path, mode='a', header=False, index=False)
            else:
                dfi.to_csv(base_enchanted_dataset_path, mode='w', header=True, index=False)
            
            # this should shut down clusters when they have finished being used. Not to be used when doing active learning
        for i, futures, executor in zip(range(len(self.executors)), all_futures, self.executors):
            completed = 0
            succeded = 0
            for future in as_completed(futures):
                completed += 1
                if future.result()['success']:
                    succeded += 1
                print("| RUNNER:", executor.runner_kwargs['type'],
                      f"\n| NUM COMPLETED: {completed}/{len(futures)}",
                      f"| NUM SUCCEDED: {succeded}/{len(futures)}")
            if self.shutdown_finished_clusters:
                if all(future.done() for future in futures):
                    print('CLOSING WORKERS FOR EXECUTOR:', i)
                    executor.clean()
        print('WALLTIME FOR ENCHANTED SURROGATES:', time.time()-start,'sec')
        print('DATASET IS WRITTEN HERE:',os.path.join(self.base_run_dir, 'enchanted_dataset.csv'))
        print('WRITTING ENCHANTED.FINNISHED FILE, SEE base_run_dir:',self.base_run_dir)
        with open(os.path.join(self.base_run_dir,'ENCHANTED.FINNISHED'), 'w') as file:
            file.write(f'ENCHANTED.FINNISHED, {__class__}')
        print('CLUSTER SHUTDOWN')
        self.clean()
    