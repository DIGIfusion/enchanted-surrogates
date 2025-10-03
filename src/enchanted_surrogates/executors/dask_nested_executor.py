import os
from .base_executor import Executor
from enchanted_surrogates.executors import simulation_task
from enchanted_surrogates.utils.precise_imports import import_sampler
import subprocess
import time
import warnings
import uuid
import numpy as np
import pandas as pd
from enchanted_surrogates.utils.precise_imports import import_executor
from dask.distributed import print, as_completed, wait
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.utils.print_stats_table import print_stats_table
from queue import Queue 
import threading

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
        self.executor_names = list(executors.keys())
        self.executors_kwargs = list(executors.values())
        self.executor_types = [exe_kwargs['type'] for exe_kwargs in self.executors_kwargs]
        self.runner_types = [executor_kwargs['runner_kwargs']['type'] for executor_kwargs in self.executors_kwargs]
        self.dask_worker_std_out_dirs = [os.path.join(self.base_run_dir, f'worker_out_{self.runner_types[i]}_{i}') for i in range(len(executors))]
        print('THE EXECUTORS WILL BE RAN WITH THESE CODES IN THE FOLLOWING ORDER:\n',
              self.runner_types)
        self.executors = []
        self.reuse_bool = []
        self.reuse_index = []
        self.keep_alive = []
        # take into account reusing executors feature might be in place
        for exe_type, exe_kwargs in zip(self.executor_types, self.executors_kwargs):
            if exe_type in self.executor_names:
                exe = import_executor(executors[exe_type]['type'], executors[exe_type])
                index = self.executor_names.index(exe_type)
                self.keep_alive.append(index)
                self.reuse_bool.append(True)
                self.reuse_index.append(index)
                if index > len(self.executors):
                    raise RuntimeError('YOU ARE TRYING TO REUSE AN EXECUTOR THAT HAS NOT YET BEEN CREATED, YOU CAN ONLY REUSE ALREADY MADE EXECUTORS')
                exe.runner_kwargs = exe_kwargs['runner_kwargs']
                self.executors.append(exe)
                
            else:
                self.reuse_bool.append(False)
                self.reuse_index.append(None)
                self.executors.append(import_executor(exe_type,exe_kwargs))
        
        self.sampler_kwargs = sampler_kwargs#kwargs.get('sampler_kwargs')
        sampler_type = self.sampler_kwargs.pop("type")
        self.sampler = import_sampler(type=sampler_type, sampler_kwargs=self.sampler_kwargs) #getattr(importlib.import_module(f'enchanted_surrogates.samplers'),sampler_type)(**sampler_kwargs)

        self.block_until_cluster_started = kwargs.get('block_until_cluster_started', False)
        for executor in self.executors:
            executor.block_until_cluster_started = self.block_until_cluster_started
        self.start_cluster_when_needed = kwargs.get('start_cluster_when_needed', False)
        self.shutdown_finished_clusters = kwargs.get('shutdown_finished_clusters', False)
        self.current_samples_df = None
        self.all_results = [{} for _ in self.executors]
        self.result_queue = Queue()
        self.all_futures = [[] for _ in self.executors]
        self._futures_locks = [threading.Lock() for _ in range(len(self.executors))]
        self.completed = [0 for _ in self.executors]
        self.succeded = [0 for _ in self.executors]
        
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

        if os.path.exists(os.path.join(self.base_run_dir, 'ENCHANTED.FINISHED')):
            raise FileExistsError(f'''The file: {self.base_run_dir}/ENCHANTED.FINISHED, exists.
                                  This signifies that there is already data in this folder. 
                                  Aborting to avoid accidental data mixing.''' )
        
        
        batch_num = 0
        batch_dir = os.path.join(self.base_run_dir, f'batch_{batch_num}')
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
            if i == 0:
                print(f"STARTING CLUSTER: {i} FOR {executor.runner_kwargs['type']}")
                executor.start_cluster(slurm_out_dir=self.dask_worker_std_out_dirs[i])
                sampler_i_params = self.sampler.all_samplers[i].parameters
                sampler_cumulative_params += sampler_i_params
                unique_df = self.current_samples_df.drop_duplicates(subset=sampler_i_params)
                filtered_df = unique_df[sampler_i_params]
                sampler_i_samples = filtered_df.to_dict(orient="records")
                print('debug SUBMITTING IN BACKGROUND, (should be 0) EXE_INDEX', i)
                self.submit_inbackground(samples=sampler_i_samples, base_run_dir=batch_dir, exe_index=i)
            else:
                if self.start_cluster_when_needed:
                    # This will wait untill atleast one future is finished of the first sub executor
                    previous_success = False
                    print('*'*100,f'\n\nWAITING FOR ONE FUTURE TO COMPLETE WITH SUCCESS FROM RUNNER {i-1}: {self.runner_types[i-1]}\n\n','*'*100)
                    start = time.time()
                    while not previous_success:
                        done_status = [future.done() for future in self.all_futures[i-1]]
                        if any(done_status):
                            prelim_results = [self.get_result(future, timeout=2) for future in self.all_futures[i-1]]
                            suc = [] 
                            for pr in prelim_results:
                                if pr:
                                    print('debug pr', pr)
                                    suc.append(pr['success'])
                                # successes = [res['success'] for res in prelim_results if res]
                            if any(suc):
                                previous_success = True
                                print('ONE FUTURE HAS COMPLETED WITH SUCCESS FOR FUTURE',i-1)                        
                
                print(f"STARTING CLUSTER: {i} FOR {executor.runner_kwargs['type']}")
                # start cluster or assign client when reusing
                if self.reuse_bool[i]:
                    executor.client = self.executors[self.reuse_index[i]].client
                    executor.cluster = self.executors[self.reuse_index[i]].cluster
                else:
                    executor.start_cluster(slurm_out_dir=self.dask_worker_std_out_dirs[i]) # If block_until_cluster_started = True, this can cause issues with the scheduler not being responsive enough for dask

                expected_num_fut = self.sampler.depth_num_runs[batch_num][i-1]
                done, pending = self.seperate_futures(self.all_futures[i-1])
                num_done = 0
                while num_done < expected_num_fut:
                    for j, future in enumerate(done):
                        result = self.get_result(future)
                        self.update_completion_stats(result, i-1)
                        
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
                        unique_df = filtered_df.drop_duplicates(subset=sampler_i_params)
                        filtered_df = unique_df[sampler_cumulative_params]
                        extra_df = pd.DataFrame([result] * len(filtered_df))
                        filtered_df = filtered_df.reset_index(drop=True)
                        for col in extra_df.columns:
                            if col in filtered_df.columns:
                                filtered_df = filtered_df.drop(columns = [col])
                        combined_df = pd.concat([filtered_df, extra_df], axis=1)
                        sampler_i_samples = combined_df.to_dict(orient="records")
                        enchanted_dataset_path = os.path.join(os.path.dirname(run_dir), f'enchanted_dataset_{i-1}.csv')
                        dfi = pd.DataFrame({r:[v] for r,v in result.items()})
                        if os.path.exists(enchanted_dataset_path):
                            dfi.to_csv(enchanted_dataset_path, mode='a', header=False, index=False)
                        else:
                            dfi.to_csv(enchanted_dataset_path, mode='w', header=True, index=False)
                        print('debug SUBMITTING IN BACKGROUND, EXE_INDEX', i)
                        self.submit_inbackground(samples=sampler_i_samples, base_run_dir=run_dir, exe_index=i)
                        # this should shut down clusters when they have finished being used. Not to be used when doing active learning
                        if self.shutdown_finished_clusters:
                            cluster_status = [future.done() for future in self.all_futures[i-1]]
                            print(f"STATUS OF CLUSTER: {i-1} | {self.executors[i-1].runner_kwargs['type']} | {cluster_status}")
                            if all(cluster_status) and not i-1 in self.keep_alive:
                                print(f'GRABBING RESULTS FOR RUNNER {self.runner_types[i-1]} BEFORE SHUTDOWN')
                                self.get_results(self.all_futures[i-1])
                                print(f'CLOSING WORKERS FOR EXECUTOR: {i-1}')
                                self.executors[i-1].clean()
                    num_done += len(done)
                    done, pendng = self.seperate_futures(done)

                
        # write the results for the last set of futures.
        base_enchanted_dataset_path = os.path.join(self.base_run_dir, f'enchanted_dataset.csv')
        done, pending = self.seperate_futures(self.all_futures[-1])
        while pending:
            for j, future in enumerate(done):
                result = self.get_result(future)
                self.update_completion_stats(result,len(self.executors)-1)

                run_dir = result['run_dir']
                enchanted_dataset_path = os.path.join(os.path.dirname(run_dir), f'enchanted_dataset_{len(self.executors)-1}.csv')
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
                if self.shutdown_finished_clusters:
                    cluster_status = [future.done() for future in self.all_futures[-1]]
                    print(f"STATUS OF CLUSTER: {len(self.executors)-1} | {executor.runner_kwargs['type']} | {cluster_status}")
                    if all(cluster_status) and not len(self.executors) in self.keep_alive:
                        print(f'GRABBING RESULTS FOR RUNNER {self.runner_types[-1]} BEFORE SHUTDOWN')
                        self.get_results(self.all_futures[-1])        
                        print('CLOSING WORKERS FOR EXECUTOR:', len(self.executors)-1)
                        self.executors[-1].clean()
            done, pending = self.seperate_futures(done)
        
        print('ALL FUTURES SHOULD BE FINISHED NOW')
        # print('WAITING FOR ALL FUTURES TO BE FINISHED')
        # for futures in self.all_futures:
        #     wait(futures)
        print('WALLTIME FOR ENCHANTED SURROGATES:', time.time()-start,'sec')
        print('DATASET IS WRITTEN HERE:',os.path.join(self.base_run_dir, 'enchanted_dataset.csv'))
        print('WRITTING ENCHANTED.FINISHED FILE, SEE base_run_dir:',self.base_run_dir)
        with open(os.path.join(self.base_run_dir,'ENCHANTED.FINISHED'), 'w') as file:
            file.write(f'ENCHANTED.FINISHED, {__class__}')
        print('CLUSTER SHUTDOWN')
        self.clean()
        
    def get_results(self, futures):
        results = []
        for future in futures:
            results.append(self.get_result(future))
        return results
    
    def seperate_futures(self, futures):
        done = [future for future in futures if future.done()]
        pending = [future for future in futures if not future.done()]
        return done, pending
    
    def get_result(self, future, timeout=60):
        result = None
        started = time.time()
        while not result and time.time()-started < timeout:
            for results in self.all_results: # check to see if the result is in all_results
                if future.key in results:
                    result = results[future.key]
            if not result: # if not in all results empty the results_queue
                while not self.result_queue.empty():
                    i, key, result_q = self.result_queue.get()
                    self.all_results[i][key] = result_q
                    if future.key == key:
                        result = result_q
            if not result:
                print('SLEEPING TO GIVE THE CONCURRENT LOOP A CHANCE TO PUT THE FUTURE RESULT IN THE ALL_RESULTS QUEUE, sec passed:', time.time()-started)
                time.sleep(0.5)
        return result
    
    def update_completion_stats(self, result, exe_i, batch_num=0):
        self.completed[exe_i] += 1
        if result['success']:
            self.succeded[exe_i] += 1
        completion_stats = {
            'header': f"BATCH {batch_num} COMPLETION STATS",
            'subheader': f"{self.executors[exe_i].runner_kwargs['type']}",
            'NESTED DEPTH': f"{exe_i}",
            'COMPLETED': f"{self.completed[exe_i]}/{len(self.all_futures[exe_i])}",
            'SUCCEDED': f"{self.succeded[exe_i]}/{len(self.all_futures[exe_i])}"
        }   
        total_completion_stats = {
            'header': f"BATCH {batch_num} COMPLETION STATS",
            'subheader': f"ENTIRE NESTED PIPELINE",
            'COMPLETED': f"{np.sum(self.completed)}/{self.sampler.total_num_runs[batch_num]}",
            'SUCCEDED': f"{np.sum(self.succeded)}/{self.sampler.total_num_runs[batch_num]}"
        }
        print_stats_table(completion_stats)
        print_stats_table(total_completion_stats)
    
    # def submit_inbackground(self, samples, base_run_dir, exe_index):
    #     def submit_and_queue(samples, base_run_dir, exe_index):
    #         futures = self.executors[exe_index].submit_batch(samples, base_run_dir)
    #         self.all_futures[exe_index].extend(futures)
    #         for future, result in as_completed(futures, with_results=True):
    #             self.result_queue.put((exe_index, future.key, result))
    #     threading.Thread(target=submit_and_queue, args=(samples, base_run_dir, exe_index), daemon=True).start()

    # def submit_inbackground(self, samples, base_run_dir, exe_index):
    #     def submit_and_stream(samples, base_run_dir, exe_index):
    #         from dask.distributed import Client, as_completed
    #         local_client = Client(address=self.executors[exe_index].client.scheduler.address, asynchronous=False)
    #         try:
    #             # submit batch using the local client API provided by your executor
    #             futures = self.executors[exe_index].submit_batch(samples, base_run_dir, client=local_client)
    #             self.all_futures[exe_index].extend(futures)

    #             for fut in as_completed(futures):
    #                 # streaming: put each result as it finishes
    #                 res = fut.result()
    #                 self.result_queue.put((exe_index, fut.key, res))
    #             wait(futures)
    #         finally:
    #             local_client.close()

    #     threading.Thread(
    #         target=submit_and_stream,
    #         args=(samples, base_run_dir, exe_index),
    #         daemon=True
    #     ).start()
    
    # def submit_inbackground(self, samples, base_run_dir, exe_index):
    #     def submit_and_stream(samples, base_run_dir, exe_index):
    #         from dask.distributed import Client, as_completed
    #         local_client = Client(address=self.executors[exe_index].client.scheduler.address, asynchronous=False)
    #         try:
    #             # submit batch using the local client API provided by your executor
    #             futures = self.executors[exe_index].submit_batch(samples, base_run_dir, client=local_client)
    #             self.all_futures[exe_index].extend(futures)
                
    #             while len(futures) > 0:
    #                 for fut in futures:
    #                     if fut.done():
    #                         res = fut.result()
    #                         self.result_queue.put((exe_index, fut.key, res))
    #                         #remove finnished future from list
    #                         futures = [f for f in futures if f.key != fut.key]
    #         finally:
    #             local_client.close()

    #     threading.Thread(
    #         target=submit_and_stream,
    #         args=(samples, base_run_dir, exe_index),
    #         daemon=True
    #     ).start()
        
    def submit_inbackground(self, samples, base_run_dir, exe_index):
        def submit_and_stream(samples, base_run_dir, exe_index):
            from distributed import Client, wait
            local_client = Client(address=self.executors[exe_index].client.scheduler.address,
                                asynchronous=False)
            try:
                futures = self.executors[exe_index].submit_batch(
                    samples, base_run_dir, client=local_client
                )
                
                with self._futures_locks[exe_index]:
                    self.all_futures[exe_index].extend(futures)

                pending = set(futures)
                while pending:
                    done, pending = wait(list(pending), return_when='FIRST_COMPLETED')
                    for fut in done:
                        res = fut.result()
                        # try:
                        #     
                        # except Exception as e:
                        #     res = {"error": str(e)}
                        self.result_queue.put((exe_index, fut.key, res))
            finally:
                local_client.close()
        threading.Thread(
            target=submit_and_stream,
            args=(samples, base_run_dir, exe_index),
            daemon=True
        ).start()