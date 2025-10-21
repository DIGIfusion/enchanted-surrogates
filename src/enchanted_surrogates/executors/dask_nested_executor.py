"""
TODO: Add module docstring
"""
import os
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from dask.distributed import print, as_completed, wait
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.utils.precise_imports import import_executor
from .base_executor import Executor
from enchanted_surrogates.utils.print_stats_table import print_stats_table

# TODO: IMPLIMENT DYNAMIC SCALE DOWN. The most likely method for success is to remove
# any storage of futures that could be tied to workers, stopping them from retireing.
# Then using cluster.scale().also call future.cancel() and future.release() del future
# to be sure it is not holding back the dynamic scaling

class DaskNestedExecutor(Executor):
    """
    TODO: add docstring
    """

    def __init__(self, base_run_dir, executors:dict, sampler_config:dict, *args, **kwargs):
        """
        TODO: add docstring
        """
        print('INITIALISING NESTED EXECUTOR')
        self.type = kwargs.get('type')
        self.base_run_dir=base_run_dir
        self.executor_names = list(executors.keys())
        self.executors_config = list(executors.values())
        self.executor_types = [exe_config['type'] for exe_config in self.executors_config]
        self.runner_types = [executor_config['runner_config']['type'] for executor_config in self.executors_config]
        self.dask_worker_std_out_dirs = [os.path.join(self.base_run_dir, f'worker_out_{self.runner_types[i]}_{i}') for i in range(len(executors))]
        print('THE EXECUTORS WILL BE RAN WITH THESE CODES IN THE FOLLOWING ORDER:\n',
              self.runner_types)
        self.executors = []
        self.reuse_bool = []
        self.reuse_index = []
        self.keep_alive = []
        # take into account reusing executors feature might be in place
        for exe_type, exe_config in zip(self.executor_types, self.executors_config):
            exe_config["base_run_dir"] = ""
            exe_config["sampler_config"] = {}
            if exe_type in self.executor_names:
                exe = import_executor(executors[exe_type]['type'], executors[exe_type])
                index = self.executor_names.index(exe_type)
                self.keep_alive.append(index)
                self.reuse_bool.append(True)
                self.reuse_index.append(index)
                if index >= len(self.executors):
                    raise RuntimeError('YOU ARE TRYING TO REUSE AN EXECUTOR THAT HAS NOT YET BEEN CREATED, YOU CAN ONLY REUSE ALREADY MADE EXECUTORS')
                exe.runner_config = exe_config['runner_config']
                self.executors.append(exe)
                
            else:
                self.reuse_bool.append(False)
                self.reuse_index.append(None)
                self.executors.append(import_executor(exe_type, exe_config))
        
        self.sampler_config = sampler_config#kwargs.get('sampler_config')
        sampler_type = self.sampler_config.pop("type")
        self.sampler = import_sampler(type=sampler_type, sampler_config=self.sampler_config) #getattr(importlib.import_module(f'enchanted_surrogates.samplers'),sampler_type)(**sampler_config)

        self.block_until_cluster_started = kwargs.get('block_until_cluster_started', False)
        for executor in self.executors:
            executor.block_until_cluster_started = self.block_until_cluster_started
        self.start_cluster_when_needed = kwargs.get('start_cluster_when_needed', False)
        self.shutdown_finished_clusters = kwargs.get('shutdown_finished_clusters', False)
        self.do_dynamic_scale_down = kwargs.get('do_dynamic_scale_down', False)
        if self.do_dynamic_scale_down:
            warnings.warn('DYNAMIC SCALE DOWN IS NOT YET IMPLIMENTED')
        # self.stop_dynamic_scale_events = [[threading.Event() for _ in self.executors]]
        
        self.current_samples_df = None
        # self.all_results = [{} for _ in self.executors]
        # self.result_queue = Queue()
        self.all_futures = [[] for _ in self.executors]
        self.all_fut_to_rundir = {}
        # self._futures_locks = [threading.Lock() for _ in range(len(self.executors))]
        self.completed = [0 for _ in self.executors]
        self.succeded = [0 for _ in self.executors]
        self.log_stats = {'header':'LOG STATS','fut_res_not_available':0, 'fut_res_available':0}
        
        # if self.do_dynamic_scale_down and self.shutdown_finished_clusters:
        #     warnings.warn('BOTH do_dynamic_scale_down AND shutdown_finished_clusters ARE TRUE. BOTH IS OVERKILL. DEFAULTING TO ONLY do_dynamic_scale_down')
        #     self.shutdown_finished_clusters = False
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

        # if self.do_dynamic_scale_down:
        #     # 1. Basic root logger configuration (call this once, before starting threads)
        #     logfile = os.path.join(self.base_run_dir, 'dynamic_scaling_log.txt')
        #     # works like `touch` — creates or updates mtime
        #     open(logfile, "a").close()
        #     handler = RotatingFileHandler(logfile, maxBytes=10_000_000, backupCount=5)
        #     formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        #     handler.setFormatter(formatter)
        #     handler.setLevel(logging.DEBUG)
        #     root = logging.getLogger()
        #     root.setLevel(logging.DEBUG)
        #     # remove default handlers if you want only file output
        #     for h in list(root.handlers):
        #         root.removeHandler(h)
        #     root.addHandler(handler)

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
            # if self.do_dynamic_scale_down:
            #     self.start_dynamic_scale_down_background(exe_i=i, batch_num=batch_num)
            if i == 0:
                print(f"STARTING CLUSTER: {i} FOR {executor.runner_config['type']}")
                executor.start_cluster(slurm_out_dir=self.dask_worker_std_out_dirs[i])
                sampler_i_params = self.sampler.all_samplers[i].parameters
                sampler_cumulative_params += sampler_i_params
                unique_df = self.current_samples_df.drop_duplicates(subset=sampler_i_params)
                filtered_df = unique_df[sampler_i_params]
                sampler_i_samples = filtered_df.to_dict(orient="records")
                futures, fut_to_rundir = self.executors[i].submit_batch(sampler_i_samples, base_run_dir=batch_dir, include_fut_to_rundir=True)
                self.all_futures[i].extend(futures)
                self.all_fut_to_rundir.update(fut_to_rundir)
            else:
                # before entering while pending: (once per executor i)
                sampler_i_params = self.sampler.all_samplers[i].parameters
                previous_sampler_params = self.sampler.all_samplers[i-1].parameters
                sampler_cumulative_params = list(set(sampler_cumulative_params + previous_sampler_params + sampler_i_params))
                if self.start_cluster_when_needed:
                    # This will wait untill atleast one future is finished of the first sub executor
                    previous_success = False
                    print('*'*100,f'\n\nTIME: {datetime.now()}\nWAITING FOR ONE FUTURE TO COMPLETE WITH SUCCESS FROM RUNNER {i-1}: {self.runner_types[i-1]}\n\n','*'*100)
                    start = time.time()
                    while not previous_success:
                        done_status = [future.done() for future in self.all_futures[i-1]]
                        if any(done_status):
                            prelim_results = [self.get_result(future, timeout=2, silent=False) for future in self.all_futures[i-1]]
                            suc = [] 
                            for pr in prelim_results:
                                if pr:
                                    suc.append(pr['success'])
                            if any(suc):
                                previous_success = True
                                print(f'{datetime.now()} ONE FUTURE HAS COMPLETED WITH SUCCESS FOR NESTED DEPTH:',i-1)
                        time.sleep(1)
                
                print(f"{datetime.now()} STARTING CLUSTER: {i} FOR {executor.runner_config['type']}")
                # start cluster or assign client when reusing
                if self.reuse_bool[i]:
                    executor.client = self.executors[self.reuse_index[i]].client
                    executor.cluster = self.executors[self.reuse_index[i]].cluster
                else:
                    executor.start_cluster(slurm_out_dir=self.dask_worker_std_out_dirs[i]) # If block_until_cluster_started = True, this can cause issues with the scheduler not being responsive enough for dask

                futures_check = {fut.key:fut for fut in self.all_futures[i-1]}
                done = [fut for fut in futures_check.values() if fut.done()]
                while futures_check:
                    for j, future in enumerate(done):
                        result = self.get_result(future, timeout=5)
                        if not result:
                            continue
                        self.update_completion_stats(result, i-1)
                        # if self.do_dynamic_scale_down:
                        #     self.dynamic_scale_down(exe_i=i-1, batch_num=batch_num)
                        futures_check.pop(future.key)
                        run_dir = result['run_dir']
                        previous_sample = {k: result[k] for k in previous_sampler_params if k in result}
                        # first filter for all the samples that have the same parameters as the previous sample
                        mask = pd.Series(True, index=self.current_samples_df.index)
                        for k, v in previous_sample.items():
                            mask &= self.current_samples_df[k] == v
                        filtered_df = self.current_samples_df[mask]
                        
                        # Then filter to remove duplicates in the parameters for the current sampler
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
                        sub_futures, fut_to_rundir = self.executors[i].submit_batch(sampler_i_samples, base_run_dir=run_dir, include_fut_to_rundir=True, request_errors=True)
                        self.all_futures[i].extend(sub_futures)
                        self.all_fut_to_rundir.update(fut_to_rundir)
                        # this should shut down clusters when they have finished being used. Not to be used when doing active learning
                    done = [fut for fut in futures_check.values() if fut.done()]
                        
                if self.shutdown_finished_clusters:
                    cluster_status = [future.done() for future in self.all_futures[i-1]]
                    print(f"STATUS OF CLUSTER: {i-1} | {self.executors[i-1].runner_config['type']} | {cluster_status}")
                    if all(cluster_status) and not i-1 in self.keep_alive:
                        print(f'CLOSING WORKERS FOR EXECUTOR: {i-1}')
                        self.executors[i-1].clean()
                            
        # write the results for the last set of futures.
        base_enchanted_dataset_path = os.path.join(self.base_run_dir, f'enchanted_dataset.csv')
        futures_check = {fut.key:fut for fut in self.all_futures[-1]}
        done = [fut for fut in futures_check.values() if fut.done()]
        while futures_check:
            for j, future in enumerate(done):
                result, error_info = self.get_result(future, timeout=5)
                if not result:
                    print('NO RESULT FOUND SKIPPING FOR NOW')
                    continue
                
                
                
                self.update_completion_stats(result,len(self.executors)-1)
                # if self.do_dynamic_scale_down:
                #     self.dynamic_scale_down(exe_i=len(self.executors)-1, batch_num=batch_num)
                futures_check.pop(future.key)
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
            done = [fut for fut in futures_check.values() if fut.done()]
            
        if self.shutdown_finished_clusters:
            cluster_status = [future.done() for future in self.all_futures[-1]]
            print(f"STATUS OF CLUSTER: {len(self.executors)-1} | {executor.runner_config['type']} | {cluster_status}")
            if all(cluster_status) and not len(self.executors) in self.keep_alive:
                print('CLOSING WORKERS FOR EXECUTOR:', len(self.executors)-1)
                self.executors[-1].clean()
        
        print_stats_table(self.log_stats)
                            
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
    
    # def dynamic_scale_down(self, exe_i, batch_num, logger=None):
    #     total_runs_todo = self.sampler.depth_num_runs[batch_num][exe_i]
    #     runs_left = total_runs_todo - self.completed[exe_i]
    #     num_alive_workers = self.executors[exe_i].count_alive_workers()
    #     print(f'debug EXE_I: {exe_i} | runs left:',runs_left, 'total_runs_todo', total_runs_todo, 'self.completed[exe_i]', self.completed[exe_i], 'num_alive_workers', num_alive_workers)
    #     if runs_left < num_alive_workers:
    #         print(f'DYNAMIC SCALE DOWN - YES - | RUNNER: {self.runner_types[exe_i]} | NESTED DEPTH: {exe_i} | ALIVE WORKERS: {num_alive_workers} | RUNS LEFT: {runs_left}')
    #         # self.executors[exe_i].scale(num_workers=runs_left)
    #         self.executors[exe_i].send_retire_task(n_workers=num_alive_workers-runs_left)
    #         if logger:
    #             logger.info(f"triggered dynamic_scale_down for scale-down-{self.runner_types[exe_i]}-nest_depth-{exe_i}-batch_num-{batch_num}")
                
    # def start_dynamic_scale_down_background(self, exe_i, batch_num, daemon=True):
    #     # print('debug start dynamic scale down')
    #     logger = logging.getLogger(__name__)

    #     def _loop():
    #         # print('debug start dynamic scale down: in loop')
    #         try:
    #             # print('debug stop event, scheduler alive', not self.stop_dynamic_scale_events[batch_num][exe_i].is_set(), self.executors[exe_i].scheduler_is_alive())
    #             while not self.stop_dynamic_scale_events[batch_num][exe_i].is_set():
    #                 # print('debug doing dynamic scale down background')
    #                 try:
    #                     alive = self.executors[exe_i].count_alive_workers()
    #                     target = self.executors[exe_i].scale_n_jobs
    #                 except Exception as e:
    #                     logger.exception("failed to query executor state \n %s", e)
    #                     time.sleep(2)
    #                     continue
    #                 print(f'debug nest_depth-{exe_i} | alive:', alive, 'target:', target, 'is alive',self.executors[exe_i].scheduler_is_alive())
    #                 logger.info(f'{batch_num}{exe_i} ALIVE: {alive} | TARGET: {target}')
    #                 if alive == target and self.executors[exe_i].scheduler_is_alive():
    #                     logger.info(f'{batch_num}{exe_i} TARGET MET - number of alive workers is equal to the intended target. target: {target}, alive: {alive}')
    #                     try:
    #                         self.dynamic_scale_down(exe_i, batch_num, logger=logger)
    #                     except Exception as e:
    #                         logger.exception(f"{batch_num}{exe_i} dynamic_scale_down failed \n {e}")
    #                     # short pause after scaling to avoid immediate churn
    #                     time.sleep(1)
    #                 else:
    #                     # backoff when no action is required
    #                     time.sleep(1)
    #         except Exception as e:
    #             logger.exception(f"{batch_num}{exe_i} background scale-down loop exited with error \n {e}")
    #         finally:
    #             logger.info(f"background scale-down loop exiting for scale-down-{self.runner_types[exe_i]}-nest_depth-{exe_i}-batch_num-{batch_num}")

    #     thread = threading.Thread(target=_loop, daemon=daemon, name=f"scale-down-{self.runner_types[exe_i]}-nest_depth-{exe_i}-batch_num-{batch_num}")
    #     thread.start()
    #     return thread
    
    def seperate_futures(self, futures):
        seen = set()
        futures = [f for f in futures if f.key not in seen and not seen.add(f.key)]
        done = [future for future in futures if future.done()]
        pending = [future for future in futures if not future.done()]
        return done, pending
    
    def get_result(self, future, timeout=60, silent=False):
        result = None
        try:
            result = future.result(timeout=timeout)
            self.log_stats['fut_res_available'] += 1
        except:
            pass
            
        if not result:
            started = time.time()
            while not result and time.time()-started < timeout:
                run_dir = self.all_fut_to_rundir.get(future.key)
                if run_dir:
                    result_dir = os.path.join(run_dir, 'enchanted_datapoint.csv')
                    if os.path.exists(result_dir):
                        try:
                            df = pd.read_csv(result_dir)
                            result = df.iloc[0].to_dict()
                            self.log_stats['fut_res_not_available'] += 1
                        except pd.errors.EmptyDataError as e:
                            print('EMPTY DATA ERROR: WAITING LONGER\n',e)
                    else:
                        if not silent: print('ENCHANTED DATA POINT FILE DOES NOT EXIST, FUTURE NOT FINISHED:', result_dir)
                else:
                    raise RuntimeError(f'THIS FUTURE HAS BEEN SUBMITTED BUT THE RUN_DIR WAS NOT ADDED TO fut_to_rundir, future.key: {future.key}')
                if not result:
                    if timeout == 0:
                        break
                    if not silent: print('SLEEPING TO SEE IF THE RESULT WILL BECOME AVAILABLE, sec passed:', time.time()-started)
                    time.sleep(1)
        return result
    
    def update_completion_stats(self, result, exe_i, batch_num=0):
        self.completed[exe_i] += 1
        if result['success']:
            self.succeded[exe_i] += 1
        completion_stats = {
            'header': f"BATCH {batch_num} COMPLETION STATS",
            'subheader': f"{self.executors[exe_i].runner_config['type']}",
            'NESTED DEPTH': f"{exe_i}",
            'COMPLETED': f"{self.completed[exe_i]}/{self.sampler.depth_num_runs[batch_num][exe_i]}",
            'SUCCEDED': f"{self.succeded[exe_i]}/{self.sampler.depth_num_runs[batch_num][exe_i]}"
        }   
        total_completion_stats = {
            'header': f"BATCH {batch_num} COMPLETION STATS",
            'subheader': f"ENTIRE NESTED PIPELINE",
            'COMPLETED': f"{np.sum(self.completed)}/{self.sampler.total_num_runs[batch_num]}",
            'SUCCEDED': f"{np.sum(self.succeded)}/{self.sampler.total_num_runs[batch_num]}"
        }
        print_stats_table(completion_stats)
        print_stats_table(total_completion_stats)
    