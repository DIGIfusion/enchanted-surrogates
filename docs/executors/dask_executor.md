::: enchanted_surrogates.executors.dask_executor

## Clusters

### Local cluster

Can be used for running on a local machine with multiple cores. Useful for testing or small scale runs.

Arguments:

```
n_workers: 2,
threads_per_worker: 1,
memory_limit: '12GB', 
processes: 1
```

Example configuration: /configs/example_dask_local.yaml

### SLURM cluster

Arguments for the SLURM workers.

```
account: 'project_xxx', 
queue: 'medium', 
cores: 1, 
memory: '12GB', 
processes: 1, 
walltime: '00:20:00',
config_name: 'slurm', 
interface: 'ib0', 
```

Example configuration: /configs/example_dask_slurm.yaml

### Notes

Other arguments:

```
job_script_prologue: ['module load your-modules-here',], 
job_extra_directives: [
    '-o tmp_path_hm/worker_out_MishkaRunner_1/%x.%j.out', 
    '-e tmp_path_hm/worker_out_MishkaRunner_1/%x.%j.err'], 
```