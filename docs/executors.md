---
layout: default
title: Executors
nav_order: 4
has_children: false
---

# Executors

Description of different executors available in the enchanted-surrogates package and the configurations needed to use them.

## Local Executor

TODO.
Example configuration: /configs/example_local.yaml

## Joblib Executor

TODO.

## Dask Executor

Dask documentation: https://jobqueue.dask.org/en/latest/

See cluster specific documentation for more details.



Other arguments:

```
job_script_prologue: ['module load your-modules-here',], 
job_extra_directives: [
    '-o tmp_path_hm/worker_out_MishkaRunner_1/%x.%j.out', 
    '-e tmp_path_hm/worker_out_MishkaRunner_1/%x.%j.err'], 
```

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


## Nested executors

Nested executors allow for running nested sampling schemes, which is useful when one code is used to generate input for another code.

![nested](./img//nested.png)

The resulting folder structure is a nested, where the outer keys are the parameters from the outer sampler, and the inner keys are the parameters from the inner sampler. The values are the results from the runner.


### Example cases

Different executors, different runners, different samplers.
Ex. Run HELENA with random sampler and MISHKA with grid sampler, and with different executors.

Same executor, different runners, different samplers.
Run HELENA with random sampler and MISHKA with grid sampler, but use one same executor.
