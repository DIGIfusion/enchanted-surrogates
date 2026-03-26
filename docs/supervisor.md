---
layout: default
title: Supervisor
nav_order: 2
has_children: false
---

# Supervisor

Supervisor is where the main loop of Enchanted Surrogates is ran. 
Supervisor orchestrates use of samplers, executors and runners. See the chart below
for overall structure of the code.

![Workflow chart](./img/enchanted-logic.png)


## Configuration of supervisor

Supervisor needs `base_run_dir` defined in the configuration file.
Example as follows:

```yaml
supervisor:
    base_run_dir: "path/to/folder"
    run_order:
    -   executor: ...
        sampler: ...
        runner: ...
```

## Multi-runner sequential execution

Alongside nested executing, the supervisor also supports sequential sampling.
In sequential sampling, the sampler's batches are called once but the samples
go through multiple runners which pass information to each other. This is 
useful for active learning use cases. Sequential sampling and nested sampling
can be used together in the same configuration. 

To utilize sequential sampling in configurations, multiple runners need to be
defined in the config file as a list. The same applies to executors.
The amount of executors and runners defined in run_order must be equal. If this is not met, an exception
is thrown. Examples of sequential sampling are provided within the configs directory
under [example_sequential.yaml](../configs/example.sequential.yaml). 

Example of how a run_order can be defined to perform sequential sampling:
```yaml
supervisor:
  base_run_dir: "data_dir/sequential_local"
  run_order:
  - sampler: code1_sampler
    executor:
      - code1_executor
      - code2_executor
      - code3_executor
    runner:
      - code1_runner
      - code2_runner
      - code3_runner
```


### Optional attributes
Also, it is possible to define that enchanted_dataset summary file combining 
all run results is parquet instead of csv. Csv is default and does not
require any configuration.

```yaml
supervisor:
    summary_datatype: "parquet" # csv by default
```

Hdf5 storage file is not saved if type for it is None. It is created
in every other case. 

```yaml
storage:
    type: "hdf5" # or "None"
```

It is possible to delete unnecessary files from base_run_dir and keep only wanted files.
By default all is saved. Option custom saves only described files. None does not save 
any files. 

**Note: `enchanted_dataset.csv` and `runs.h5` are saved by default.**

```yaml
supervisor:
  save_files: "all" # or "custom" or "none"
  # if using custom, only described files are saved
  save_files_list:
    - file.txt
    - file2.txt 

```


See config folder for a config file examples.
