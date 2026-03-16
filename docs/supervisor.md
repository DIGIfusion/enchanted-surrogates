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
