---
layout: default
title: Supervisor
nav_order: 2
has_children: false
---

# Supervisor

Supervisor is where the main loop of Enchanted Surrogates is ran. Supervisor
orchestrates use of samplers, executors and runners. See the chart below for
overall structure of the code.

![Workflow chart](./img/enchanted-logic.png)

## Configuration of supervisor

Supervisor needs `base_run_dir` defined in the configuration file. Example as
follows:

```yaml
supervisor:
    base_run_dir: "path/to/folder"
    run_order:
        - executor: ...
          sampler: ...
          runner: ...
```

### Nested execution

TODO

### Resuming/extending previous runs

The supervisor supports seamlessly resuming a previous run, in case of crashes
or timeouts. The previous run can also be extended with more sample points if
desired. This is configured with the `run_mode` config option.

```yaml
supervisor:
    base_run_dir: ...
    run_mode: "fresh" # "resume" / "extend"
    run_order:
        - executor: ...
          sampler: ...
          runner: ...
```

#### "fresh"

Normal run and the default

#### "resume"

- Resume an interrupted previous run:
  - Set `run_mode: resume` under `Supervisor` in the config file, no other
    changes are needed.
  - Supervisor keeps track of the run state (how many batches sampled, which
    nesting depth, and how many samples have been submitted) and using this data
    restores the run from where it left off and continues from there.

- Resume a previous run that was completed but increase the budget:
  - Set run mode to resume and also increase the budget of the sampler. From
    Supervisor point of view, this is same as if the budget always was that much
    and the run was just interrupted.
  - Eg. increasing budget from 50 to 60 and re-running generates 10 new samples.

#### "extend"

- Extend a previous run:
  - Set `run_mode: extend` under `Supervisor` in the config file
  - Now setting sampler budget to 10 means that 10 new samples are created.

### HPC cluster local storage

Some partitions on some HPC clusters have access to local memory on the run
node. Setting the `local_storage` environment variable appropriately could
potentially improve I/O operation performance.

```yaml
supervisor:
    local_storage: TMPDIR
```

### Optional attributes

Also, it is possible to define that enchanted_dataset summary file combining all
run results is parquet instead of csv. CSV is default and does not require any
configuration.

```yaml
supervisor:
    summary_datatype: "parquet" # csv by default
```

Hdf5 storage file is not saved if type for it is None. It is created in every
other case.

```yaml
storage:
    type: "hdf5" # or "None"
```

It is possible to delete unnecessary files from base_run_dir and keep only
wanted files. By default all is saved. Option custom saves only described files.
None does not save any files.

**Note: `enchanted_dataset.csv` and `runs.h5` are saved by default.**

```yaml
supervisor:
    save_files: "all" # or "custom" or "none"
    # if using custom, only described files are saved
    save_files_list:
        - file.txt
        - file2.txt
```

See config folder for example configurations.
