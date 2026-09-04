# Config Explained

Enchanted Surrogates is driven entirely by a single YAML config file, passed
to `run.py` with `-cf`/`--config_file`. This page walks through every
top-level section of that file. See
[example_local.yaml](../configs/example_local.yaml) for a minimal working
example, and the `configs/` folder for more.

```yaml
logging: INFO

executors: ...
samplers: ...
runners: ...
supervisor: ...
storage: ...
post_processing: ...
```

## `logging`

Sets the log level for the run, e.g. `INFO` or `DEBUG`. Optional, defaults to
`INFO`.

```yaml
logging: INFO
```

## `executors`

Defines one or more executors, keyed by a name you choose (e.g. `e1`). An
executor determines *where* and *how* runner code actually executes -
locally, via joblib, or on an HPC cluster through Dask.

```yaml
executors:
  e1:
    type: LocalExecutor
```

For cluster execution with `DaskExecutor`, a `SLURMcluster_config` block
configures the underlying SLURM job (account, queue, cores, memory,
walltime, etc.):

```yaml
executors:
  e1:
    type: DaskExecutor
    SLURMcluster_config:
      account: "project_xxxxxx"
      queue: "small"
      cores: 1
      memory: "1GB"
      processes: 1
      walltime: "00:05:00"
      interface: "ib0"
      job_name: enchanted-surrogates-example
    scale_n_jobs: 1
```

See [Executors](executors/local_executor.md) for the full list of executor
types and their options.

## `samplers`

Defines one or more samplers, keyed by name (e.g. `s1`). A sampler decides
*which points* in parameter space to evaluate, given `bounds`, `budget`, and
`parameters`.

```yaml
samplers:
  s1:
    type: RandomSampler
    bounds: [[0.0, 1.0], [0.0, 10.0]]
    budget: 10
    parameters: ['x', 'y']
```

Different sampler types (`GridSampler`, `BayesianOptimizationSampler`,
`LatinHypercubeSampler`, `SobolSequenceSampler`, `ActiveLearningSampler`,
`NestedSampler`, ...) accept additional, type-specific options. See
[Samplers](samplers/random_sampler.md) for details on each.

## `runners`

Defines one or more runners, keyed by name (e.g. `r1`). A runner wraps the
actual simulation or model code that's evaluated at each sampled point.

```yaml
runners:
  r1:
    type: SyntheticDoubleGaussianRunner
    dimensions: 2
    other_params: {}
```

See [Plugins](plugins/index.md) and the runner reference docs (e.g.
[Example Runner](runners/example_runner.md)) for runner-specific options.

## `supervisor`

Ties everything together: which executor/sampler/runner combination(s) run,
in what order, where output is stored, and how to handle resuming/extending
runs. `base_run_dir` and `run_order` are required.

```yaml
supervisor:
  base_run_dir: "data_dir/local"
  run_mode: "fresh" # "resume" / "extend"
  save_files: "all" # or "custom" or "none"
  save_files_list:
    - enchanted_dataset.csv
    - example_local.csv
  run_order:
    - executor: e1
      sampler: s1
      runner: r1
```

The supervisor also supports nested and sequential multi-runner execution,
resuming/extending previous runs, HPC local storage, and configurable output
retention. See [Supervisor](supervisor.md) for the full breakdown of these
options.

## `storage`

Configures the HDF5 results storage backend.

```yaml
storage:
  type: hdf5 # or "None" to disable
```

## `post_processing`

Optional. Runs a script once the supervisor's `run_order` completes, useful
for analysis or cleanup steps that depend on the full set of results being
available. The script is run from `base_run_dir`, and only runs if the main
run completes without an unhandled exception.

```yaml
post_processing:
  script_path: "path/to/script.sh" # must be executable, run from base_run_dir
```

If the script exits with a non-zero status, the failure is logged
(`log.error`) but does not fail the overall enchanted-surrogates run. Omit
`post_processing` entirely to skip this step.

See [example_post_processing.yaml](../configs/example_post_processing.yaml)
for a full working example.
