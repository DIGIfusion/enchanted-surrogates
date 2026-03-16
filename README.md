# enchated-surrogates

A framework for creating databases for surrogate models of complex physics codes.

See Github Pages: https://digifusion.github.io/enchanted-surrogates/.

### Database generation for a simulation consists of: 

1. Running the code
    - Every code has it's own runtime entry points (e.g., I/O, actual execution)
2. On a search space
    - e.g., hypercube, or efficiently searching across a space with active learing

i.e., step 1. is repeated many times to fill volume spanned by 2. 

The idea is to abstract away the iterative process, and just uniquely handle 1. for each individual code, while being able to use multiple searches types.

In the simplest case, the iterative process could be automated via `batch` job submission on `SLURM`, but this doesn't scale well if we use different HPC systems. So, idea is to use [dask](https://jobqueue.dask.org/en/latest/examples.html#slurm-deployments). 

### Practicalities

The main entry point is `run.py`, which you will run via shell for HPC or local. `run.py` takes one argument, a path to a config file. In the config, the following items are specified: 

- `Supervisor` which orchestrates the run is given base running directory path and order of executor, sampler and runner.
- Code to be executed in terms of the `Runner` and `Parser` classes and associated arguments 
- Method of sampling points, or `Sampler` 
- How multiple/parallel code running is executed, or `Executor`
- Storage type (csv, parquet, hdf5)

`Supervisor` is responsible for data handling and calling the needed classes for creating database. It is robust to any combination of `Sampler`, `Executor`, `Runner` and `Parser`. `Supervisor` receives samples from `Sampler`, gives samples to `Executor` which parallelizes code running set in `Runner`. Support of different combinations is enabled by `abstractmethod`s defined in each base class, which defines clear borders of responsibilities.

Because of this, most variables that a user might wish to change, such as code to run, samplers or other modules used, nesting order, parameters to vary, etc., are all defined in the config file, while the parameter that are simulation specific are handled by the derived meta-classes.

See `configs/` folder for example configuration files. There are config files for many different cases, see definitions for arguments in [documentation](https://digifusion.github.io/enchanted-surrogates/).

### Example: running with SLURM

After modifying `SLURMrun.bash` to point to a specific configuration file, e.g, `example_dask_slurm.yaml`, run with your slurm or local shell, e.g., `sbatch SLURMrun.bash`. This will dump `*.out` files for each worker specified in the config, while also dumping `run.out` for the nanny. The `SLURMrun.batch` creates a nanny, thus should be executed on an interactive node, not a compute node. The `run.py` nanny sets up `workers` (or equivilantly `jobs` as per below).

Some extra notes on the config file: 

- `scale_n_jobs` will modify the number of `sbatch` commands sent, each of which is a `worker`.

### Seamless sampling

If the run crashes or you run out of time with HPC, or if you want to generate additional samples to a dataset, it is possible with seamless sampling. 

Configuration is under `Supervisor` like, see the `run_mode` for example in `configs/example_local.yaml`

Options:
- `fresh`: Default option, a new run.
- `resume`: Running resume means continuing with previous interrupted or completed run to the budget limit defined in configuration. `Supervisor` keeps track how many batches have been saved.
- `extend`: Option for just adding more samples, now setting sampler `budget:10` means that 10 additional samples are created.

### Nested sampling

`Supervisor` supports configuring the order of nested samplers and reuse of single type of module. For example if runner and executor are same for each nesting level but sampling is differing, it is possible to set it in `Supervisor` configuration parameter `run_order`. See `configs/example_nested.yaml` for example.

### Saving files

If running in HPC environment, the amount of files possible to save might be limited. For this issue it is possible to save only the summary file or selected files. 

Options:
- `all`: All files are saved
- `none`: Only `enchanted_dataset.csv` and `runs.h5` are saved
- `custom`: Files as a list that are saved additional to `enchanted_dataset.csv` and `runs.h5`

### Optional dependencies

There is possibility to have active learning sampler. This requires `scikit-activeml` package installed, which can be done with `pip install -e enchanted-surrogates[activelearning]`. See `configs/example_active_yaml` for configuration. Key thing is to include `query_strategy` in sampler.

#### Submodules 

Active learning with the submodule [`bmdal`](https://github.com/BlackHC/2302.08981/tree/main) is to be supported in the future.

To add submodules: 
    git submodule update --init --recursive

## What we do not plan to handle

1. Compiling code
2. Training downstream surrogate (outside of active learning)

## Acknowledgements
The development of this framework has been support by multiple funding sources:
- Research Council of Finland project numbers: 355460, 358941.
- EUROfusion Consortium, funded by the European Union via the Euratom Research and Training Programme (Grant Agreement No 1010522200 - EUROfusion) through the Advanced Computing Hub framework of the E-TASC program as well as dedicated machine learning projects, such as the project focused on surrogating pedestal MHD stability models.
- Multiple CSC IT Center for Science projects have provided the necessary computing resources for the development and application of the framework. 
