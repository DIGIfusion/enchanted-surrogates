# enchated-surrogates

A framework for creating databases for surrogate models of complex physics codes.

## Database generation for a simulation consists of: 

1. Running the code
    - Every code has it's own runtime entry points (e.g., I/O, actual execution)
2. On a search space
    - e.g., hypercube, or efficiently searching across a space with active learinng

i.e., step 1. is repeated many times to fill volume spanned by 2. 

Idea is to abstract away the iterative process, and just uniquely handle 1. for each individual code, while being able to use mutliple searches types. 

In the simplest case, the iterative process could be automated via `batch` job submission on `SLURM`, but this doesn't scale well if we use different HPC systems. So, idea is to use [dask](https://jobqueue.dask.org/en/latest/examples.html#slurm-deployments). 

## Practicalities

The main entry point is `run.py`, which you will run via shell for HPC or local (see `SLURMrun.bash`). `run.py` takes one argument, a path to a config file. In the config, the following items are specified: 

- Code to be executed in terms of the `Runner` and `Parser` classes and associated arguments 
- Method of sampling points, or `Sampler` 
- How multiple/parallel code running is executed, or `Executor`

The `Executor` is robust to any combination of `Runner`, `Parser` and `Sampler`, i.e., things like changing directories are all handled in the code specific `Runner`. In this way, the `Executor` relies on the other classes via their `abstractmethod`s defined in the respective metaclasses, `base.py`. For example, the `Executor` relies on the `Runner`'s `single_code_run()` method.  

Because of this, most variables that a user might wish to change, such as code to run, active learning sampler, parameters to vary, etc., are all defined in the config file, while the parameter that are simulation specific are handled by the derived meta-classes.  

See `configs/tglf_config.yaml` for example configuration file. That which is commented `HPC`, `CODE`, `USER` refers to wether the config parameter is modified based on the HPC system used (e.g., `Mahti`), the simulation used (e.g., `TGLF`) or user, (e.g., specific directories).

After modifying `SLURMrun.bash` to point to a specific configuration file, e.g, `tglf_config.yaml`, run with your slurm or local shell, e.g., `sbatch SLURMrun.bash`. This will dump `*.out` files for each worker specified in the config, while also dumping `run.out` for the nanny.  

## What we do not plan to handle

1. Compiling code
2. Training downstream surrogate (outside of active learning)

## Contribution guidlines

- `main` branch is for stable code
- `develop/{feature}` or `develop/{user}` for changes, but try to keep `main` up to date and minimize lifetime of branches
- For longer term items to be integrated, e.g., Active Learning, suggest to use `Issues` followed by a branch. 


## Coding Style Standards

The coding standard [PEP8](https://peps.python.org/pep-0008/) should be used. 
Although we will likely lint with `flake` so don't worry too much about it. 


## Testing

The `tests` folder contains unit tests. These can be run manually by using the command:

    python -m pytest tests/test_*.py -v -s

and will also be automatically run by Github Actions at certain pushes.
A Github Actions workflow is also used for running Pylint tests. These are currently only testing for issues categorized as Errors or Fatal. Message overview [here](https://pylint.pycqa.org/en/latest/user_guide/messages/messages_overview.html).
To check the linting locally and get a full overview of all possible issues, run:

    pylint src/runners/SIMPLErunner.py 

where the argument is the path to the file you want to check.

