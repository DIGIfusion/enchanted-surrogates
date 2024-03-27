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

The main entry point is `run.py`, which you will run via shell for HPC or local (see `test.bash`). `run.py` takes one argument, a path to a config file, see `tglf_config.yaml` for an example for the TGLF code. In the config, the following items are specified: 

- Code to be executed in terms of the `Runner` and `Parser` classes and associated arguments 
- Method of sampling points, or `Sampler` 
- How multiple/parallel code running is executed, or `Executor`

The `Executor` is robust to any combination of `Runner`, `Parser` and `Sampler`, i.e., things like changing directories are all handled in the code specific `Runner`. In this way, the `Executor` relies on the other classes via their `abstractmethod`s defined in the respective metaclasses, `base.py`. For example, the `Executor` relies on the `Runner`'s `single_code_run()` method.  

Because of this, most variables that a user might wish to change, such as code to run, active learning sampler, parameters to vary, etc., are all defined in the config file, while the parameter that are simulation specific are handled by the derived meta-classes.  

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






