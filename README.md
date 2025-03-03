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

After modifying `SLURMrun.bash` to point to a specific configuration file, e.g, `tglf_config.yaml`, run with your slurm or local shell, e.g., `sbatch SLURMrun.bash`. This will dump `*.out` files for each worker specified in the config, while also dumping `run.out` for the nanny. The `SLURMrun.batch` creates a nanny, thus should be executed on an interactive node, not a compute node. The `run.py` nanny sets up `workers` (or equivilantly `jobs` as per below).

Some extra notes on the config file: 

- `n_jobs` will modify the number of `sbatch` commands sent, each of which is a `worker` (worker resources defined in the config under `worker_args`)
 

#### Submodules 

If one plans on doing active learning, the submodule [`bmdal`](https://github.com/BlackHC/2302.08981/tree/main) is necessary, and has to be added to paths as well (see example configuration file and `SLURM.bash`). 

To add submodules: `git submodule update --init --recursive` 

## What we do not plan to handle

1. Compiling code
2. Training downstream surrogate (outside of active learning)

## Contribution guidlines

- `main` branch is for stable code
- `develop/{feature}` or `develop/{user}` for changes, but try to keep `main` up to date and minimize lifetime of branches
- For longer term items to be integrated, e.g., Active Learning, suggest to use `Issues` followed by a branch. 
- The configs folder on git hub is to be kept for test config files and example cases that would be benifical to the wider community. Please keep personal config files in your local repositories either by adding them to .gitignore or keeping them in a seperate branch. 

## Coding Style Standards

The coding standard [PEP8](https://peps.python.org/pep-0008/) should be used. 
Although we will likely lint with `flake` so don't worry too much about it. 


## Testing
### Automated Testing at Pull Request
The `tests` folder contains unit tests. These can be run manually by using the command:

    python -m pytest tests -v -s

and will also be automatically run by Github Actions at certain pushes.

### Linting Tests
A Github Actions workflow is also used for running Pylint tests. These are currently only testing for issues categorized as Errors or Fatal. Message overview [here](https://pylint.pycqa.org/en/latest/user_guide/messages/messages_overview.html).
To check the linting locally and get a full overview of all possible issues, run:

    pylint src/runners/SIMPLErunner.py 

where the argument is the path to the file you want to check.

### Machine Specific Tests
For security reasons it is currently not possible for automated tests to access HPC harware. So if you use enchanted surrogates on a specific machine it is your responsibility to test updates on that machine. When submitting a pull request please suggest assignees that you believe should test the new branch on their machine before the merge. 

## Acknowledgements
The development of this framework has been support by multiple funding sources:
- Research Council of Finland project numbers: 355460, 358941.
- EUROfusion Consortium, funded by the European Union via the Euratom Research and Training Programme (Grant Agreement No 1010522200 - EUROfusion) through the Advanced Computing Hub framework of the E-TASC program as well as dedicated machine learning projects, such as the project focused on surrogating pedestal MHD stability models.
- Multiple CSC IT Center for Science projects have provided the necessary computing resources for the development and application of the framework. 
