# enchated-surrogates

enchanted-surrogates is a framework for creating surrogate models of complex physics codes with active learning on HPC systems.

enchanted-surrogates is still in the initial stages of development. 

The following physics codes are being integrated:

- TGLF
- HELENA/MISHKA
- GENE

The following HPC systems are being integrated:

- MAHTI
- LUMI

Currently enchanted surrogates can be used in the following way: 

TODO: Write user guide

## Database generation for a simulation consists of: 

- 1. Running the code
    - Every code has it's own runtime entry points (e.g., I/O, actual execution)
- 2. On a search space
    - e.g., hypercube, or efficiently searching across a space with active learinng

i.e., step 1 is repeated many times to iterate over 2. 

Idea is to abstract away the iterative process, and just uniquely handle `1.` for each individual code, while being able to use mutliple searches types. 

In the simplest case, the iterative process could be automated via `batch` job submission on `SLURM`, but this doesn't scale well if we use different HPC systems. So, idea is to use [dask](https://jobqueue.dask.org/en/latest/examples.html#slurm-deployments). 

## Pre-steps 

1. Code has to be compiled before hand

## Contribution guidlines

- `main` branch is for stable code
- `develop/{feature}` for the individual improvements to be `pushed` to main
- don't use `Issues`
- Be crystal clear in each commit about what changes you have made so if we need to revert back we can.
- Don't change file names since then we loose history
- Use .gitignore for large files you don't want on the repo.

## TODO

1. **TESTING** Add examples with simplified fortran code
    - `CI/CD` tests on `main` branch 
2. Restart from save using simulation dir

## Coding Style Standards

The coding standard [PEP8](https://peps.python.org/pep-0008/) should be used. 
This includes for example:
- Use 4 space indentation
- Classes defined with an upper case first letter (CamelCase)
- Variables and functions are defined with full words seperated by _ all lower case
- Constants on module level in all capital letters

For good docstring conventions, see [PEP257](https://peps.python.org/pep-0257/).

All variables that a user might wish to change should be defined in a config file.





