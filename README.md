# enchated-surrogates

Active Learning Steps to Surrogate a Physics Code:
1. Initiate the search space
    - decide lower and upper bound of input parameters
    - select some initial points within the space
2. Run The code on new points (ask the oracle for lables)
   - The code needs to ran once for each new point to generate output
   - this can be handled with the dask slurm deployments [dask](https://jobqueue.dask.org/en/latest/examples.html#slurm-deployments)
3. Train ML Model
    - potentially reserve some validation data that could be useful for active learning step
    - training uses the inputs as data points and outputs as lables
    - this part is no different than any standard ML training algorythem
5. Active Learning step to generate new points
   - Using information from the current inputs, outputs and current model performance the most informative new data points can be determined via various means
6. Repeat steps 2-5 untill a certain perfoemance level is reached, resource limit is reached or loss converges.

We should write the framework so it is easy to swap out different **Search Space Initators** (static sparse grids, latin hypercube), **Physics Codes**, **ML Models**/ training strategies, **Active Learning Strategies** 


Database generation for a simulation consists of: 

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

## TODO

1. **TESTING** Add examples with simplified fortran code
    - `CI/CD` tests on `main` branch 
2. Restart from save using simulation dir

## Coding Style Standards
Classes defined with an uper case first letter and the rest is lower case.
variables defined with full words seperated by _ all lower case.

All variables that a user might wish to change should be defined in a config file





