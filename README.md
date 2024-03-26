# enchated-surrogates

## Active Learning Steps to Surrogate a Physics Code:
1. Initiate the search space, Sample_initial
    - decide lower and upper bound of input parameters
    - select some initial points within the space
    - **Dummy Example** Generate random floats between 0 and 10
2. Run The code on new points (ask the oracle for lables)
   - The code needs to ran once for each new point to generate output
   - this can be handled with the dask slurm deployments [dask](https://jobqueue.dask.org/en/latest/examples.html#slurm-deployments)
   - **Dymmy Example** Assign label 0 to numbers below 5 and 1 to labels above.
3. Train ML Model
    - potentially reserve some validation data that could be useful for active learning step
    - training uses the inputs as data points and outputs as lables
    - this part is no different than any standard ML training algorythem
    - **Dummy Example** Train 3 SKLearn classifiers to do the binary classification from data to labels, just train for a few epochs so they don't get too good, to make active step viable
5. Active Learning step to generate new points, Sample_active
   - Using information from the current inputs, outputs and current model performance the most informative new data points can be determined via various means
   - **Dummy Example** Generate more floats between 0 and 10 that the oracle hasn't labeled and hasn't been seen in training. Run an inference with the 3 SKlearn classifiers and see where they disagree to suggest new points
6. Repeat steps 2-5 untill a certain perfoemance level is reached, resource limit is reached or loss converges.

We should write the framework so it is easy to swap out different **Sample_initial** (static sparse grids, latin hypercube), **Physics Codes**, **ML Models**/ training strategies, **Sample_active** 

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
Classes defined with an uper case first letter and the rest is lower case.
variables defined with full words seperated by _ all lower case.

All variables that a user might wish to change should be defined in a config file





