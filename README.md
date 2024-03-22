# enchated-surrogates

Database generation for a simulation consists of: 

- 1. Running the code
    - Every code has it's own runtime entry points (e.g., I/O, actual execution)
- 2. On a search space
    - e.g., hypercube, or efficiently searching across a space with active learinng

i.e., step 1 is repeated many times to iteratove over 2. 

Idea is to abstract away the iterative process, and just uniquely handle `1.` for each indivudal code, while being able to use mutliple searches types. 

In the simplest case, the iterative process could be automated via `batch` job submission on `SLURM`, but this doesn't scale well if we use different HPC systems. So, idea is to use [dask](https://jobqueue.dask.org/en/latest/examples.html#slurm-deployments). 