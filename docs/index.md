# Enchanted surrogates

<span class="hero-subtitle">
A framework for creating databases for surrogate models of complex physics codes.
</span>


[View it on GitHub](https://github.com/DIGIfusion/enchanted-surrogates){.md-button .md-button--primary}


---

!!! notes
    This documentation is under development.

Machine learning surrogate model development requires large amounts of data,
which is often generated using complex and computationally expensive simulation
codes. The `enchanted-surrogates` package provides a flexible framework for
creating databases for surrogate models of such complex physics codes. Database
generation for a simulation consists of:

1. Running the code:
    - Every code has it's own runtime entry points (e.g., I/O, actual execution)
      and computational resource needs
2. On a search space:
    - e.g., hypercube, or efficiently searching across a space with active
      learning

i.e., step 1. is repeated many times to fill volume spanned by 2.

The idea is to abstract away the iterative process, and just uniquely handle 1.
for each individual code, while being able to use mutliple searches types.

## How to install

Make sure you have a clean virtual environment with Python 3.10 or higher.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

Then clone the repository and install the package with pip:

```bash
git clone https://github.com/DIGIfusion/enchanted-surrogates.git
pip install -e enchanted-surrogates/
```

This will install the core package and its dependencies. In addition, you can
install any plugins you want to use, by cloning their repositories and
installing them with pip as well. See the [Plugins](./plugins/index.md) section
for more details.

Please note that some samplers require optional dependencies.
Check the sampler's documentation to see if any optional dependencies are required to run. 
Optional dependencies can be installed by listing them inside square brackets,
comma-separated without spaces, e.g.:

```bash
pip install -e enchanted-surrogates[bo,GPy,activelearning]
```

Note: in some environments, the command python may still point to system-wide
Python (e.g. `/usr/bin/python` or `/Library/Frameworks/...`) rather than the
virtual environment. You can check which python is active with:

```bash
which python
which python3
```

If neither is referring to created virtual environment, it can be referred with
`.venv/bin/python` instead of `python` in the example below.

## How to run

After installing the package and any desired plugins, you can use the command
line interface to run simulations. For example, to run a simulation with the
example runner and parser, you can use the following command:

```bash
python enchanted-surrogates/src/run.py -cf path/to/config/file
```

Make sure to replace `path/to/config/file` with the actual path to your
configuration file. The configuration file should be in YAML format and specify
the runners, samplers, executors, supervisor and other parameters needed for the
simulation.

The configuration file should list all the executors, samplers and runners to be
used. The supervisor `run_order` should then be specified for the desired
workflow. For a simple, non-nested workflow, `run_order` contains only one
executor, sampler and runner. For nested workflows, see
[Nested execution](supervisor.md#nested-execution) for more information.

```yaml
logging: # NOTSET, DEBUG (default), INFO, WARNING, ERROR, CRITICAL

executors:
  e1:
    type: ...
samplers:
  s1:
    type: ...
runners:
  r1:
    type: ...
supervisor:
  base_run_dir: ...
  run_order:
    - executor: e1
      sampler: s1
      runner: r1
```

### Output

```
base_run_dir/
├── data/
│    └── ... 
├── logs/
│    └── ... 
├── config/
│    └── ... 
├── enchanted_datapoints.csv
└── runs.h5
```

#### summary files

The summary files are structured as such: 

|   | param1 | param2 | paramN | output | success | run_dir                           |
|---|--------|--------|--------|--------|---------|-----------------------------------|
| 0 | 0.1    | 0.2    | 0.3    | 0.6    | true    | data_dir/example/data/d0_b0_r0_s0 |
| 1 | 0.1    | 0.2    | 0.3    | 0.6    | true    | data_dir/example/data/d0_b0_r1_s0 |
| N | 0.1    | 0.2    | 0.3    | 0.6    | true    | data_dir/example/data/d0_bn_rn_s0 |


All user defined sampled parameters are included for each sample.
The runner output is defined as output. There is also a success field which is a boolean.
Run directories are also included for clarity. 

#### data

Contains the enchanted_datapoint.csv files and runner outputs.

#### logs

Contains all log files from the supervisor and workers.

#### config

Contains a copy of the configuration file.

**Note: Output files to be saved can be configured, see
[Configuring output files](supervisor.md#configuring-output-files).**

### Quick start example

The following command runs the example local executor with the example
configuration file. It creates a run directory in the current working directory,
where it generates random samples and runs the example code.

```bash
python enchanted-surrogates/src/run.py -cf enchanted-surrogates/configs/example_local.yaml
```

## Code structure

The Supervisor is the entry point. The Supervisor reads parameters and has
Sampler, Executor and Runner. The Executor is chosen based on the system where
the code is running. It sends samples for execution. A Runner is a code-specific
module for running the code in question. Commonly paired with a code-specific
parser. A Parser is a code-specific module for reading and writing files
produced by the code. Code-specific Runners and Parsers are developed as
plugins.

The Supervisor initializes a Sampler and fetches samples from it. The Supervisor
gives samples to the Executor. The Executor initializes some cluster or job
queue or similar. The Executor calls `simulation_task.py` which initializes a
Runner and creates files. The Supervisor creates data structures from the files
created to specified running directory.

See documentation for [Supervisor](supervisor.md) for a graph about module
structure.

## About the project

### License

Enchanted surrogates is distributed by an MIT license.

### Citation

If you use this package in your research, please cite:

```bibtex
@Misc{enchanted-surrogates,
  title =        {Enchanted Surrogates: A flexible framework for surrogate modelling of fusion plasma simulations},
  author =       {Adam Kit and Amanda Bruncrona and Daniel Jordan and Aaro Järvinen and Anna Niemelä},
  howpublished = {Github},
  year =         {2025},
  url =          {https://github.com/DIGIfusion/enchanted-surrogates}
}
```


### Acknowledgements

The development of this framework has been support by multiple funding sources:

- Research Council of Finland project numbers: 355460, 358941.

- EUROfusion Consortium, funded by the European Union via the Euratom Research
  and Training Programme (Grant Agreement No 1010522200 - EUROfusion) through
  the Advanced Computing Hub framework of the E-TASC program as well as
  dedicated machine learning projects, such as the project focused on
  surrogating pedestal MHD stability models.

- Multiple CSC IT Center for Science projects have provided the necessary
  computing resources for the development and application of the framework.
