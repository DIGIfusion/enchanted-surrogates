---
layout: default
title: Start
nav_order: 1
has_children: true
---

# Enchanted surrogates

A framework for creating databases for surrogate models of complex physics codes.
{: .fs-6 .fw-300 }

[View it on GitHub](https://github.com/DIGIfusion/enchanted-surrogates){: .btn .fs-5 .mb-4 .mb-md-0 }

---
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

This will install the core package and its dependencies.
In addition, you can install any plugins you want to use, by cloning their repositories and installing them with pip as well. See the [Plugins](./plugins/index.md) section for more details.


## How to run

After installing the package and any desired plugins, you can use the command line interface to run simulations.
For example, to run a simulation with the example runner and parser, you can use the following command:

```bash
python enchanted-surrogates/src/run.py -cf path/to/config/file
```

Make sure to replace `path/to/config/file` with the actual path to your configuration file. 
The configuration file should be in YAML format and specify the runner, sampler, executor, and other parameters needed for the simulation. 
  
```yaml
executor:
    type: ...
sampler: 
    type: ...
runner:
    type: ...
```



## Code structure

See specific sections for detailed intormation about the different options. 

### Samplers
Modules for generating samples according to some rule set. 

### Executors
Modules for distributing and executing the runs. 
 
### Runners
Code specifiic modules for executing the code in question. Commonly paired with a code specific parser. 

### Parsers
Code specific modules for reading and writing files produced by the code. 


The executor initalizes a sampler and fetches samples from it. The executor initialized some cluster or job queue or similar. 
The sampler generates a batch of samples. A sample is sent to the Runner. 
The Runner initalizes and uses a Parser for writing input files based on the sample parameters and postprocessing output files. 

## Quick start

TODO.



## Contribution guidlines

- `main` branch is for stable code
- `develop/{feature}` or `develop/{user}` for changes, but try to keep `main` up to date and minimize lifetime of branches
- For longer term items to be integrated, e.g., Active Learning, suggest to use `Issues` followed by a branch. 
- The configs folder in the source is to be kept for test config files and example cases that would be benefical to the wider community.  

### Coding Style Standards

The coding standard [PEP8](https://peps.python.org/pep-0008/) should be used. 
Although we will likely lint with `flake` so don't worry too much about it. 


## Testing
### Automated Testing at Pull Requests
The `tests` folder contains unit tests. These can be run manually by using the command:

    pytest tests/automated_tests_no_HPC -v -s

and will also be automatically run by Github Actions at certain pushes and pull requests. It is recommended to locally run the tests before making a commit.

If on HPC you must be using an interactive session with roughly 4 cores and at least 500MB of memory. **NB:** submodules are necessary to run the tests.


### Linting Tests
A Github Actions workflow is also used for running Pylint tests. These are currently only testing for issues categorized as Errors or Fatal. Message overview [here](https://pylint.pycqa.org/en/latest/user_guide/messages/messages_overview.html).
To check the linting locally and get a full overview of all possible issues, run:  

For single file check:  

    pylint /path/to/file.py --disable=R,C,W,E0401

For all python files in $PWD:  

    pylint $(find $PWD -name "*.py") --disable=R,C,W,E0401


### Machine Specific Tests
For now, no HPC specific tests are run as part of the automated testsing procedure. So if you use enchanted surrogates on a specific machine it is your responsibility to test updates on that machine. 

Alternatively, one may ceate a tests folder for a specific machine in  `/enchanted-surrogates/tests/MACHINE_NAME_tests`, which should be executable via

    python -m pytest tests/MACHINE_NAME_tests


## About the project

### License

Enchanted surrogates is distributed by an MIT license.


### Citation

If you use this package in your research, please cite:

```bibtex
@Misc{enchanted-surrogates,
  title =        {Enchanted Surrogates: A flexible framework for surrogate modelling of fusion plasma simulations},
  author =       {},
  howpublished = {Github},
  year =         {2025},
  url =          {https://github.com/DIGIfusion/enchanted-surrogates}
}
```

### Acknowledgements
The development of this framework has been support by multiple funding sources:
- Research Council of Finland project numbers: 355460, 358941.
- EUROfusion Consortium, funded by the European Union via the Euratom Research and Training Programme (Grant Agreement No 1010522200 - EUROfusion) through the Advanced Computing Hub framework of the E-TASC program as well as dedicated machine learning projects, such as the project focused on surrogating pedestal MHD stability models.
- Multiple CSC IT Center for Science projects have provided the necessary computing resources for the development and application of the framework. 
