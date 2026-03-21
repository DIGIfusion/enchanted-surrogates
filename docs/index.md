# Enchanted surrogates

<span class="hero-subtitle">
A framework for creating databases for surrogate models of complex physics codes.
</span>
    
[View it on GitHub](https://github.com/DIGIfusion/enchanted-surrogates){ .md-button .md-button--primary }

---

!!! note
    This documentation is under development.

Machine learning surrogate model development requires large amounts of data, which is often generated using complex and computationally expensive simulation codes. The `enchanted-surrogates` package provides a flexible framework for creating databases for surrogate models of such complex physics codes.
Database generation for a simulation consists of: 

1. Running the code
    - Every code has it's own runtime entry points (e.g., I/O, actual execution) and computational resource needs
2. On a search space
    - e.g., hypercube, or efficiently searching across a space with active learing

i.e., step 1. is repeated many times to fill volume spanned by 2. 

The idea is to abstract away the iterative process, and just uniquely handle 1. for each individual code, while being able to use mutliple searches types. 


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

Optional dependencies can be installed by listing them inside square brackets, comma-separated without spaces, e.g.:

```bash
pip install -e enchanted-surrogates[bo,GPy]
```

Note: in some environments, the command python may still point to system-wide Python (e.g. `/usr/bin/python` or `/Library/Frameworks/...`) rather than the virtual environment. You can check which python is active with: 

```bash
which python
which python3
```

If neither is referring to created virtual environment, it can be referred with `.venv/bin/python` instead of `python` in the example below. 

## How to run

After installing the package and any desired plugins, you can use the command line interface to run simulations.
For example, to run a simulation with the example runner and parser, you can use the following command:

```bash
python enchanted-surrogates/src/run.py -cf path/to/config/file
```

Make sure to replace `path/to/config/file` with the actual path to your configuration file. 
The configuration file should be in YAML format and specify the runner, sampler, executor, supervisor and other parameters needed for the simulation. 
  
```yaml
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
    -   executor: e1
        sampler: s1
        runner: r1
```

### Quick start example

The following command runs the example local executor with the example configuration file. It creates a run directory in the current working directory, where it generates random samples and runs the example code.

```bash
python enchanted-surrogates/src/run.py -cf enchanted-surrogates/configs/example_local.yaml
```


## Code structure

The Supervisor is the entry point. The Supervisor reads parameters and has Sampler, Executor and Runner. The Executor is chosen based on the system where the code is running. It sends samples for execution. A Runner is a code-specific module for running the code in question. Commonly paired with a code-specific parser. A Parser is a code-specific module for reading and writing files produced by the code. Code-specific Runners and Parsers are developed as plugins.

The Supervisor initializes a Sampler and fetches samples from it. 
The Supervisor gives samples to the Executor. The Executor initializes some cluster or job queue or similar. 
The Executor calls `simulation_task.py` which initializes a Runner and creates files. The Supervisor creates 
data structures from the files created to specified running directory.

See documentation for [Supervisor](supervisor.md) for a graph about module structure.


## Contribution guidlines

We encourage contributions to the enchanted-surrogates package! Here are some guidelines to help you get started. If you are interested in adding a new code plugin, please refer to the [Plugins](./plugins/index.md) section for more details.

- `main` branch is for stable code (releases)
- `develop` branch is for latest development code (merges from feature branches)
- `develop/{feature}` or `develop/{user}` for changes.
- `bug/{descriptive_name}` for bug fixes.
- **One feature or fix per pull request**. This ensures that changes are isolated and easier to review. Be respectful of your fellow developers and create small, focused pull requests.
- Use pull requests to merge branches. Delete branch after merge.
- Use `Issues` for bug reports, feature requests, etc.
- For longer term items to be integrated, e.g., Active Learning, suggest to use `Issues` followed by a branch. 
- The configs folder in the source is to be kept for example config files and example cases that would be benefical to the wider community. Plugin-specific config files should be kept in the plugin repository.

### Coding Style Standards

The coding standard [PEP8](https://peps.python.org/pep-0008/) should be used. 
Although we will likely lint with `flake` so don't worry too much about it. 


### Testing
#### Automated Testing at Pull Requests
The `tests` folder contains unit tests. These can be run manually by using the command:

    pytest tests -v -s

and will also be automatically run by Github Actions at certain pushes and pull requests. It is recommended to locally run the tests before making a commit.

If on HPC you must be using an interactive session with roughly 4 cores and at least 500MB of memory. 
<!-- **NB:** submodules are necessary to run the tests.  -->

#### Workflow tests
The `workflow_tests` folder contains larger workflow tests. These can be run manually by using the command:

    pytest workflow_tests/automated_tests_no_HPC -v -s


#### Linting Tests
A Github Actions workflow is also used for running Pylint tests. These are currently only testing for issues categorized as Errors or Fatal. Message overview [here](https://pylint.pycqa.org/en/latest/user_guide/messages/messages_overview.html).
To check the linting locally and get a full overview of all possible issues, run:  

For single file check:  

    pylint /path/to/file.py --disable=R,C,W,E0401

For all python files in $PWD:  

    pylint $(find $PWD -name "*.py") --disable=R,C,W,E0401


#### Machine Specific Tests
For now, no HPC specific tests are run as part of the automated testsing procedure. So if you use enchanted surrogates on a specific machine it is your responsibility to test updates on that machine. 

Alternatively, one may create a tests folder for a specific machine in  `/enchanted-surrogates/tests/MACHINE_NAME_tests`, which should be executable via

    python -m pytest tests/MACHINE_NAME_tests

### Documentation
#### Main documentation
This project supports automatic documentation generation from docstrings of classes and modules, while still allowing manual documentation via Markdown (.md) files in the `docs/` directory.
There are three main approaches to documenting modules or classes:
1. Docstring-only documentation
    You can rely entirely on docstrings to generate documentation. There is no need to create a .md file if all documentation is provided via docstrings.
    Write a docstring in the module or class.
    Important: the docstring must be placed at the very top of the file, before any imports, see other samplers and executors as example.
    The documentation will be automatically extracted and included in the generated site.
2. Docstring + Markdown file
    You can combine docstrings with a manually written Markdown file.
    Create a .md file in the docs directory with a name similar to the module or class and write additional documentation in the .md file.
    You can still use docstrings for part of the documentation.
    Important: keep the same rule — include a docstring at the top of the file (before imports).
    Docstrings will be automatically imported and rendered within the corresponding documentation page.
    Ensure that your .md file and module/class file have same or almost same name.
3. Markdown-only documentation
    You can rely exclusively on manually written Markdown documentation.
    Create a .md file in the docs directory and write all documentation there.
    If you do not want docstrings to be imported:
       Either do not write a docstring at all, or
       Do not include a docstring at the top of the file (before imports). In this case, the documentation generator will skip importing the module/class.

#### Plugin documentation
Documentation for plugins can also be imported from docstrings contained within plugin modules and classes. The same three documentation approaches described above can be applied to plugins as well.
For more detailed information on importing documentation from plugins, please refer to the [Plugins](https://digifusion.github.io/enchanted-surrogates/plugins/#development-steps) section.


## About the project

### License

Enchanted surrogates is distributed by an MIT license.


### Citation

If you use this package in your research, please cite:

```bibtex
@Misc{enchanted-surrogates,
  title =        {Enchanted Surrogates: A flexible framework for surrogate modelling of fusion plasma simulations},
  author =       {A. Kit, A.M. Bruncrona, D. Jordan, A.E. Järvinen},
  howpublished = {Github},
  year =         {2025},
  url =          {https://github.com/DIGIfusion/enchanted-surrogates}
}
```


### Contributors

**Feel free to contribute!**

<a href = "https://github.com/Tanu-N-Prabhu/Python/graphs/contributors">
  <img src = "https://contrib.rocks/image?repo = DIGIfusion/enchanted-surrogates"/></a>


### Acknowledgements
The development of this framework has been support by multiple funding sources:

- Research Council of Finland project numbers: 355460, 358941.

- EUROfusion Consortium, funded by the European Union via the Euratom Research and Training Programme (Grant Agreement No 1010522200 - EUROfusion) through the Advanced Computing Hub framework of the E-TASC program as well as dedicated machine learning projects, such as the project focused on surrogating pedestal MHD stability models.
  
- Multiple CSC IT Center for Science projects have provided the necessary computing resources for the development and application of the framework. 
