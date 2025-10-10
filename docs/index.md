---
layout: default
title: Start
nav_order: 1
has_children: true
---

# Enchanted surrogates

## How to install

Make sure you have a clean virtual environment with Python 3.10 or higher.
Then clone the repository and install the package with pip:

```bash
git clone https://github.com/DIGIfusion/enchanted-surrogates.git
pip install -e enchanted-surrogates/
```

This will install the core package and its dependencies.
In addition, you can install any plugins you want to use, by cloning their repositories and installing them with pip as well.


## How to run

After installing the package and any desired plugins, you can use the command line interface to run simulations.
For example, to run a simulation with the example runner and parser, you can use the following command:

```bash
python enchanted-surrogates/src/run.py -cf path/to/config/file
```

Make sure to replace `path/to/config/file` with the actual path to your configuration file.




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


## Other

Github repository: https://github.com/DIGIfusion/enchanted-surrogates


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
