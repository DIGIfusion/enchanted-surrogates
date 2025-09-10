---
layout: default
title: HELENA
nav_order: 4
parent: Plugins
---

# HELENA plugin

Mandatory fields in the configuration file for using the HELENA plugin:

```yaml
runner:
  type: HelenaRunner
  executable_path: "/path/to/helena/executable"
  other_params: {
    "namelist_path": "/path/to/fort.10",
    "only_generate_files": True,
    "input_parameter_type": 3, 
  }  
```


## Input parameter types

The HELENA plugin supports multiple different input parameter settings, which determine how the input files are generated based on the sample parameters. The supported types are:

- Type 1: Standard input parameters. The sample parameters directly correspond to the input parameters in the HELENA namelist.
TODO: Add more.


## Example configuration

