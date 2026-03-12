---
layout: default
title: MISHKA
nav_order: 5
parent: Plugins
---

# MISHKA plugin

Mandatory fields in the configuration file for using the MISHKA plugin:

```yaml
runner:
  type: MishkaRunner
  executable_path: "/path/to/mishka/executable"
  other_params:
    namelist_path: "/path/to/fort.10"
 
```

### runner.executable_path
Path to the MISHKA executable.

### runner.other_params.namelist_path
Path to the MISHKA namelist file (fort.10) that is used as base for generating input files.

### runner.other_params.input_fort12
Path to the HELENA output file (fort.12) needed by MISHKA. If this is not provided, the plugin will look for a `helena_dir` parameter in the sampler parameters (see below).

## Parameters

### ntor
The toroidal mode number to be used in the MISHKA run. This is a mandatory parameter.

```yaml
sampler:
  type: ArraySampler
  bounds: [
    [5,7,10,15,20,30,50],]
  num_samples: [7]
  parameters: ['ntor']
```

### (optional) helena_dir
Path to the directory containing the HELENA output files (fort.12) needed by MISHKA.

```yaml
sampler:
  type: ArraySampler
  bounds: [
    [5,7,10,15,20,30,50],
    ['path/to/helena/run1', 'path/to/helena/run2', 'path/to/helena/run3'],]
  num_samples: [7, 3]
  parameters: ['ntor', 'helena_dir']
```

::: enchanted_plugin_mishka
    options:
          show_source: false
          show_submodules: true
          members: true
          show_root_heading: true