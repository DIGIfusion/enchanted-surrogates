---
layout: default
title: MISHKA
nav_order: 5
parent: Plugins
---

# MISHKA plugin

MISHKA is an ideal MHD stability code. Given an equilibrium and a toroidal mode number $N_{tor}$, MISHKA solves the growth rate of that mode number.

The toridal mode number `ntor` must be defined as a sampling parameter. A path to the directory containing the HELENA output files can also optionally be defined.

Below is an example of a configuration setup. This example will run MISHKA for mode numbers 5, 7 and 10 for three equilibria defined by parameter `helena_dir`.

```yaml
runner:
  type: MishkaRunner
  executable_path: "/path/to/mishka/executable"
  other_params:
    namelist_path: "/path/to/fort.10"

sampler:
  type: ArraySampler
  bounds: [
    [5,7,10],
    ['path/to/helena/run1', 'path/to/helena/run2', 'path/to/helena/run3'],
    ]
  num_samples: [7, 3]
  parameters: ['ntor', 'helena_dir']
 
```

