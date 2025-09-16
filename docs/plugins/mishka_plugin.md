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
  other_params: {
    "namelist_path": "/path/to/fort.10"
  }  
```


