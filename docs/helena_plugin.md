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


