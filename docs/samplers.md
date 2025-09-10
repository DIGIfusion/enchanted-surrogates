# Samplers

Description of different samplers available in the enchanted-surrogates package and the configurations needed to use them.

## Random sampler

The random sampler generates samples randomly within the specified bounds for each parameter.
To use the random sampler, you need to specify it in the configuration file as follows:

```yaml
sampler:
  type: RandomSampler
  parameters: ['x', 'y']
  bounds: [[1, 10], [0, 1]]
  num_samples: 100
```

where
- parameters (list of str): The names of the parameters.
- bounds (list of tuple of float): The bounds of each parameter.
- num_samples (int): The number of samples.


## Grid sampler

The grid sampler generates samples on a grid defined by the number of samples per parameter and the parameter bounds.

To use the grid sampler, you need to specify it in the configuration file as follows:

```yaml
sampler:
  type: GridSampler
  parameters: ['x', 'y']
  bounds: [[1, 10], [0, 1]]
  num_samples: [4, 3]
```
where 
- parameters (list of str): The names of the parameters.
- bounds (list of tuple of float): The bounds of each parameter.
- num_samples (list of int): The number of samples for each parameter.

In the above example, the grid sampler will generate a grid of 4 samples for parameter 'x' between 1 and 10, and 3 samples for parameter 'y' between 0 and 1, resulting in a total of 12 samples.

## Array sampler

The array sampler will generate samples by taking the Cartesian product of the provided discrete values for each parameter.

For example:
```yaml
    sampler:
        type: ArraySampler
        parameters: ['x', 'y']
        bounds: [[0, 7, 100], ['a', 'b']]
        num_samples: [3, 2]
```
where
- parameters (list of str): The names of the parameters.
- bounds (list of list of ...): The discrete values for each parameter.
- num_samples (list of int): The number of samples for each parameter. This can be left empty for the array sampler, as the number of samples is determined by the length of the bounds.
  
The example above would create the following sample space:
```python
    [[0, 'a'], [0, 'b'], [7, 'a'], [7, 'b'], [100, 'a'], [100, 'b']]
``` 
