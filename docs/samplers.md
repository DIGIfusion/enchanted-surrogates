# Samplers

Description of different samplers available in the enchanted-surrogates package and the configurations needed to use them.

## Random sampler

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

