import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler


class RandomSampler(Sampler):
    """

    ---

    ## Overview

    The random sampler generates samples randomly within the specified bounds
    for each parameter.

    ---

    ## Configuration

    To use the `RandomSampler`, specify it in the configuration file as in following example:

    ```yaml
      sampler:
        type: RandomSampler
        parameters: ['x', 'y']
        bounds: [[1, 10], [0, 1]]
        num_samples: 100
    ```

    ---
    
    Args:
        type (str): Sampler identifier. Must be set to `RandomSampler`.
        parameters (list[str]): Names of the parameters to be sampled. The order of parameters must correspond to the order of bounds.
        bounds (list[list[float]]): Lower and upper bounds for each parameter. Each element must be a list or tuple of two floats: `[min, max]`.
        num_samples (int): Total number of samples to generate. 

    ---

    ## Assumptions and Notes

     - Parameter values are sampled independently using a uniform distribution
      (`np.random.uniform`) within the specified bounds.

     - The sampler generates samples in batches; the batch size is controlled
      by `batch_size`.

     - If `batch_size` is not provided, it defaults to the full sampling budget.
     - The sampler does not adapt based on previous evaluations.

     - The sampler maintains an internal counter (`self.submitted`) tracking
      the number of generated samples.
      
     - The current implementation assumes continuous parameter spaces.
    
    ---

    ## Methods

      **`get_initial_parameters:`** Gets the initial parameters.

      **`get_next_samples:`** Gets the next sampled parameter configurations.

    ---

    """
    BATCH_SAMPLE_SIZE = 1

    def __init__(self, bounds, budget, parameters, **kwargs):
        """
        **`__init__:`** Initializes the RandomSampler.

        Args:
          bounds (list[tuple[float, float]]): Lower and upper bounds for each parameter.
          budget (int): Total number of samples that can be generated.
          parameters (list[str]): Names of the parameters to be sampled. The order must correspond to the order of bounds.
          batch_size (int, optional): Number of samples returned per call to `get_next_samples`. Defaults to the full sampling budget.
        """
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)

    def get_next_samples(self) -> list[dict]:
        """
        Generates the next batch of randomly sampled parameter configurations.

        Each parameter value is sampled independently from a uniform distribution
        within the specified bounds.

        Returns:
           list[dict[str, float]]:
              A batch of parameter dictionaries, where each dictionary maps
              parameter names to sampled numeric values.
        """
        # TODO not use uniform?
        # TODO batch samples
        list_param_dicts = []
        for _ in range(self.batch_size):
            params = [np.random.uniform(low, high) for low, high in self.bounds]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            list_param_dicts.append(param_dict)
        self.submitted += len(list_param_dicts)
        return list_param_dicts

    def register_future(self, future):
        """
        Registers a completed or scheduled evaluation.

        This method is part of the sampler interface but is not used by
        the RandomSampler, as sampling does not depend on evaluation results.

        Args:
          future:
              A future or handle representing an asynchronous evaluation.

        Returns:
          None
        """
        return None

    def register_futures(self, futures):
        """
        Registers multiple completed or scheduled evaluations.

        This method is part of the sampler interface but is not used by
        the RandomSampler. It is implemented as a no-op.

        Args:
          futures:
            An iterable of futures or handles representing asynchronous
            evaluations.

        Returns:
          None
        """
        return None
    
