"""
samplers/activelearner.py

As per the executor, and in addition to the and get_initial_parameters(),
active learners samplers need the following methods:
- collect_batch_results(outputs)

If using active learning with your code, your parser must also have:
- Parameters:
    - train, valid, test
- Methods:
    - update_pool_and_train(outputs) -> None
        - Updates the pool and train attributes
    - get_train_valid_datasplit() -> (x_train, y_train), (x_valid, y_valid)
    -
"""

try:
    from bmdal_reg.bmdal.algorithms import select_batch
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
except ImportError as e:
    print(
        "Cannot find bmdal_reg.",
        "Please point to those in your bash script if you intend to use ActiveLearnerBMDAL",
        str(e),
    )
import os
import torch
import numpy as np
from common import S
# import parsers
from .base import Sampler
import h5py 

from samplers.ActiveLearnerBMDAL import ActiveLearnerBMDAL



class ActiveLearningBMDALStaticPoolSampler(ActiveLearnerBMDAL):
    """
    Active Learning BMDAL Static Pool Sampler.

    Loads a database of runs that have already been collected.
    This is useful for testing various acquisition strategies
    in Active Learning on fixed datasets. Uses Pandas under the hood.
    """

    sampler_interface = S.ACTIVEDB

    def __init__(self, **kwargs):
        """

        """
        super().__init__(**kwargs)

    def collect_batch_results(self, results: list[dict[str, dict]]) -> torch.Tensor:
        """
        Collects results, ignoring outputs as this is labeled static pool-based.
        """
        idxs_as_tensor = torch.tensor([res["input"]["pool_idxs"] for res in results])
        return idxs_as_tensor

    def get_initial_parameters(self) -> list[dict]:
        """
        Returns a subset of initial parameters.
        """
        params = []
        for _ in range(self.init_num_samples):
            idxs = np.random.randint(0, len(self.parser.pool))
            result = {
                "input": self.parser.pool[idxs, self.parser.input_col_idxs],
                "output": self.parser.pool[idxs, self.parser.output_col_idxs],
                "pool_idxs": idxs,
            }
            params.append(result)
        return params

    def get_next_parameter(
        self, model: torch.nn.Module, train: torch.Tensor, pool: torch.Tensor
    ) -> list[dict[str, float]]:
        """
        Generates the next parameter samples based on the model predictions.
        """
        idxs = self._get_new_idxs_from_pool(model, train, pool)
        results = []
        for idx in idxs:
            results.append(
                {
                    "input": self.parser.pool[idx, self.parser.input_col_idxs],
                    "output": self.parser.pool[idx, self.parser.output_col_idxs],
                    "pool_idxs": idx,
                }
            )
        return results

