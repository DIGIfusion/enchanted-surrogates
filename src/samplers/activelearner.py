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
import parsers
from .base import Sampler
import h5py 

class ActiveLearnerBMDAL(Sampler):
    """
    Active Learner using BMDAL library for Active Learning.

    Args:
        total_budget (int): Total budget for the active learning process.
        parser_kwargs (dict): Keyword arguments for the parser.
        model_kwargs (dict): Keyword arguments for the model.
        train_kwargs (dict): Keyword arguments for training.

    Attributes:
        total_budget (int): Total budget for the active learning process.
        bounds (list): Bounds for the parameter space.
        parameters (list): Names of the parameters.
        aquisition_batch_size (int): Batch size for acquisition.
        init_num_samples (int): Number of initial samples.
        selection_method (str): Method for selecting samples.
        kernel_transform (list): Kernel transformation methods.
        base_kernel (str): Base kernel for BMDAL.
        parser_kwargs (dict): Keyword arguments for the parser.
        model_kwargs (dict): Keyword arguments for the model.
        train_kwargs (dict): Keyword arguments for training.
        metrics (list): List to store metrics.
    """

    sampler_interface = S.ACTIVE

    def __init__(
        self,
        total_budget: int,
        parser_kwargs,
        model_kwargs,
        train_kwargs,
        *args,
        **kwargs,
    ):
        """ 
        - needs parser_kwargs, model_kwargs and train_kwargs (model training keyword arguments)
        Sets the following parameters from passed kwargs: 
        - set param: keyword argument 
        - acquisition_batch_size: aquisition_batch_size
        - bounds: bounds
        - init_num_samples: num_initial_points
        - selection_method: selection_method
        - kernel_transform: kernel_transform
        - base_kernel: base_kernel
        """

        # fmt: off
        self.total_budget          = total_budget
        self.bounds                = kwargs.get('bounds', [None])
        self.parameters            = kwargs.get('parameters', ['NA']*len(self.bounds))
        self.aquisition_batch_size = kwargs.get('aquisition_batch_size', 512)
        self.init_num_samples      = kwargs.get('num_initial_points', self.aquisition_batch_size)
        self.selection_method      = kwargs.get('selection_method', 'random')
        self.kernel_transform      = kwargs.get('kernel_transform', [('rp', [512])])
        self.base_kernel           = kwargs.get('base_kernel', 'll')
        self.parser_kwargs         = parser_kwargs
        self.model_kwargs          = model_kwargs
        self.train_kwargs          = train_kwargs
        self.metrics = []

        self.parser                = getattr(parsers, parser_kwargs.pop('type'))(**parser_kwargs)
        # fmt: on

    def get_initial_parameters(self):
        """
        Generates initial parameter samples.

        Returns:
            list: List of dictionaries containing initial parameter samples.

        """
        # TODO: more than just random samples
        assert self.parameters[0] != "NA"
        batch_samples = []
        for _ in range(self.init_num_samples):
            params = [
                torch.distributions.Uniform(lb, ub).sample().item()
                for (lb, ub) in self.bounds
            ]
            param_dict = dict(zip(self.parameters, params))
            batch_samples.append(param_dict)
        return batch_samples

    def _get_new_idxs_from_pool(self, model: torch.nn.Module, train: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        """
        Selects new indices from the pool using BMDAL's select_batch function.
        NOTE: The training and pool data are expected to be normalized. 

        Returns:
            torch.Tensor: Tensor containing new indices.
        """
        print('Entering get new idx with',  train.shape, pool.shape)
        y_train = train[:, self.parser.output_col_idxs].float()
        x_train = train[:, self.parser.input_col_idxs].float()
        x_pool = pool[:, self.parser.input_col_idxs].float()
        train_data = TensorFeatureData(x_train)
        pool_data = TensorFeatureData(x_pool)
        new_idxs, _ = select_batch(
            batch_size=self.aquisition_batch_size,
            models=[model],
            data={"train": train_data, "pool": pool_data},
            y_train=y_train,
            selection_method=self.selection_method,
            sel_with_train=True,
            base_kernel=self.base_kernel,
            kernel_transforms=self.kernel_transform,
        )

        return new_idxs

    def get_next_parameter(
        self,
        model: torch.nn.Module,
        train: torch.Tensor,
        pool: torch.Tensor,
        *args,
        **kwargs,
    ) -> list[dict[str, float]]:
        """
        Generates the next parameter samples based on the model predictions given the train and pool.
        NOTE: The train and pool are expected to be normalised, as model and BMDAL expect data to be normalized. 

        Args:
            model (torch.nn.Module): PyTorch model.
            train (torch.Tensor): 
            pool (torch.Tensor):  

        Returns:
            list: List of dictionaries containing next parameter samples.

        """
        # train, pool = self.parser.train, self.parser.pool
        new_idxs = self._get_new_idxs_from_pool(model, train, pool)
        results = []
        for idx in new_idxs:
            results.append(
                {
                    "input": self.parser.pool[idx, self.parser.input_col_idxs],
                    "output": None,
                    "pool_idxs": idx,
                }
            )
        # TODO: convert indicies to unscaled parameters
        return results

    def collect_batch_results(self, results: list[dict[str, dict]]) -> torch.Tensor:
        """
        Collects results from the batch.

        Args:
            results (list): List of dictionaries containing batch results.

        Returns:
            torch.Tensor: Tensor containing batch results.

        """
        outputs_as_tensor = torch.empty(len(results), len(self.parser.keep_keys))
        for n, result in enumerate(results):
            x = result["inputs"]
            y = result["output"]
            outputs_as_tensor[n] = torch.cat((x, y))
        return outputs_as_tensor

    def update_pool_and_train(self):
        """
        """
        raise NotImplementedError()

    def update_metrics(self, metrics: dict) -> None:
        """
        NOTE: this assumes R2 score and regression metric is returned in the metrics dictionary.
        TODO: possibly update for more metrics
        """
        r2_score = metrics["val_r2_losses"]
        best_r2_score = max(r2_score)
        self.metrics.append(best_r2_score)


    def dump_iteration_results(self, base_run_dir: str, iterations: int, trained_model_state_dict: dict):
        """ 
        iteration number, size of train set, and metrics get appened to the file 'final_metrics.txt'
        while the current train set is saved  "train_{iterations}.pth" # TODO: update train set saving
        model is saved every 5 iterations 
        """

        
        train_fname = os.path.join(base_run_dir, f"train_{iterations}.pth")
        torch.save(self.parser.train, train_fname)
        

        final_metric_file = os.path.join(base_run_dir, "final_metrics.txt")
        with open(final_metric_file, "a") as f:
            f.write(f"{iterations}, {len(self.parser.train)}, {self.metrics[-1]}\n")
        
        model_path = os.path.join(base_run_dir, f'model_{iterations}.pth')
        if iterations % 5 == 0: 
            torch.save(trained_model_state_dict, model_path)

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

