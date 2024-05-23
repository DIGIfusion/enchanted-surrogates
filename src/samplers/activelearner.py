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
    print('Cannot find bmdal_reg, please point to those in your bash script if you intend to use ActiveLearnerBMDAL')
import os
import torch
import numpy as np
from common import S
import parsers 
from .base import Sampler


class ActiveLearnerBMDAL(Sampler):
    sampler_interface = S.ACTIVE
    def __init__(self, total_budget: int, parser_kwargs, model_kwargs, train_kwargs, *args, **kwargs): 
        self.total_budget          = total_budget
        self.bounds                = kwargs.get('bounds', [None])
        self.parameters            = kwargs.get('parameters', ['NA']*len(self.bounds))
        self.aquisition_batch_size = kwargs.get('aquisition_batch_size', 512)
        self.init_num_samples      = kwargs.get('num_initial_points', self.aquisition_batch_size)
        self.selection_method      = kwargs.get('selection_method', 'random')
        self.kernel_transform      = kwargs.get('kernel_transform', [])
        self.base_kernel           = kwargs.get('base_kernel', 'll')

        self.parser_kwargs         = parser_kwargs
        self.model_kwargs          = model_kwargs
        self.train_kwargs          = train_kwargs
        self.metrics = []

        self.parser                = getattr(parsers, parser_kwargs.pop('type'))(**parser_kwargs)

    def get_initial_parameters(self):
        # TODO: more than just random samples
        assert self.parameters[0] != 'NA'
        batch_samples = []
        for _ in range(self.init_num_samples):
            params = [torch.distributions.Uniform(lb, ub).sample().item() for (lb, ub) in self.bounds]
            param_dict = dict(zip(self.parameters, params))
            batch_samples.append(param_dict)
        return batch_samples

    def _get_new_idxs_from_pool(self, model, train, pool) -> torch.Tensor:
        y_train = train[:, self.parser.output_col_idxs].float()
        train = train[:, self.parser.input_col_idxs].float()
        pool = pool[:, self.parser.input_col_idxs].float()
        train_data = TensorFeatureData(train)
        pool_data = TensorFeatureData(pool)

        new_idxs, _ = select_batch(batch_size=self.aquisition_batch_size, models=[model],
                           data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                           selection_method=self.selection_method, sel_with_train=True,
                           base_kernel=self.base_kernel, kernel_transforms=self.kernel_transform)

        return new_idxs

    def get_next_parameter(self, model: torch.nn.Module, *args, **kwargs) -> list[dict[str, float]]:
        train, pool = self.parser.train, self.parser.pool
        new_idxs = self._get_new_idxs_from_pool(model, train,pool)
        results = []
        for idx in new_idxs:
            results.append({'input': self.parser.pool[idx, self.parser.input_col_idxs], 'output': None, 'pool_idxs':idx}) 
        # TODO: convert indicies to unscaled parameters
        return results

    def collect_batch_results(self, results: list[dict[str, dict]]) -> torch.Tensor:
        outputs_as_tensor = torch.empty(len(results), len(self.parser.keep_keys))
        for n, result in enumerate(results):
            x = result['inputs']
            y = result['output']
            outputs_as_tensor[n] = torch.cat((x, y))
        return outputs_as_tensor

    def update_pool_and_train(self):
        raise NotImplementedError()
    
    def update_metrics(self, metrics: dict) -> None:
        """ Metrics come from the training run """
        r2_score =  metrics['val_r2_losses']
        best_r2_score = max(r2_score)
        self.metrics.append(best_r2_score)


class ActiveLearningBMDALStaticPoolSampler(ActiveLearnerBMDAL):
    """
    Loads a database of runs that have already been collected. 
    This is useful for testing various acquisition strategies
    in Active Learning on fixed datasets. Uses Pandas under the hood.
    """
    sampler_interface = S.ACTIVEDB
    def __init__(self, **kwargs):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """
        super().__init__(**kwargs)

    def collect_batch_results(self, results: list[dict[str, dict]]) -> torch.Tensor:
        """ ignores outputs as this is labeled static pool based """
        idxs_as_tensor = torch.tensor([res['input']['pool_idxs'] for res in results])
        return idxs_as_tensor
    
    def get_initial_parameters(self) -> list[dict]:
        """ Returns a subset of initial parameters """
        params = []
        for _ in range(self.init_num_samples):
            idxs   = np.random.randint(0, len(self.parser.pool))
            result = {'input': self.parser.pool[idxs, self.parser.input_col_idxs], 'output': self.parser.pool[idxs, self.parser.output_col_idxs], 'pool_idxs':idxs}
            params.append(result)
        return params

    def get_next_parameter(self, model: torch.nn.Module) -> list[dict[str, float]]:
        idxs    = self._get_new_idxs_from_pool(model, self.parser.train, self.parser.pool)
        results = []
        for idx in idxs: 
            results.append({'input': self.parser.pool[idx, self.parser.input_col_idxs], 'output': self.parser.pool[idx, self.parser.output_col_idxs], 'pool_idxs':idx}) 
        return results

    def dump_results(self, base_run_dir) -> None:
        print(100*'=')
        print(f'Final Results for {self.selection_method}')

        print(self.metrics)
        print(f'Saving final datasets to disk: {base_run_dir}')
        for name, dset in zip(['train', 'valid', 'test'], [self.parser.train, self.parser.valid, self.parser.test]): 
            fpath = os.path.join(base_run_dir, f'{name}.pt')
            torch.save(dset, fpath)

        