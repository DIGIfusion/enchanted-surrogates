import numpy as np
import warnings
import os
from common import S
import pysgpp
from pysgpp import BoundingBox1D
import importlib
from scipy.stats import sobol_indices, uniform, entropy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# from .base import Sampler
from runners.MMMGrunner import MaxOfManyGaussians
from GPy.models import GPRegression
from GPy.kern import RBF
from GPy.kern import Matern32
from GPy.kern import Matern52
from GPy.kern import RatQuad
import numpy as np

class ActiveMaxGprUncertaintySampler():
    def __init__(self, bounds, parameters, num_samples_per_batch, initial_sampler, pool_sampler, *args, **kwargs):
        self.sampler_interface = S.ACTIVE
        self.bounds = np.array(bounds)
        self.parameters = parameters
        if type(parameters[0]) == type([]):
            print('CONVERTING LIST TO TUPLE')
            self.parameters = [tuple(pa) for pa in parameters]
            print(self.parameters, type(self.parameters[0]))
        
        self.custom_limit = np.inf
        self.custom_limit_value = 0
        self.dim=len(parameters)

        self.num_active_cycles = 0
        self.num_samples_by_cycle = []        
        
        # train is updated in active executor
        self.train = {}
        self.bounds_points = {}
        self.parent_points = {}
        
        initial_sampler_type = initial_sampler['type']
        initial_sampler['parameters'] = self.parameters
        initial_sampler['bounds'] = self.bounds
        self.initial_sampler = getattr(importlib.import_module(f'samplers.{initial_sampler_type}'),initial_sampler_type)(**initial_sampler) 
        
        pool_sampler_type = pool_sampler['type']
        pool_sampler['parameters'] = self.parameters
        pool_sampler['bounds'] = self.bounds
        self.pool_sampler = getattr(importlib.import_module(f'samplers.{pool_sampler_type}'),pool_sampler_type)(**pool_sampler)
        self.pool = self.pool_sampler.samples
        
        self.num_samples_per_batch = num_samples_per_batch
                
    def get_initial_parameters(self):
        return self.initial_sampler.get_initial_parameters()
    
    def get_next_parameters(
        self,
        cycle_dir:str=None,
        *args,
        **kwargs) -> list[dict[str, float]]:
        # returns a list of parameters that are the next batch to be labeled, based on the training samples, models used and selection criteria.
        """
        args
        train, dict: A dictionary where the keys are a tuple of inputs (x0,x1,x2) (*in the order of the origional parameters) and the value is a label
        """
        x = np.array(list(self.train.keys()))
        y = np.array(list(self.train.values()))

        # Shuffle in unison
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Split into folds
        x_folds = np.array_split(x_shuffled, self.num_samples_per_batch)
        y_folds = np.array_split(y_shuffled, self.num_samples_per_batch)
        
        batch_samples = []
        for x, y in zip(x_folds, y_folds):           
            regressor = self.model_fit(x,y)
            pool_pred_value, pool_pred_error = self.model_predict(self.pool, regressor)
            x_max_error = self.pool[np.argmax(pool_pred_error)]
            param_dict = {p:xi for p,xi in zip(self.parameters, x_max_error)}
            batch_samples.append(param_dict)
            
        return batch_samples
    
    def update_custom_limit_value(self): # Necessary
        NotImplemented
    
    def model_fit(self, x, y):
        # GPy needs a 2d Array
        x = np.array(x)
        y = np.array(y)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim == 1:
            x = x[:, None]

        kernel = RBF(input_dim=len(self.parameters))#, **fixed_kernel_args)
        
        print('OPTIMISING THE HYPERPERS')
        regressor = GPRegression(x, y, kernel, noise_var=0)
        print('CURRENT HYPERS:\n', regressor)
        print('OPTIMISING THE HYPERPERS:')
        regressor.Gaussian_noise.variance.fix(0)
        regressor.optimize_restarts(num_restarts = 10)
        print('RESULTING HYPERS:\n',regressor)
        return regressor

    def model_predict(self, x, regressor):
        input = np.array(x)
        if input.ndim == 1:
            input = input[:, None]
        if len(input.shape) == 1 and len(input) == self.dim: #prediction for one point
            input = np.array([x])
        y_predict, y_var = regressor.predict(input)
        # print(y_predict.shape)
        # print(y_predict) 
        y_2sig = np.sqrt(y_var[:,0]) * 2 
        return [y_predict[:,0], y_2sig]    
