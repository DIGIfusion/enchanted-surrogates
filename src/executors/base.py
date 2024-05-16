# executors/base.py
import os
import shutil
from abc import ABC, abstractmethod
import uuid
import runners
from common import S
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Union, Tuple, List
from nn.models import Regressor
import numpy as np 
import copy 

def run_simulation_task(runner_args, params_from_sampler, base_run_dir):
    print("Making Run dir")
    run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
    os.mkdir(run_dir)
    runner = getattr(runners, runner_args["type"])(**runner_args)
    runner_output = runner.single_code_run(params_from_sampler, run_dir)
    result = {'input': params_from_sampler, 'output': runner_output} # TODO force all runners to return a dictionary
    return result


def run_train_model(
    model_kwargs: dict,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    valid_data: Tuple[torch.Tensor, torch.Tensor],
    # train_loader: DataLoader,
    # valid_loader: DataLoader,
) -> Tuple[List[float], List[float], nn.Module]:
    """Fits a Multi Layer Perceptron model.

    Args:
        model (nn.Module): The model to be fit
        train_loader (DataLoader): The training data
        valid_loader (DataLoader): The validation data
        learning_rate (int, optional): The learning rate. Defaults to 5e-4.
        weight_decay (int, optional): The weight decay. Defaults to 1.0e-4.
        epochs (int, optional): The training epochs. Defaults to 10.
        patience (Union[None, int], optional): The patience value. Defaults to None.
        do_validation (bool, optional): Whether to do the validation loop (for testing purposes should be False). Defaults to True.

    Returns:
        List: The training loss
        List: The validation loss
        nn.Module: The trained model.
    """
    model_kwargs['inputs'] = train_data[0].shape[-1]
    model = Regressor(**model_kwargs)
    # x_train, y_train = train_data
    train = TensorDataset(*train_data)
    valid = TensorDataset(*valid_data)

    train_loader = DataLoader(train, batch_size=25)
    valid_loader = DataLoader(valid, batch_size=25)
    

    # TODO: take from train_kwargs
    learning_rate = 5.0E-4
    weight_decay = 1.0E-4
    epochs = 10
    patience: Union[None, int] = None
    do_validation: bool = True

    if not patience:
        patience = epochs
    best_loss = np.inf
    # instantiate optimiser    
    opt = torch.optim.Adam(params=model.parameters(), lr=0.004, weight_decay=0.0001)
    # create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=1000,
        min_lr=(1 / 16) * learning_rate,
    )
    train_loss = []
    val_loss = []
    counter = 0
    for epoch in range(epochs):

        # logging.debug(f"Train Step:  {epoch}")

        loss = model.train_step(train_loader, opt, epoch=epoch)
        if isinstance(
            loss, tuple
        ):  # classifier also returns accuracy which we are not tracking atm
            loss = loss[0]
        train_loss.append(loss.item())

        if do_validation:
            validation_loss = model.validation_step(valid_loader)
            scheduler.step(validation_loss)
            val_loss.append(validation_loss)
            if validation_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = validation_loss

            else:
                counter += 1
                if counter > patience:
                    # logging.debug("Early stopping criterion reached")
                    break
        else:
            best_model = model

    return train_loss, val_loss, best_model

class Executor(ABC):
    def __init__(
        self, sampler, runner_args, base_run_dir, config_filepath, *args, **kwargs
    ):
        print("Starting Setup")
        self.sampler = sampler  # kwargs.get('sampler')
        self.runner_args = runner_args  # kwargs.get('runner_args')
        self.base_run_dir = base_run_dir  # , kwargs.get('base_run_dir')
        self.max_samples = self.sampler.total_budget
        self.config_filepath = config_filepath  # kwargs.get('config_filepath')
        self.clients = []
        print(config_filepath)
        print(f"Making directory of simulations at: {self.base_run_dir}")
        os.makedirs(self.base_run_dir, exist_ok=True)

        print("Base Executor Initialization")

        shutil.copyfile(config_filepath, os.path.join(self.base_run_dir, "CONFIG.yaml"))

    @abstractmethod
    def start_runs(self):
        raise NotImplementedError()

    def clean(self): 
        for client in self.clients: 
            client.close()
