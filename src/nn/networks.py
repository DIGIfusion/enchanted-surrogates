""" 
nn/networks.py

handles neural networks for active learning surrogates

- model initialization: create_model()
- model training: run_train_model(), train_step(), validation_step() 
- model loading: load_saved_model()
"""

import io
import torch
from torch import nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from dask.distributed import print


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Calculates r2 score and returns float"""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0)) ** 2)
    return 1.0 - (ss_res / ss_tot)


def cast_state_dict_from_gpu_to_cpu(model_state_dict: dict) -> dict:
    buffer = io.BytesIO()
    torch.save(model_state_dict, buffer)
    buffer.seek(0)

    cpu_state_dict = torch.load(buffer, map_location=torch.device("cpu"))
    return cpu_state_dict


def load_saved_model(
    model_kwargs: dict, model_state_dict: dict, device="cpu"
) -> nn.Module:
    """
    Loads saved model use create_model and a saved state dict
    the loading via bytesIo is for GPU to CPU conversion
    """
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(model_kwargs)
    cpu_state_dict = cast_state_dict_from_gpu_to_cpu(model_state_dict)
    model.load_state_dict(cpu_state_dict)
    model.to(device)
    return model


def create_model(model_kwargs: dict) -> nn.Module:
    """given the model kwargs, returns a neural network"""
    model_width = model_kwargs.get("model_width", 512)
    model_depth = model_kwargs.get("model_depth", 1)
    dropout = model_kwargs.get("dropout", 0.0)
    input_dim = model_kwargs["input_dim"]
    output_dim = model_kwargs["output_dim"]

    # TODO: more complicated models
    layers = [nn.Linear(input_dim, model_width), nn.Dropout(p=dropout), nn.ReLU()]
    for _ in range(model_depth - 2):
        layers.append(nn.Linear(model_width, model_width))
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(model_width, output_dim))
    return nn.Sequential(*layers)


def run_train_model(
    model_kwargs: dict,
    train_data: tuple[torch.Tensor, torch.Tensor],
    valid_data: tuple[torch.Tensor, torch.Tensor],
    train_kwargs: dict,
    model_state_dict: dict=None,
) -> tuple[dict[str, list[float]], dict]:
    """Returns metrics and state dictionary, model is loaded via create_model()
       If model_state_dict is passed, a model will be initialised based on it. Useful for continual learning.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = train_kwargs.get("batch_size", 512)
    epochs = train_kwargs.get("epochs", 100)
    learning_rate = train_kwargs.get("learning_rate", 0.0001)
    weight_decay = train_kwargs.get("weight_decay", 0.0)
    patience = train_kwargs.get("patience",5)

    # NOTE:  Create data loaders
    traindataset = TensorDataset(*train_data)
    valdataset = TensorDataset(*valid_data)

    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valdataset, batch_size=len(valdataset), shuffle=False)

    # NOTE: Create model  & Optimizer
    if model_state_dict is not None:
        print("obtaining previous model in run function")
        model = load_saved_model(model_kwargs, model_state_dict, device)
    else:
        model = create_model(model_kwargs)
        model.to(device=device)

    model.float()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # TODO: Scheduler & Early Stopping
    best_model_state_dict = model.state_dict()
    train_losses, val_losses, r2_losses = [], [], []
    best_loss = torch.inf
    counter = 0 
    for epoch in range(epochs):
        train_loss = train_step(model, optimizer, train_loader, epoch, device=device)
        val_loss, r2_loss = validation_step(model, valid_loader, device=device)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        r2_losses.append(r2_loss.item())
        if val_losses[-1] < best_loss or epoch == 0:
            best_model_state_dict = model.state_dict()
            best_loss = val_losses[-1]
            counter = 0
        else:
            counter+=1
            if counter==patience:
                print(f"Early stopping reached at epoch: {epoch}")
                break
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_r2_losses": r2_losses,
    }
    best_model_state_dict = cast_state_dict_from_gpu_to_cpu(best_model_state_dict)
    return metrics, best_model_state_dict


def train_step(model, optimizer, train_loader, epoch, device="cpu") -> torch.Tensor:
    """
    A single epoch training step, takes model, optimizer, dataloader, epoch and device (default 'cpu')
    returns loss averaged over batch
    """
    step_loss = torch.tensor([0.0], dtype=torch.float, device=device)
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch_on_device = tuple(item.to(device).float() for item in batch)
        x, y = batch_on_device
        y_hat = model.forward(x)
        loss = F.mse_loss(y, y_hat)
        #print(f"Step: {step}, Loss: {loss},x: {x}, y: {y}, yhat: {y_hat}")
        loss.backward()
        optimizer.step()
        step_loss += loss.detach().item()
    return step_loss / len(train_loader)


def validation_step(
    model, valid_loader, device="cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A single epoch validation step, takes model, dataloader, and device (default 'cpu')
    returns loss and r2 score averaged over batch
    """
    step_loss = [] #torch.tensor([0.0], dtype=torch.float, device=device)
    step_r2 =  []  #torch.tensor([0.0], dtype=torch.float, device=device)
    with torch.no_grad():
        for _, batch in enumerate(valid_loader):
            batch_on_device = tuple(item.to(device).float() for item in batch)
            X, y = batch_on_device
            y_hat = model.forward(X)
            loss = F.mse_loss(y, y_hat)
            val_r2 = r2_score(y_hat, y)
            step_loss.append(loss.item())
            # NOTE: if val_r2 is inifinite, it means that there is no diversity in the batch, we exclude it            
           # if (not torch.isinf(val_r2)) and val_r2>-10: # -10 is arbitrary, sometimes the batch diversity is tiny and therefore val_r2 can be very negative although not inf
            step_r2.append(val_r2.item())
    return np.mean(step_loss), np.mean(step_r2)


def test_model_continual_learning(
    trained_model_kwargs: dict,
    trained_model_state_dict: dict,
    test_data: tuple[torch.Tensor, torch.Tensor],
    acquisition_batch: int,
    current_iteration: int,   
    train_kwargs: dict
) -> dict[str, list[float]]:
    """
    Test on previous tasks in a continual learning scenario where only the latest training data is used but performance on previous tasks must be retained
    """
    batch_size = train_kwargs.get("batch_size",512)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_saved_model(trained_model_kwargs, trained_model_state_dict, device)
    metrics = []

    if current_iteration==0:
        metrics.append(
            {'test_losses': np.nan,
            'test_r2_losses': np.nan}
            )    
    else:
        for i in range(1,current_iteration+1): 
            print("unwrapping iterations ", i)
            x_test, y_test = test_data
            x_test = x_test[(i-1)*acquisition_batch:i*acquisition_batch, :] 
            y_test = y_test[(i-1)*acquisition_batch:i*acquisition_batch, :] 
            test = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
            mse, r2 = validation_step(model, test_loader, device)
            print(f'testing', mse, r2)
            metrics.append(
                {'test_losses': mse,
                'test_r2_losses': r2}
                )
    return metrics


