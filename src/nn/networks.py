import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """ Calculates r2 score and returns float"""
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0))**2)
    return 1.0 - (ss_res / ss_tot)


def create_model(model_kwargs: dict) -> nn.Module:
    """given the model kwargs, returns a neural network"""
    model_width   = model_kwargs.get('model_width', 512)
    model_depth   = model_kwargs.get('model_depth', 1)
    dropout       = model_kwargs.get('dropout', 0.0)
    input_dim     = model_kwargs['input_dim']
    output_dim    = model_kwargs['output_dim']

    # TODO: more complicated models
    layers = [nn.Linear(input_dim, model_width), nn.Dropout(p=dropout), nn.ReLU()]
    for _ in range(model_depth - 2):
        layers.append(nn.Linear(model_width, model_width))
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(model_width, output_dim))
    return nn.Sequential(*layers)


def run_train_model(model_kwargs: dict,
                    train_data: tuple[torch.Tensor, torch.Tensor],
                    valid_data: tuple[torch.Tensor, torch.Tensor],
                    train_kwargs: dict,
                    ) -> tuple[dict[str, list[float]], dict]:
    """ Returns metrics and state dictionary, model is loaded via create_model()"""

    # TODO: Move to config file
    batch_size    = train_kwargs.get('batch_size', 512)
    epochs        = train_kwargs.get('epochs', 100)
    learning_rate = train_kwargs.get('learning_rate', 0.0001)
    weight_decay  = train_kwargs.get('weight_decay', 0.0)

    # NOTE:  Create data loaders
    traindataset  = TensorDataset(*train_data)
    valdataset    = TensorDataset(*valid_data)

    train_loader  = DataLoader(traindataset, batch_size=batch_size)
    valid_loader  = DataLoader(valdataset, batch_size=batch_size)

    # NOTE: Create model  & Optimizer

    model = create_model(model_kwargs)

    model.float()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    # TODO: Scheduler & Early Stopping

    train_losses, val_losses, r2_losses = [], [], []
    best_loss = torch.inf
    for epoch in range(epochs): 
        train_loss = train_step(model, optimizer, train_loader, epoch)
        val_loss, r2_loss  = validation_step(model, valid_loader)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        r2_losses.append(r2_loss)
        if val_losses[-1] < best_loss:
            best_model = model
            best_loss = val_losses[-1]

    metrics = {'train_losses': train_losses, 'val_losses': val_losses, 'val_r2_losses': r2_losses}
    return metrics, best_model.state_dict()

def train_step(model, optimizer, train_loader, epoch, device='cpu') -> float:
    """
    A single epoch training step, takes model, optimizer, dataloader, epoch and device (default 'cpu')
    returns loss averaged over batch
    """
    step_loss = torch.tensor([0.0], dtype=torch.float, device=device)
    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch_on_device = tuple(item.to(device).float() for item in batch)
        x, y = batch_on_device
        y_hat = model.forward(x)
        loss = F.mse_loss(y, y_hat)
        loss.backward()
        optimizer.step()
        step_loss += loss.detach().item()
    return step_loss / len(train_loader)


def validation_step(model, valid_loader, device='cpu') -> float: 
    """
    A single epoch validation step, takes model, dataloader, and device (default 'cpu')
    returns loss and r2 score averaged over batch 
    """
    step_loss = torch.tensor([0.0], dtype=torch.float, device=device)
    step_r2   = torch.tensor([0.0], dtype=torch.float, device=device)
    with torch.no_grad():
        for _, batch in enumerate(valid_loader):
            batch_on_device = tuple(item.to(device).float() for item in batch)
            X, y = batch_on_device 
            y_hat = model.forward(X)
            loss = F.mse_loss(y, y_hat)
            val_r2  = r2_score(y_hat, y)
            step_loss += loss.item()
            step_r2   += val_r2.item()
    return step_loss / len(valid_loader), step_r2 / len(valid_loader)