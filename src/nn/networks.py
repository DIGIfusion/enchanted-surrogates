import torch 
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

def r2_score(y_pred, y_true): 
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0))**2)
    return 1.0 - (ss_res / ss_tot)

def create_model(model_kwargs: dict):
    model_width   = 512
    model_depth   = 1
    input_dim     = model_kwargs['input_dim']
    output_dim    = model_kwargs['output_dim']
    dropout       = 0.0

    nn.Sequential(nn.Linear(4, 100), nn.ReLU(), 
                          nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1))

    layers = [nn.Linear(input_dim, model_width), nn.Dropout(p=dropout), nn.ReLU()]
    for i in range(model_depth - 2):
        layers.append(nn.Linear(model_width, model_width))
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(model_width, output_dim))
    return nn.Sequential(*layers)


def run_train_model(model_kwargs: dict, train_data: tuple[torch.Tensor, torch.Tensor], valid_data: tuple[torch.Tensor, torch.Tensor]) -> nn.Module: 
    # TODO: Move to config file 
    batch_size    = 512
    EPOCHS        = 100
    learning_rate = 0.0001
    weight_decay  = 0.0
    
    # NOTE:  Create data loaders
    traindataset = TensorDataset(*train_data)
    valdataset   = TensorDataset(*valid_data)

    train_loader = DataLoader(traindataset, batch_size=batch_size)
    valid_loader = DataLoader(valdataset, batch_size=batch_size)

    # NOTE: Create model  & Optimizer & 

    model = create_model(model_kwargs)

    model.float()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # TODO: Scheduler & Early Stopping

    train_losses, val_losses, r2_losses = [], [], []
    best_loss = torch.inf
    for epoch in range(EPOCHS): 
        train_loss = train_step(model, optimizer, train_loader, epoch)
        val_loss, r2_loss  = validation_step(model, valid_loader)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        r2_losses.append(r2_loss)
        if val_losses[-1] < best_loss: 
            best_model = model 
            best_loss = val_losses[-1]
    return train_losses, val_losses, r2_losses, best_model
    

def train_step(model, optimizer, train_loader, epoch, device='cpu') -> float:
    step_loss = torch.tensor([0.0], dtype=torch.float, device=device)
    for batch_idx, batch in enumerate(train_loader): 
        optimizer.zero_grad()
        batch_on_device = tuple(item.to(device).float() for item in batch)
        X, y = batch_on_device 
        y_hat = model.forward(X)
        loss = F.mse_loss(y, y_hat)
        loss.backward() 
        optimizer.step() 
        step_loss += loss.detach().item()
    return step_loss / len(train_loader)

def validation_step(model, valid_loader, device='cpu') -> float: 
    step_loss = torch.tensor([0.0 ])
    step_r2   = torch.tensor([0.0])
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(valid_loader): 
            batch_on_device = tuple(item.to(device).float() for item in batch)
            X, y = batch_on_device 
            y_hat = model.forward(X)
            loss = F.mse_loss(y, y_hat)
            val_r2  = r2_score(y_hat, y)
            step_loss += loss.item()
            step_r2   += val_r2.item()
    return step_loss / len(valid_loader), step_r2 / len(valid_loader) 


class Regressor(nn.Module): 
    """  Simple regressor module """
    def __init__(self, input_dim, output_dim, model_depth=8, model_width=512,dropout=0.0, **kwargs):
        super(Regressor, self).__init__()
        
        layers = [nn.Linear(input_dim, model_width), nn.Dropout(p=dropout), nn.ReLU()]
        for i in range(model_depth - 2):
            layers.append(nn.Linear(model_width, model_width))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(model_width, output_dim))
        self.block = nn.Sequential(*layers)

        # self.block = nn.Sequential(nn.Linear(input_dim, model_width), nn.ReLU(), 
        #                            nn.Linear(model_width, model_width), nn.ReLU(), 
        #                            nn.Linear(100, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        y = self.block(x)
        return self.block(x)

