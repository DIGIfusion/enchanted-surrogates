import torch 
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

def run_train_model(model_kwargs: dict, train_data: tuple[torch.Tensor, torch.Tensor], valid_data: tuple[torch.Tensor, torch.Tensor]) -> nn.Module: 
    
    # TODO: Move to config file 
    batch_size    = 32
    EPOCHS        = 10
    learning_rate = 0.001
    weight_decay  = 0.0001
    
    # NOTE:  Create data loaders
    traindataset = TensorDataset(*train_data)
    valdataset   = TensorDataset(*valid_data)

    train_loader = DataLoader(traindataset, batch_size=batch_size)
    valid_loader = DataLoader(valdataset, batch_size=batch_size)

    model_kwargs['input_dim'] = train_data[0].shape[-1]
    model_kwargs['output_dim'] = train_data[1].shape[-1]
    
    # NOTE: Create model  & Optimizer & 

    # model = # Regressor(**model_kwargs)
    # layers = [nn.Linear(input_dim, model_width), nn.Dropout(p=dropout), nn.ReLU()]
    # for i in range(model_depth - 2):
    #     layers.append(nn.Linear(model_width, model_width))
    #     layers.append(nn.Dropout(p=dropout))
    #     layers.append(nn.ReLU())
    # layers.append(nn.Linear(model_width, output_dim))
    # model = nn.Sequential(*layers)
    model = nn.Sequential(nn.Linear(4, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1))

    model.float()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # TODO: Scheduler 

    train_losses, val_losses, counter = [], [], 0
    best_loss = torch.inf
    for epoch in range(EPOCHS): 
        train_loss = train_step(model, optimizer, train_loader, epoch)
        val_loss   = validation_step(model, valid_loader)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if val_losses[-1] < best_loss: 
            best_model = model 
            best_loss = val_losses[-1]
    return train_losses, val_losses, best_model 
    

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
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(valid_loader): 
            batch_on_device = tuple(item.to(device).float() for item in batch)
            X, y = batch_on_device 
            y_hat = model.forward(X)
            loss = F.mse_loss(y, y_hat)
            step_loss += loss.item()
    return step_loss / len(valid_loader) 


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

