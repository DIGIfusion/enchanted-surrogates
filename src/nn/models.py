import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class Regressor(nn.Module):  # type: ignore
    def __init__(
        self,
        target_idx,
        inputs=15,
        outputs=1,
        model_size=8,
        dropout=0.1,
        scaler=StandardScaler,
        device=None,
    ) -> None:
        super().__init__()
        self.scaler = scaler
        self.dropout = dropout
        self.model_size = model_size
        layers = [nn.Linear(inputs, 512), nn.Dropout(p=dropout), nn.ReLU()]
        for i in range(model_size - 2):
            layers.append(nn.Linear(512, 512))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(512, outputs))
        self.model = nn.Sequential(*layers)
        self.set_device(device)
        return

    def set_device(self, device) -> None:
        self.device = device
        self.model.to(device)
        self.R2 = R2Score().to(device)
        return

    def set_scaler(self, scaler) -> None:
        self.scaler = scaler
        return

    def forward(self, x):  # type: ignore
        y_hat = self.model(x.float())
        return y_hat

    def unscale(self, y) -> np.array:
        # get the index of the scaler that corresponds to the target
        scaler_features = self.scaler.feature_names_in_
        scaler_index = np.where(scaler_features == self.flux)[0][0]
        return y * self.scaler.scale_[scaler_index] + self.scaler.mean_[scaler_index]

    def enable_dropout(self):  # type: ignore
        """Function to enable the dropout layers during test-time"""
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def loss_function(self, y, y_hat, train=True):  # type: ignore

        loss = self.loss(y_hat, y.float())
        loss = torch.sum(loss)
        if not train:
            y_hat = torch.Tensor(self.unscale(y_hat.detach().cpu().numpy()))
            y = torch.Tensor(self.unscale(y.detach().cpu().numpy()))
            loss_unscaled = self.loss(y_hat, y.float())
        else:
            loss_unscaled = None
        return loss, loss_unscaled

    def train_step(self, dataloader, optimizer, epoch=None, disable_tqdm=False):  # type: ignore

        losses = []
        for batch, (X, y, _, idx) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}", disable=disable_tqdm),
            0,
        ):

            X = X.to(self.device)
            y = y.to(self.device)
            y_hat = self.forward(X.float())

            loss, _ = self.loss_function(y.unsqueeze(-1).float(), y_hat)
            for param in self.model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
            loss = loss.item()
            losses.append(loss)

        average_loss = np.mean(losses)

        return average_loss

    def validation_step(self, dataloader):  # type: ignore

        validation_loss = []
        with torch.no_grad():
            for X, y, _, _ in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                y_hat = self.forward(X.float())
                loss, _ = self.loss_function(y.unsqueeze(-1).float(), y_hat)
                validation_loss.append(loss.item())

        average_loss = np.mean(validation_loss)

        return average_loss

    def predict(self, dataloader):  # type: ignore

        size = len(dataloader.dataset)
        pred = []
        losses = []
        losses_unscaled = []
        r2 = []
        popback = []

        for batch, (x, y, _, idx) in enumerate(tqdm(dataloader)):
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.forward(x.float())
            loss = self.loss_function(y.unsqueeze(-1).float(), y_hat, train=False)
            r2.append(self.R2(y_hat, y.unsqueeze(-1).float()).item())
            y_hat = y_hat.squeeze().detach().cpu().numpy()
            losses_unscaled.append(loss[1].item())
            loss = loss[0]
            y = self.unscale(y.squeeze().detach().cpu().numpy())
            y_hat = self.unscale(y_hat)

            if (self.flux not in particle_fluxes) and (
                self.flux not in momentum_fluxes
            ):  # particle  and momentum fluxes allowed to be <0
                try:
                    popback.append(len(y_hat[y_hat < 0]))
                except:
                    if y_hat < 0:
                        popback.append(1)

            losses.append(loss.item())

            try:
                pred.extend(y_hat)
            except:
                pred.extend([y_hat])

        average_loss = np.mean(losses)

        pred = np.hstack(pred)

        unscaled_avg_loss = np.mean(losses_unscaled)  # / size
        popback = np.sum(popback) / size * 100
        r2 = np.mean(r2)
        return pred, [average_loss, unscaled_avg_loss, popback, r2]

    def predict_avg_std(
        self,
        dataloader: DataLoader,
        n_runs: int = 50,
    ) -> Tuple[List[float], List[float], List[float]]:

        self.eval()
        self.enable_dropout()

        runs = []
        idx_list = []
        for i in range(n_runs):
            step_list = []
            for step, (x, y, _, idx) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                predictions = self(x.float()).detach().cpu().numpy()

                step_list.append(predictions)
                if i == 0:
                    idx_list.append(idx.detach().cpu().numpy())

            flat_list = [item for sublist in step_list for item in sublist]
            flattened_predictions = np.array(flat_list).flatten()
            runs.append(flattened_predictions)

        out_std = np.std(np.array(runs), axis=0)
        out_avg = np.mean(np.array(runs), axis=0)
        flat_list = [item for sublist in idx_list for item in sublist]
        idx_array = np.asarray(flat_list, dtype=object).flatten()

        return out_avg, out_std, idx_array

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        learning_rate: float = 5e-4,
        weight_decay: float = 1.0e-4,
        epochs: int = 10,
        patience: Union[None, int] = None,
        do_validation: bool = True,
        save_model: bool = False,
        save_dir: str = None,
        **cfg,  # type: ignore
    ) -> Tuple[List[float], List[float]]:

        train_loss, val_loss, model_reg = _fit_mlp(
            self,
            train_loader,
            valid_loader,
            learning_rate,
            weight_decay,
            epochs,
            patience,
            do_validation,
        )

        self.model = model_reg.model
        if save_model:
            torch.save(self.model.state_dict(), f"{save_dir}/regressor.h5")
        return train_loss, val_loss
