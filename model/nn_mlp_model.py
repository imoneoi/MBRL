from abc import ABC
from typing import Union, Dict

import torch
from torch import nn
import numpy as np

from model.base_model import BaseModel


class NNMlpModel(BaseModel, ABC):
    def __init__(self,
                 state_dims: int,
                 action_dims: int,

                 num_hidden: int = 256,
                 num_layers: int = 2,

                 learning_rate: float = 1e-3,
                 batch_size: int = 1024,

                 device: torch.device = torch.device("cuda:0")
                 ):
        super().__init__()
        self.device = device

        self.kernel_size = 3
        self.x_dims = self.kernel_size * (state_dims + action_dims)
        self.y_dims = state_dims

        # Multilayer perceptron
        layers = [
            nn.BatchNorm1d(self.x_dims, momentum=1e-4),  # Data Normalization, using 10000 (1e-4) sample average

            self.linear_layer(self.x_dims, num_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden)
        ]

        for _ in range(num_layers - 1):  # Hidden layers
            layers += [
                self.linear_layer(num_hidden, num_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(num_hidden)
            ]

        layers += [  # Output layers
            self.linear_layer(num_hidden, self.y_dims)
        ]

        self.net = nn.Sequential(*layers).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    '''
        Linear layer with orthogonal init
    '''
    @staticmethod
    def linear_layer(*args, **kwargs):
        linear = nn.Linear(*args, **kwargs)

        torch.nn.init.orthogonal_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        return linear

    '''
        Trigonometric Kernel
    '''
    @staticmethod
    def kernel(x: torch.Tensor):
        return torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)

    def forward(self, x: torch.Tensor):
        # Kernel
        x = self.kernel(x)

        # Infer
        delta = self.net(x)
        return delta

    def infer(self,
              obs: Union[torch.Tensor, np.ndarray],
              act: Union[torch.Tensor, np.ndarray]):
        # Switch to eval mode
        self.net.eval()

        # Concatenate
        x = torch.cat([obs, act], dim=-1)

        # Infer
        delta = self(x)

        # Add
        next_obs = obs + delta
        return next_obs

    def fit(self,
            dataset: Dict[str, Union[torch.Tensor, np.ndarray]],
            num_epochs: int = 100):
        # Switch to train mode
        self.net.train()

        # Concatenate dataset
        dataset_x = np.concatenate([dataset["obs"], dataset["act"]], axis=-1)
        dataset_y = dataset["next_obs"] - dataset["obs"]

        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(dataset_x, dtype=torch.float),
                torch.tensor(dataset_y, dtype=torch.float)
            ),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True)

        # Loss fn
        loss_fn = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            losses = []
            for data_x, data_y in data_loader:
                # Transfer data
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)

                # Train model
                y_hat = self(data_x)
                loss = loss_fn(y_hat, data_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Record loss
                losses.append(loss.detach().cpu().numpy())

            # Print
            print("Epoch #{}: loss {}".format(epoch, np.mean(losses)))

    def test(self, dataset: Dict[str, Union[torch.Tensor, np.ndarray]]):
        # Concatenate dataset
        dataset_x = np.concatenate([dataset["obs"], dataset["act"]], axis=-1)
        dataset_y = dataset["next_obs"]

        # Dimensions
        state_dims = dataset["obs"].shape[-1]

        # Infer model
        y_hat = []
        with torch.no_grad():
            n = dataset_x.shape[0]
            bs = self.batch_size
            num_batch = int(np.ceil(n / bs))

            for b_id in range(num_batch):
                x_batch = torch.tensor(dataset_x[b_id * bs: (b_id + 1) * bs],
                                       dtype=torch.float, device=self.device)

                y_hat_batch = self.infer(
                    x_batch[:, :state_dims],
                    x_batch[:, state_dims:]
                )

                y_hat.append(y_hat_batch.cpu().numpy())

        y_hat = np.concatenate(y_hat, axis=0)

        # Calculate relative rmse
        mse = np.mean((y_hat - dataset_y) ** 2, axis=0)
        rel_rmse = np.sqrt(mse / np.std(dataset_y))
        overall_rel_rmse = np.mean(rel_rmse)

        print("Relative RMSE (%):\n{}\nAverage (%): {}".format(
            rel_rmse * 100,
            overall_rel_rmse * 100))

        return {
            "mse": mse,
            "rel_rmse": rel_rmse
        }
