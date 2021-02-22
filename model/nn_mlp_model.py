from abc import ABC
from typing import Union, Dict
import time

import torch
from torch import nn
import numpy as np

from model.base_model import BaseModel
from utils.moving_average_module import MovingAverageModule


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

        # Normalizer
        self.norm_x = MovingAverageModule(self.x_dims, device=self.device)
        self.norm_y = MovingAverageModule(self.y_dims, device=self.device)

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

    def infer(self,
              obs: Union[torch.Tensor, np.ndarray],
              act: Union[torch.Tensor, np.ndarray]):
        # Switch to eval mode
        self.net.eval()

        # Concatenate
        x = torch.cat([obs, act], dim=-1)

        # Kernel
        x = self.kernel(x)

        # Normalize
        x = self.norm_x.normalize(x)

        # Infer
        delta = self.net(x)

        # Denormalize
        delta = self.norm_y.denormalize(delta)

        # Add
        next_obs = obs + delta
        return next_obs

    def fit(self,
            dataset: Dict[str, Union[torch.Tensor, np.ndarray]],
            num_epochs: int = 100):
        # Switch to train mode
        self.net.train()

        # Convert dataset
        dataset_x = np.concatenate([dataset["obs"], dataset["act"]], axis=-1)
        dataset_y = dataset["next_obs"] - dataset["obs"]

        # To tensor
        dataset_x = torch.tensor(dataset_x, dtype=torch.float)
        dataset_y = torch.tensor(dataset_y, dtype=torch.float)

        # Kernel
        dataset_x = self.kernel(dataset_x)

        # Normalize
        self.norm_x.update(dataset_x)
        self.norm_y.update(dataset_y)
        dataset_x = self.norm_x.normalize(dataset_x)
        dataset_y = self.norm_y.normalize(dataset_y)

        # Data loader
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(dataset_x, dataset_y),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True)

        # Loss fn
        loss_fn = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            time_start = time.time()
            losses = []
            for data_x, data_y in data_loader:
                # Transfer data
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)

                # Train model
                y_hat = self.net(data_x)
                loss = loss_fn(y_hat, data_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Record loss
                losses.append(loss.item())

            # Print
            time_elapsed = time.time() - time_start
            print("Epoch #{}: loss {} time {:.2f}s".format(epoch, np.mean(losses), time_elapsed))

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

        # Calculate relative root mse
        mse = np.mean((y_hat - dataset_y) ** 2, axis=0)
        rel_root_mse = np.sqrt(mse / np.std(dataset_y))
        overall_rel_root_mse = np.mean(rel_root_mse)

        print("Relative Root MSE (%):\n{}\nAverage (%): {}".format(
            rel_root_mse * 100,
            overall_rel_root_mse * 100))

        return {
            "mse": mse,
            "rel_root_mse": rel_root_mse
        }
