"""
Deep Q-Network (DQN) architecture.
Supports different state representations (grid, feature, image).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Snake game.

    Supports different state representations:
    - Grid: 2D convolutional layers
    - Feature: Fully connected layers
    - Image: 3D convolutional layers
    """

    def __init__(
        self,
        state_shape: Tuple,
        num_actions: int = 4,
        hidden_sizes: List[int] = [128, 128, 64],
        activation: str = "relu",
        state_representation: str = "feature"
    ):
        """
        Initialize Q-network.

        Args:
            state_shape: Shape of state observations
            num_actions: Number of possible actions
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ("relu", "tanh", "elu")
            state_representation: "grid", "feature", or "image"
        """
        super(DQNNetwork, self).__init__()

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.state_representation = state_representation


        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")


        if state_representation == "grid":
            self._build_grid_network(hidden_sizes)
        elif state_representation == "feature":
            self._build_feature_network(hidden_sizes)
        elif state_representation == "image":
            self._build_image_network(hidden_sizes)
        else:
            raise ValueError(f"Unknown state_representation: {state_representation}")

    def _build_feature_network(self, hidden_sizes: List[int]):
        """Build network for feature vector representation."""
        input_dim = self.state_shape[0]

        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size


        layers.append(nn.Linear(prev_size, self.num_actions))

        self.network = nn.Sequential(*layers)

    def _build_grid_network(self, hidden_sizes: List[int]):
        """Build network for grid representation."""
        grid_size = self.state_shape[0]


        self.flatten = nn.Flatten()
        input_dim = grid_size * grid_size

        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size


        layers.append(nn.Linear(prev_size, self.num_actions))

        self.network = nn.Sequential(*layers)

    def _build_image_network(self, hidden_sizes: List[int]):
        """Build network for image representation."""

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)


        grid_size = self.state_shape[0]

        conv_output_size = grid_size * grid_size * 64


        fc_layers = []
        prev_size = conv_output_size

        for hidden_size in hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, hidden_size))
            fc_layers.append(self.activation)
            prev_size = hidden_size

        fc_layers.append(nn.Linear(prev_size, self.num_actions))
        self.fc = nn.Sequential(*fc_layers)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: Batch of states

        Returns:
            Q-values for each action [batch_size, num_actions]
        """
        if self.state_representation == "image":

            x = state.float() / 255.0
            x = x.permute(0, 3, 1, 2)
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.state_representation == "grid":
            x = self.flatten(state.float())
            return self.network(x)
        else:
            return self.network(state.float())
