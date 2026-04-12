"""
Deep Q-Network (DQN) agent with experience replay and target network.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from ..networks.dqn_network import DQNNetwork
    from ..utils.replay_buffer import ReplayBuffer
except ImportError:
    from networks.dqn_network import DQNNetwork
    from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN with replay buffer, target network, and epsilon-greedy exploration."""

    def __init__(
        self,
        state_shape: Tuple,
        num_actions: int = 4,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_frequency: int = 1000,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        state_representation: str = "feature",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Build Q- and target networks, optimizer, and replay buffer.

        Args:
            state_shape: Observation shape (feature length or grid dimensions).
            num_actions: Discrete action count.
            learning_rate: Adam learning rate.
            gamma: Discount factor.
            epsilon_start: Initial exploration rate.
            epsilon_end: Floor for epsilon after decay.
            epsilon_decay: Per-step multiplicative decay (after each train step).
            replay_buffer_size: Max transitions stored.
            batch_size: SGD minibatch size.
            target_update_frequency: Copy online weights to target every N train steps.
            hidden_sizes: MLP widths (feature/grid) or FC head (image).
            activation: ``relu``, ``tanh``, or ``elu``.
            state_representation: ``feature``, ``grid``, or ``image``.
            device: Torch device; default CUDA else CPU.
            seed: Optional RNG seed for numpy/torch/replay.
        """
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 64]

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.state_representation = state_representation
        self.train_steps = 0
        self.training_losses: List[float] = []

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.q_network = DQNNetwork(
            state_shape=state_shape,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
            state_representation=state_representation,
        ).to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_shape, seed=seed)

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert a single observation to a float batch of shape ``(1, ...)`` on ``device``."""
        x = np.asarray(state)
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return t.unsqueeze(0)

    def act(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> Union[int, Tuple]:
        """Epsilon-greedy action; greedy if ``deterministic`` or after exploration draw."""
        if not deterministic and random.random() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))

        self.q_network.eval()
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                s = state.float().to(self.device)
                if s.dim() == 1:
                    s = s.unsqueeze(0)
            else:
                s = self._state_to_tensor(state)
            q = self.q_network(s)
            return int(q.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append one transition to the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Dict[str, float]:
        """One gradient step from a replay minibatch; decay epsilon; periodic target sync."""
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=self.device)

        self.q_network.train()

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        li = loss.item()
        self.training_losses.append(li)
        return {"loss": li, "epsilon": self.epsilon}

    def eval(self) -> None:
        """Set Q-networks to eval mode (no dropout)."""
        self.q_network.eval()
        self.target_network.eval()

    def train(self) -> None:
        """Set Q-networks to training mode."""
        self.q_network.train()
        self.target_network.train()

    def save(self, filepath: str) -> None:
        """Persist weights, optimizer, epsilon, and metadata to ``filepath``."""
        checkpoint = {
            "algorithm": "dqn",
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "state_shape": self.state_shape,
            "num_actions": self.num_actions,
            "state_representation": self.state_representation,
        }
        d = os.path.dirname(filepath)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load checkpoint produced by :meth:`save`."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        tgt = checkpoint.get("target_network_state_dict")
        if tgt is not None:
            self.target_network.load_state_dict(tgt)
        else:
            self.target_network.load_state_dict(checkpoint["q_network_state_dict"])
        opt = checkpoint.get("optimizer_state_dict")
        if opt is not None:
            self.optimizer.load_state_dict(opt)
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_end))
        self.train_steps = int(checkpoint.get("train_steps", 0))
        print(f"Model loaded from {filepath}")
