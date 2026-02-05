"""
Deep Q-Network (DQN) Agent for Snake game.
Implements DQN with experience replay, target network, and Double DQN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os

try:
    from ..networks.dqn_network import DQNNetwork
    from ..utils.replay_buffer import ReplayBuffer
except ImportError:
    # Fallback for Colab/direct execution
    from networks.dqn_network import DQNNetwork
    from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent with experience replay and target network.
    
    Features:
    - Epsilon-greedy exploration
    - Experience replay buffer
    - Target network for stable Q-learning
    - Double DQN (use main network for action selection, target for value)
    """
    
    def __init__(
        self,
        state_shape: Tuple,
        num_actions: int = 4,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        replay_buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_frequency: int = 1000,
        hidden_sizes: list = [128, 128, 64],
        activation: str = "relu",
        state_representation: str = "feature",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_shape: Shape of state observations
            num_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon value
            epsilon_decay: Epsilon decay rate per step
            replay_buffer_size: Size of experience replay buffer
            batch_size: Batch size for training
            target_update_frequency: Steps between target network updates
            hidden_sizes: Hidden layer sizes for Q-network
            activation: Activation function
            state_representation: "grid", "feature", or "image"
            device: PyTorch device (CPU/GPU)
            seed: Random seed
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.state_representation = state_representation
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create Q-networks (main and target)
        self.q_network = DQNNetwork(
            state_shape=state_shape,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
            state_representation=state_representation
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            state_shape=state_shape,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
            state_representation=state_representation
        ).to(self.device)
        
        # Initialize target network with same weights as main network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            state_shape=state_shape,
            seed=seed
        )
        
        # Training statistics
        self.step_count = 0
        self.training_losses = []
        
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            deterministic: If True, always use greedy action (for evaluation)
        
        Returns:
            Selected action
        """
        if deterministic or np.random.random() > self.epsilon:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        else:
            # Random action (exploration)
            action = np.random.randint(0, self.num_actions)
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary with training metrics
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_mean": 0.0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Handle different state representations
        if self.state_representation == "grid":
            states_tensor = states_tensor.long()
            next_states_tensor = next_states_tensor.long()
        elif self.state_representation == "image":
            states_tensor = states_tensor.long()
            next_states_tensor = next_states_tensor.long()
        
        # Current Q-values
        current_q_values = self.q_network(states_tensor)
        current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1))
        
        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Use main network to select best action
            next_q_values_main = self.q_network(next_states_tensor)
            next_actions = next_q_values_main.argmax(1, keepdim=True)
            
            # Use target network to evaluate that action
            next_q_values_target = self.target_network(next_states_tensor)
            next_q_values = next_q_values_target.gather(1, next_actions)
            
            # Compute target: r + gamma * max Q(s', a') * (1 - done)
            target_q_values = rewards_tensor.unsqueeze(1) + \
                            self.gamma * next_q_values * (~dones_tensor).unsqueeze(1).float()
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store metrics
        metrics = {
            "loss": loss.item(),
            "q_mean": current_q_values.mean().item(),
            "epsilon": self.epsilon
        }
        self.training_losses.append(loss.item())
        
        return metrics
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.q_network.eval()
    
    def train(self):
        """Set agent to training mode."""
        self.q_network.train()
    
    def save(self, filepath: str):
        """Save agent state to file."""
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "state_shape": self.state_shape,
            "num_actions": self.num_actions,
            "state_representation": self.state_representation,
            "algorithm": "dqn"
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.step_count = checkpoint.get("step_count", 0)
        print(f"Model loaded from {filepath}")
