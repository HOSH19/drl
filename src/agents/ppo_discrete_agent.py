"""
PPO Agent adapted for discrete action spaces (Snake game).
Uses Categorical distribution instead of Normal distribution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import os

from ..networks.dqn_network import DQNNetwork


class PPODiscreteAgent:
    """
    PPO Agent for discrete action spaces.
    
    Uses Categorical distribution for action sampling and policy updates.
    Implements PPO with clipped surrogate objective.
    """
    
    def __init__(
        self,
        state_shape: Tuple,
        num_actions: int = 4,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        hidden_sizes: List[int] = [128, 128, 64],
        activation: str = "relu",
        state_representation: str = "feature",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize PPO agent for discrete actions.
        
        Args:
            state_shape: Shape of state observations
            num_actions: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of update epochs per batch
            batch_size: Batch size for updates
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            state_representation: "grid", "feature", or "image"
            device: PyTorch device
            seed: Random seed
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
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
        
        # Create actor (policy) network
        self.actor = DQNNetwork(
            state_shape=state_shape,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
            state_representation=state_representation
        ).to(self.device)
        
        # Create critic (value) network - use a separate simple network
        # Build critic network manually
        if state_representation == "feature":
            input_dim = state_shape[0]
            layers = []
            prev_size = input_dim
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "elu":
                    layers.append(nn.ELU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, 1))
            self.critic = nn.Sequential(*layers).to(self.device)
        elif state_representation == "grid":
            grid_size = state_shape[0]
            input_dim = grid_size * grid_size
            layers = [nn.Flatten()]
            prev_size = input_dim
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "elu":
                    layers.append(nn.ELU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, 1))
            self.critic = nn.Sequential(*layers).to(self.device)
        else:  # image
            # Use CNN for image
            self.critic_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.critic_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.critic_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            grid_size = state_shape[0]
            conv_output_size = grid_size * grid_size * 64
            fc_layers = []
            prev_size = conv_output_size
            for hidden_size in hidden_sizes:
                fc_layers.append(nn.Linear(prev_size, hidden_size))
                if activation == "relu":
                    fc_layers.append(nn.ReLU())
                elif activation == "tanh":
                    fc_layers.append(nn.Tanh())
                elif activation == "elu":
                    fc_layers.append(nn.ELU())
                prev_size = hidden_size
            fc_layers.append(nn.Linear(prev_size, 1))
            self.critic_fc = nn.Sequential(*fc_layers).to(self.device)
            self.critic_activation = nn.ReLU() if activation == "relu" else (nn.Tanh() if activation == "tanh" else nn.ELU())
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        
        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Training statistics
        self.training_losses = []
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action using policy network.
        
        Args:
            state: Current state observation
            deterministic: If True, use greedy action
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action logits
            logits = self.actor(state_tensor)
            
            # Create categorical distribution
            dist = torch.distributions.Categorical(logits=logits)
            
            if deterministic:
                action = logits.argmax().item()
                log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
            else:
                action_tensor = dist.sample()
                action = action_tensor.item()
                log_prob = dist.log_prob(action_tensor).item()
            
            # Get value estimate
            if self.state_representation == "image":
                x = state_tensor.float() / 255.0
                x = x.permute(0, 3, 1, 2)
                x = self.critic_activation(self.critic_conv1(x))
                x = self.critic_activation(self.critic_conv2(x))
                x = self.critic_activation(self.critic_conv3(x))
                x = x.reshape(x.size(0), -1)
                value = self.critic_fc(x).item()
            else:
                value = self.critic(state_tensor.float()).item()
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """
        Store transition in buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate
            done: Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value of next state (0 if episode ended)
        
        Returns:
            Tuple of (advantages, returns)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
        
        # Compute returns
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform PPO update on stored trajectories.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.states) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Handle different state representations
        if self.state_representation == "grid":
            states_tensor = states_tensor.long()
        elif self.state_representation == "image":
            states_tensor = states_tensor.long()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Multiple update epochs
        indices = np.arange(len(self.states))
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start_idx in range(0, len(self.states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Get current policy distribution
                logits = self.actor(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                
                # Compute new log probabilities
                new_log_probs = dist.log_prob(batch_actions)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.state_representation == "image":
                    x = batch_states.float() / 255.0
                    x = x.permute(0, 3, 1, 2)
                    x = self.critic_activation(self.critic_conv1(x))
                    x = self.critic_activation(self.critic_conv2(x))
                    x = self.critic_activation(self.critic_conv3(x))
                    x = x.reshape(x.size(0), -1)
                    values = self.critic_fc(x).squeeze()
                else:
                    values = self.critic(batch_states.float()).squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        num_updates = self.update_epochs * (len(indices) // self.batch_size + 1)
        metrics = {
            "loss": (total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy) / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }
        
        self.training_losses.append(metrics["loss"])
        return metrics
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
    
    def train(self):
        """Set agent to training mode."""
        self.actor.train()
        self.critic.train()
    
    def save(self, filepath: str):
        """Save agent state to file."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_shape": self.state_shape,
            "num_actions": self.num_actions,
            "state_representation": self.state_representation
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {filepath}")
