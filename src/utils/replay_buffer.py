"""
Experience Replay Buffer for DQN.
Stores and samples transitions for training.
"""

import numpy as np
from typing import Tuple, Optional
import random


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores transitions (state, action, reward, next_state, done) and
    provides efficient random sampling for training.
    """
    
    def __init__(self, capacity: int, state_shape: Tuple, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of state observations
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.size = 0
        self.idx = 0
        
        # Initialize buffers
        if len(state_shape) == 1:
            # 1D state (feature vector)
            self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
            self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        elif len(state_shape) == 2:
            # 2D state (grid)
            self.states = np.zeros((capacity, *state_shape), dtype=np.int32)
            self.next_states = np.zeros((capacity, *state_shape), dtype=np.int32)
        elif len(state_shape) == 3:
            # 3D state (image)
            self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
            self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported state shape: {state_shape}")
        
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size
