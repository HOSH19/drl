"""
Training utilities for Snake RL project.
Includes metrics tracking, checkpointing, and evaluation functions.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime


class MetricsTracker:
    """Track training metrics over episodes."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.episode_rewards = []
        self.episode_scores = []  # Food eaten per episode
        self.episode_lengths = []
        self.training_losses = []
        self.epsilon_values = []  # For DQN
        
    def record_episode(
        self,
        reward: float,
        score: int,
        length: int,
        loss: Optional[float] = None,
        epsilon: Optional[float] = None
    ):
        """
        Record metrics for an episode.
        
        Args:
            reward: Total episode reward
            score: Number of food eaten (score)
            length: Episode length in steps
            loss: Training loss (optional)
            epsilon: Epsilon value for DQN (optional)
        """
        self.episode_rewards.append(reward)
        self.episode_scores.append(score)
        self.episode_lengths.append(length)
        
        if loss is not None:
            self.training_losses.append(loss)
        if epsilon is not None:
            self.epsilon_values.append(epsilon)
    
    def get_statistics(self, window: int = 100) -> Dict[str, float]:
        """
        Get statistics over recent episodes.
        
        Args:
            window: Number of recent episodes to consider
        
        Returns:
            Dictionary with statistics
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_scores = self.episode_scores[-window:]
        recent_lengths = self.episode_lengths[-window:]
        
        stats = {
            "mean_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "max_reward": np.max(recent_rewards),
            "mean_score": np.mean(recent_scores),
            "std_score": np.std(recent_scores),
            "max_score": np.max(recent_scores),
            "mean_length": np.mean(recent_lengths),
            "std_length": np.std(recent_lengths),
        }
        
        if len(self.training_losses) > 0:
            recent_losses = self.training_losses[-window:]
            stats["mean_loss"] = np.mean(recent_losses)
        
        if len(self.epsilon_values) > 0:
            stats["current_epsilon"] = self.epsilon_values[-1]
        
        return stats
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_scores": self.episode_scores,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
            "epsilon_values": self.epsilon_values
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.episode_rewards = data["episode_rewards"]
        self.episode_scores = data["episode_scores"]
        self.episode_lengths = data["episode_lengths"]
        self.training_losses = data.get("training_losses", [])
        self.epsilon_values = data.get("epsilon_values", [])


def evaluate_agent(
    env,
    agent,
    num_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Evaluate agent performance.
    
    Args:
        env: Environment instance
        agent: Agent instance
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        deterministic: Whether to use deterministic policy
    
    Returns:
        Dictionary with evaluation metrics
    """
    agent.eval()
    
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            if hasattr(agent, 'act'):
                # DQN agent
                action = agent.act(state, deterministic=deterministic)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                state = next_state
            else:
                # PPO agent (returns action, log_prob, value)
                action, _, _ = agent.act(state, deterministic=deterministic)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                state = next_state
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_scores.append(info.get("score", 0))
        episode_lengths.append(episode_length)
    
    agent.train()
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_score": np.mean(episode_scores),
        "std_score": np.std(episode_scores),
        "max_score": np.max(episode_scores),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_scores": episode_scores,
        "episode_lengths": episode_lengths
    }
    
    return results


def create_checkpoint_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create checkpoint directory with timestamp.
    
    Args:
        base_dir: Base directory for checkpoints
        experiment_name: Name of experiment
    
    Returns:
        Path to checkpoint directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_dir, experiment_name, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
