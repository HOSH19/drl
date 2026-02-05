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

def create_checkpoint_dir(path: str):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

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
        render: Whether to render
    """
    episode_rewards = []
    episode_scores = []
    episode_lengths = []

    if render:
        from src.environments.snake_renderer import SnakeRenderer
        renderer = SnakeRenderer(env.grid_size)

    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.act(state, deterministic=deterministic)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state

            if render:
                # Pass specific arguments to renderer.render()
                renderer.render(env.snake, env.food, env.score, env.steps)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(env.score) # Fixed: Use env.score directly

        if render:
            renderer.close()

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)

    mean_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    max_score = np.max(episode_scores)

    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "max_reward": float(max_reward),
        "mean_score": float(mean_score),
        "std_score": float(std_score),
        "max_score": int(max_score),
        "mean_length": float(mean_length),
        "std_length": float(std_length),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_scores": [int(s) for s in episode_scores],
        "episode_lengths": [int(l) for l in episode_lengths],
    }
