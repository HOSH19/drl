"""
Visualization utilities for Snake RL project.
Includes training curves, policy visualization, and game replays.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Any
import os


def plot_training_curves(
    metrics_tracker,
    save_path: Optional[str] = None,
    window: int = 100
):
    """
    Plot training curves for rewards, scores, and episode lengths.
    
    Args:
        metrics_tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        window: Window size for moving average
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = np.arange(1, len(metrics_tracker.episode_rewards) + 1)
    
    # Episode rewards
    axes[0, 0].plot(episodes, metrics_tracker.episode_rewards, alpha=0.3, color='blue', label='Raw')
    if len(metrics_tracker.episode_rewards) >= window:
        moving_avg = np.convolve(
            metrics_tracker.episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        axes[0, 0].plot(
            episodes[window-1:],
            moving_avg,
            color='blue',
            linewidth=2,
            label=f'Moving Avg ({window})'
        )
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode scores (food eaten)
    axes[0, 1].plot(episodes, metrics_tracker.episode_scores, alpha=0.3, color='green', label='Raw')
    if len(metrics_tracker.episode_scores) >= window:
        moving_avg = np.convolve(
            metrics_tracker.episode_scores,
            np.ones(window) / window,
            mode='valid'
        )
        axes[0, 1].plot(
            episodes[window-1:],
            moving_avg,
            color='green',
            linewidth=2,
            label=f'Moving Avg ({window})'
        )
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score (Food Eaten)')
    axes[0, 1].set_title('Training Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(episodes, metrics_tracker.episode_lengths, alpha=0.3, color='red', label='Raw')
    if len(metrics_tracker.episode_lengths) >= window:
        moving_avg = np.convolve(
            metrics_tracker.episode_lengths,
            np.ones(window) / window,
            mode='valid'
        )
        axes[1, 0].plot(
            episodes[window-1:],
            moving_avg,
            color='red',
            linewidth=2,
            label=f'Moving Avg ({window})'
        )
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Episode Length')
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training loss (if available)
    if len(metrics_tracker.training_losses) > 0:
        loss_episodes = np.arange(1, len(metrics_tracker.training_losses) + 1)
        axes[1, 1].plot(loss_episodes, metrics_tracker.training_losses, alpha=0.5, color='purple')
        if len(metrics_tracker.training_losses) >= window:
            moving_avg = np.convolve(
                metrics_tracker.training_losses,
                np.ones(window) / window,
                mode='valid'
            )
            axes[1, 1].plot(
                loss_episodes[window-1:],
                moving_avg,
                color='purple',
                linewidth=2,
                label=f'Moving Avg ({window})'
            )
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


def plot_algorithm_comparison(
    metrics_dict: Dict[str, Any],
    save_path: Optional[str] = None,
    window: int = 100
):
    """
    Compare multiple algorithms' performance.
    
    Args:
        metrics_dict: Dictionary mapping algorithm names to MetricsTracker instances
        save_path: Path to save figure (optional)
        window: Window size for moving average
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for algo_name, tracker in metrics_dict.items():
        episodes = np.arange(1, len(tracker.episode_rewards) + 1)
        
        # Rewards
        if len(tracker.episode_rewards) >= window:
            moving_avg = np.convolve(
                tracker.episode_rewards,
                np.ones(window) / window,
                mode='valid'
            )
            axes[0].plot(
                episodes[window-1:],
                moving_avg,
                label=algo_name,
                linewidth=2
            )
        
        # Scores
        if len(tracker.episode_scores) >= window:
            moving_avg = np.convolve(
                tracker.episode_scores,
                np.ones(window) / window,
                mode='valid'
            )
            axes[1].plot(
                episodes[window-1:],
                moving_avg,
                label=algo_name,
                linewidth=2
            )
        
        # Lengths
        if len(tracker.episode_lengths) >= window:
            moving_avg = np.convolve(
                tracker.episode_lengths,
                np.ones(window) / window,
                mode='valid'
            )
            axes[2].plot(
                episodes[window-1:],
                moving_avg,
                label=algo_name,
                linewidth=2
            )
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Episode Scores Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Length')
    axes[2].set_title('Episode Lengths Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def plot_policy_heatmap(
    agent,
    env,
    grid_size: int = 20,
    save_path: Optional[str] = None
):
    """
    Visualize policy as heatmap showing action probabilities.
    
    Args:
        agent: Trained agent
        env: Environment instance
        grid_size: Size of grid
        save_path: Path to save figure (optional)
    """
    # Create a grid of states (simplified - using head position only)
    # This is a simplified visualization
    action_probs = np.zeros((grid_size, grid_size, 4))
    
    agent.eval()
    
    # Sample states by running episodes
    for _ in range(100):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 50:
            # Get action probabilities
            if hasattr(agent, 'q_network'):
                # DQN: use Q-values as proxy for probabilities
                with torch.no_grad():
                    import torch
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    q_values = agent.q_network(state_tensor).cpu().numpy()[0]
                    probs = np.exp(q_values) / np.sum(np.exp(q_values))  # Softmax
            else:
                # PPO: get distribution
                with torch.no_grad():
                    import torch
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    logits = agent.actor(state_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    probs = dist.probs.cpu().numpy()[0]
            
            # Get head position (simplified - assumes feature representation)
            if env.state_representation == "feature":
                head_x = int(state[0] * grid_size)
                head_y = int(state[1] * grid_size)
                head_x = np.clip(head_x, 0, grid_size - 1)
                head_y = np.clip(head_y, 0, grid_size - 1)
                action_probs[head_x, head_y] += probs
            
            action, _, _ = agent.act(state, deterministic=False) if hasattr(agent, 'act') else (agent.act(state, deterministic=False)[0], 0, 0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
    
    agent.train()
    
    # Normalize
    for i in range(grid_size):
        for j in range(grid_size):
            total = np.sum(action_probs[i, j])
            if total > 0:
                action_probs[i, j] /= total
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    for idx, (ax, name) in enumerate(zip(axes.flat, action_names)):
        im = ax.imshow(action_probs[:, :, idx], cmap='hot', interpolation='nearest')
        ax.set_title(f'Probability of {name}')
        ax.set_xlabel('Y Position')
        ax.set_ylabel('X Position')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Policy heatmap saved to {save_path}")
    else:
        plt.show()
