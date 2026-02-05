"""
Main training script for Snake RL project.
Supports both DQN and PPO algorithms.
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from environments import SnakeEnv
from agents import DQNAgent, PPODiscreteAgent
from utils.training import MetricsTracker, evaluate_agent, create_checkpoint_dir
from utils.visualization import plot_training_curves


def train_dqn(env, agent, config, metrics_tracker, checkpoint_dir):
    """Train DQN agent."""
    total_episodes = config['training']['total_episodes']
    eval_frequency = config['training']['eval_frequency']
    save_frequency = config['training']['save_frequency']
    update_frequency = config['training']['update_frequency']
    
    print(f"Starting DQN training for {total_episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"State representation: {config['environment']['state_representation']}")
    
    best_score = -np.inf
    
    for episode in tqdm(range(total_episodes), desc="Training"):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Collect episode
        final_info = info  # Initialize with reset info
        while not done:
            action = agent.act(state, deterministic=False)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            final_info = step_info  # Update with latest step info (contains final score)
        
        # Train agent
        if len(agent.replay_buffer) >= agent.batch_size:
            if episode % update_frequency == 0:
                metrics = agent.train_step()
                metrics_tracker.record_episode(
                    reward=episode_reward,
                    score=final_info.get("score", 0),  # Use final step info
                    length=episode_length,
                    loss=metrics.get("loss", None),
                    epsilon=metrics.get("epsilon", None)
                )
            else:
                metrics_tracker.record_episode(
                    reward=episode_reward,
                    score=final_info.get("score", 0),  # Use final step info
                    length=episode_length,
                    epsilon=agent.epsilon
                )
        else:
            metrics_tracker.record_episode(
                reward=episode_reward,
                score=final_info.get("score", 0),  # Use final step info
                length=episode_length,
                epsilon=agent.epsilon
            )
        
        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_results = evaluate_agent(env, agent, num_episodes=5, deterministic=True)
            stats = metrics_tracker.get_statistics(window=100)
            
            print(f"\nEpisode {episode + 1}")
            print(f"  Recent Avg Reward: {stats.get('mean_reward', 0):.2f}")
            print(f"  Recent Avg Score: {stats.get('mean_score', 0):.2f}")
            print(f"  Recent Avg Length: {stats.get('mean_length', 0):.2f}")
            print(f"  Eval Avg Score: {eval_results['mean_score']:.2f}")
            print(f"  Eval Max Score: {eval_results['max_score']:.2f}")
            
            # Save best model
            if eval_results['mean_score'] > best_score:
                best_score = eval_results['mean_score']
                agent.save(os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_ep{episode+1}.pth'))
            metrics_tracker.save(os.path.join(checkpoint_dir, 'metrics.json'))
            plot_training_curves(
                metrics_tracker,
                save_path=os.path.join(checkpoint_dir, 'training_curves.png')
            )
    
    # Final save
    agent.save(os.path.join(checkpoint_dir, 'final_model.pth'))
    metrics_tracker.save(os.path.join(checkpoint_dir, 'metrics.json'))
    plot_training_curves(
        metrics_tracker,
        save_path=os.path.join(checkpoint_dir, 'training_curves.png')
    )
    
    print("\nTraining complete!")
    print(f"Best score: {best_score:.2f}")


def train_ppo(env, agent, config, metrics_tracker, checkpoint_dir):
    """Train PPO agent."""
    total_episodes = config['training']['total_episodes']
    eval_frequency = config['training']['eval_frequency']
    save_frequency = config['training']['save_frequency']
    update_frequency = config['training']['update_frequency']
    
    print(f"Starting PPO training for {total_episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"State representation: {config['environment']['state_representation']}")
    
    best_score = -np.inf
    
    for episode in tqdm(range(total_episodes), desc="Training"):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Collect episode
        final_info = info  # Initialize with reset info
        while not done:
            action, log_prob, value = agent.act(state, deterministic=False)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            final_info = step_info  # Update with latest step info (contains final score)
        
        # Train agent - update more frequently for PPO (every episode recommended)
        # Bootstrap value is 0.0 since episode ended
        if episode % update_frequency == 0 and len(agent.states) > 0:
            metrics = agent.train_step(next_value=0.0)
            metrics_tracker.record_episode(
                reward=episode_reward,
                score=final_info.get("score", 0),  # Use final step info
                length=episode_length,
                loss=metrics.get("loss", None)
            )
        else:
            metrics_tracker.record_episode(
                reward=episode_reward,
                score=final_info.get("score", 0),  # Use final step info
                length=episode_length
            )
        
        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_results = evaluate_agent(env, agent, num_episodes=5, deterministic=True)
            stats = metrics_tracker.get_statistics(window=100)
            
            print(f"\nEpisode {episode + 1}")
            print(f"  Recent Avg Reward: {stats.get('mean_reward', 0):.2f}")
            print(f"  Recent Avg Score: {stats.get('mean_score', 0):.2f}")
            print(f"  Recent Avg Length: {stats.get('mean_length', 0):.2f}")
            print(f"  Eval Avg Score: {eval_results['mean_score']:.2f}")
            print(f"  Eval Max Score: {eval_results['max_score']:.2f}")
            
            # Save best model
            if eval_results['mean_score'] > best_score:
                best_score = eval_results['mean_score']
                agent.save(os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_ep{episode+1}.pth'))
            metrics_tracker.save(os.path.join(checkpoint_dir, 'metrics.json'))
            plot_training_curves(
                metrics_tracker,
                save_path=os.path.join(checkpoint_dir, 'training_curves.png')
            )
    
    # Final save
    agent.save(os.path.join(checkpoint_dir, 'final_model.pth'))
    metrics_tracker.save(os.path.join(checkpoint_dir, 'metrics.json'))
    plot_training_curves(
        metrics_tracker,
        save_path=os.path.join(checkpoint_dir, 'training_curves.png')
    )
    
    print("\nTraining complete!")
    print(f"Best score: {best_score:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train Snake RL Agent')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/snake_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(
        config['training']['checkpoint_dir'],
        config['training']['experiment_name']
    )
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Create environment
    env = SnakeEnv(
        grid_size=config['environment']['grid_size'],
        state_representation=config['environment']['state_representation'],
        initial_length=config['environment']['initial_length'],
        reward_food=config['environment']['reward_food'],
        reward_death=config['environment']['reward_death'],
        reward_step=config['environment']['reward_step'],
        reward_distance=config['environment']['reward_distance']
    )
    
    # Determine state shape
    obs_space = env.observation_space
    if hasattr(obs_space, 'shape'):
        state_shape = obs_space.shape
    else:
        state_shape = (obs_space.n,)
    
    # Create agent
    algorithm = config['training']['algorithm'].lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if algorithm == "dqn":
        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=env.action_space.n,
            learning_rate=config['dqn']['learning_rate'],
            gamma=config['dqn']['gamma'],
            epsilon_start=config['dqn']['epsilon_start'],
            epsilon_end=config['dqn']['epsilon_end'],
            epsilon_decay=config['dqn']['epsilon_decay'],
            replay_buffer_size=config['dqn']['replay_buffer_size'],
            batch_size=config['dqn']['batch_size'],
            target_update_frequency=config['dqn']['target_update_frequency'],
            hidden_sizes=config['dqn']['network'],
            activation=config['dqn']['activation'],
            state_representation=config['environment']['state_representation'],
            device=device,
            seed=args.seed
        )
    elif algorithm == "ppo":
        agent = PPODiscreteAgent(
            state_shape=state_shape,
            num_actions=env.action_space.n,
            learning_rate=config['ppo']['learning_rate'],
            gamma=config['ppo']['gamma'],
            gae_lambda=config['ppo']['gae_lambda'],
            clip_epsilon=config['ppo']['clip_epsilon'],
            value_coef=config['ppo']['value_coef'],
            entropy_coef=config['ppo']['entropy_coef'],
            max_grad_norm=config['ppo']['max_grad_norm'],
            update_epochs=config['ppo']['update_epochs'],
            batch_size=config['ppo']['batch_size'],
            hidden_sizes=config['ppo']['network'],
            activation=config['ppo']['activation'],
            state_representation=config['environment']['state_representation'],
            device=device,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Train
    if algorithm == "dqn":
        train_dqn(env, agent, config, metrics_tracker, checkpoint_dir)
    else:
        train_ppo(env, agent, config, metrics_tracker, checkpoint_dir)


if __name__ == "__main__":
    main()
