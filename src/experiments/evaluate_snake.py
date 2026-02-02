"""
Evaluation script for Snake RL project.
Loads trained models and evaluates performance with visualization.
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from environments import SnakeEnv
from agents import DQNAgent, PPODiscreteAgent
from utils.training import evaluate_agent
from utils.visualization import plot_policy_heatmap


def create_game_video(env, agent, num_episodes=3, save_dir="./videos"):
    """
    Create video of agent playing the game.
    
    Args:
        env: Environment instance
        agent: Trained agent
        num_episodes: Number of episodes to record
        save_dir: Directory to save videos
    """
    os.makedirs(save_dir, exist_ok=True)
    agent.eval()
    
    for episode in range(num_episodes):
        frames = []
        state, info = env.reset()
        done = False
        
        while not done:
            # Render frame
            if env.render_mode == "rgb_array":
                frame = env.render()
            else:
                # Use renderer
                from environments.snake_renderer import SnakeRenderer
                renderer = SnakeRenderer(env.grid_size)
                renderer.render(
                    env.snake,
                    env.food,
                    env.score,
                    env.steps
                )
                # Capture matplotlib figure
                fig = plt.gcf()
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            frames.append(frame)
            
            # Take action
            if isinstance(agent, DQNAgent):
                action = agent.act(state, deterministic=True)
            else:
                action, _, _ = agent.act(state, deterministic=True)
            
            state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
        
        # Save video (simplified - save as image sequence)
        episode_dir = os.path.join(save_dir, f"episode_{episode+1}")
        os.makedirs(episode_dir, exist_ok=True)
        
        for idx, frame in enumerate(frames):
            plt.imsave(
                os.path.join(episode_dir, f"frame_{idx:04d}.png"),
                frame
            )
        
        print(f"Episode {episode+1} saved to {episode_dir} ({len(frames)} frames)")
    
    agent.train()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Snake RL Agent')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/snake_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10,
        help='Number of episodes to evaluate'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes during evaluation'
    )
    parser.add_argument(
        '--save_videos',
        action='store_true',
        help='Save video of episodes'
    )
    parser.add_argument(
        '--policy_heatmap',
        action='store_true',
        help='Generate policy heatmap'
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = SnakeEnv(
        grid_size=config['environment']['grid_size'],
        state_representation=config['environment']['state_representation'],
        initial_length=config['environment']['initial_length'],
        reward_food=config['environment']['reward_food'],
        reward_death=config['environment']['reward_death'],
        reward_step=config['environment']['reward_step'],
        reward_distance=config['environment']['reward_distance'],
        render_mode="human" if args.render else None
    )
    
    # Determine state shape
    obs_space = env.observation_space
    if hasattr(obs_space, 'shape'):
        state_shape = obs_space.shape
    else:
        state_shape = (obs_space.n,)
    
    # Determine algorithm from checkpoint or config
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    algorithm = checkpoint.get('algorithm', config['training']['algorithm'].lower())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent
    if algorithm == "dqn" or 'q_network_state_dict' in checkpoint:
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
            device=device
        )
    else:
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
            device=device
        )
    
    # Load checkpoint
    agent.load(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")
    
    # Evaluate
    print(f"\nEvaluating agent for {args.num_episodes} episodes...")
    results = evaluate_agent(
        env,
        agent,
        num_episodes=args.num_episodes,
        render=args.render,
        deterministic=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Max Reward: {results['max_reward']:.2f}")
    print(f"Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"Max Score: {results['max_score']:.2f}")
    print(f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print("="*50)
    
    # Save videos if requested
    if args.save_videos:
        print("\nCreating videos...")
        create_game_video(env, agent, num_episodes=3, save_dir="./videos")
    
    # Generate policy heatmap if requested
    if args.policy_heatmap:
        print("\nGenerating policy heatmap...")
        plot_policy_heatmap(
            agent,
            env,
            grid_size=config['environment']['grid_size'],
            save_path="./policy_heatmap.png"
        )
        print("Policy heatmap saved to ./policy_heatmap.png")


if __name__ == "__main__":
    main()
