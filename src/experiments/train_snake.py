"""
Main training script for Snake RL project.
Supports both DQN and PPO algorithms.
"""

import sys
import os
import yaml
import argparse
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from environments.snake_env import snake_env_from_config
from agents import DQNAgent, PPODiscreteAgent
from utils.training import MetricsTracker, evaluate_agent, create_checkpoint_dir, get_device
from utils.visualization import plot_training_curves


def train_dqn(env, agent, config, metrics_tracker, checkpoint_dir) -> None:
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


        final_info = info
        while not done:
            action = agent.act(state, deterministic=False)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated


            agent.store_transition(state, action, reward, next_state, done)

            episode_reward += reward
            episode_length += 1
            state = next_state
            final_info = step_info


        if len(agent.replay_buffer) >= agent.batch_size:
            if episode % update_frequency == 0:
                metrics = agent.train_step()
                metrics_tracker.record_episode(
                    reward=episode_reward,
                    score=final_info.get("score", 0),
                    length=episode_length,
                    loss=metrics.get("loss", None),
                    epsilon=metrics.get("epsilon", None)
                )
            else:
                metrics_tracker.record_episode(
                    reward=episode_reward,
                    score=final_info.get("score", 0),
                    length=episode_length,
                    epsilon=agent.epsilon
                )
        else:
            metrics_tracker.record_episode(
                reward=episode_reward,
                score=final_info.get("score", 0),
                length=episode_length,
                epsilon=agent.epsilon
            )


        if (episode + 1) % eval_frequency == 0:
            eval_results = evaluate_agent(env, agent, num_episodes=5, deterministic=True)
            stats = metrics_tracker.get_statistics(window=100)

            print(f"\nEpisode {episode + 1}")
            print(f"  Recent Avg Reward: {stats.get('mean_reward', 0):.2f}")
            print(f"  Recent Avg Score: {stats.get('mean_score', 0):.2f}")
            print(f"  Recent Avg Length: {stats.get('mean_length', 0):.2f}")
            print(f"  Eval Avg Score: {eval_results['mean_score']:.2f}")
            print(f"  Eval Max Score: {eval_results['max_score']:.2f}")


            if eval_results['mean_score'] > best_score:
                best_score = eval_results['mean_score']
                agent.save(os.path.join(checkpoint_dir, 'best_model.pth'))


        if (episode + 1) % save_frequency == 0:
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_ep{episode+1}.pth'))
            metrics_tracker.save(os.path.join(checkpoint_dir, 'metrics.json'))
            plot_training_curves(
                metrics_tracker,
                save_path=os.path.join(checkpoint_dir, 'training_curves.png')
            )


    agent.save(os.path.join(checkpoint_dir, 'final_model.pth'))
    metrics_tracker.save(os.path.join(checkpoint_dir, 'metrics.json'))
    plot_training_curves(
        metrics_tracker,
        save_path=os.path.join(checkpoint_dir, 'training_curves.png')
    )

    print("\nTraining complete!")
    print(f"Best score: {best_score:.2f}")


def train_ppo(env, agent, config, metrics_tracker, checkpoint_dir) -> None:
    """Train PPO agent with optional rollout-based updates."""
    total_episodes = config['training']['total_episodes']
    eval_frequency = config['training']['eval_frequency']
    save_frequency = config['training']['save_frequency']
    update_frequency = config['training']['update_frequency']
    ppo_rollout_steps = config['training'].get('ppo_rollout_steps', 0)

    print(f"Starting PPO training for {total_episodes} episodes...")
    print(f"Device: {agent.device}")
    if ppo_rollout_steps > 0:
        print(f"PPO rollout: {ppo_rollout_steps} steps per update")

    best_score = -np.inf
    episode = 0
    pbar = tqdm(total=total_episodes, desc="Training")

    while episode < total_episodes:
        steps_collected = 0
        rollout_episodes = []

        while (ppo_rollout_steps <= 0 or steps_collected < ppo_rollout_steps) and episode < total_episodes:
            state, info = env.reset()
            ep_reward, ep_length = 0, 0
            done = False
            final_info = info
            while not done:
                action, log_prob, value = agent.act(state, deterministic=False)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, log_prob, value, done)
                ep_reward += reward
                ep_length += 1
                steps_collected += 1
                state = next_state
                final_info = step_info
                if ppo_rollout_steps > 0 and steps_collected >= ppo_rollout_steps:
                    break

            rollout_episodes.append((ep_reward, final_info.get("score", 0), ep_length))
            episode += 1
            pbar.update(1)
            if ppo_rollout_steps > 0 and steps_collected >= ppo_rollout_steps:
                break

        should_update = (ppo_rollout_steps > 0) or (episode % update_frequency == 0)
        metrics = {}
        if should_update and len(agent.states) > 0:
            next_value = 0.0 if done else agent.act(state, deterministic=True)[2]
            metrics = agent.train_step(next_value=next_value)
        for r, s, l in rollout_episodes:
            metrics_tracker.record_episode(
                reward=r, score=s, length=l,
                loss=metrics.get("loss") if metrics else None
            )


        if episode > 0 and episode % eval_frequency == 0:
            eval_results = evaluate_agent(env, agent, num_episodes=5, deterministic=True)
            stats = metrics_tracker.get_statistics(window=100)

            print(f"\nEpisode {episode}")
            print(f"  Recent Avg Reward: {stats.get('mean_reward', 0):.2f}")
            print(f"  Recent Avg Score: {stats.get('mean_score', 0):.2f}")
            print(f"  Recent Avg Length: {stats.get('mean_length', 0):.2f}")
            print(f"  Eval Avg Score: {eval_results['mean_score']:.2f}")
            print(f"  Eval Max Score: {eval_results['max_score']:.2f}")


            if eval_results['mean_score'] > best_score:
                best_score = eval_results['mean_score']
                agent.save(os.path.join(checkpoint_dir, 'best_model.pth'))


        if episode > 0 and episode % save_frequency == 0:
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pth'))
            metrics_tracker.save(os.path.join(checkpoint_dir, 'metrics.json'))
            plot_training_curves(
                metrics_tracker,
                save_path=os.path.join(checkpoint_dir, 'training_curves.png')
            )


    agent.save(os.path.join(checkpoint_dir, 'final_model.pth'))
    metrics_tracker.save(os.path.join(checkpoint_dir, 'metrics.json'))
    plot_training_curves(
        metrics_tracker,
        save_path=os.path.join(checkpoint_dir, 'training_curves.png')
    )

    print("\nTraining complete!")
    print(f"Best score: {best_score:.2f}")


def watch_agent(env, agent, num_episodes: int = 3) -> None:
    """Roll out deterministic episodes and print score, steps, and return per episode."""
    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        steps, total_reward = 0, 0
        while not done:
            result = agent.act(state, deterministic=True)
            action = result[0] if isinstance(result, tuple) else result
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state
        print(f"Episode {ep + 1}: Score={env.score}, Steps={steps}, Reward={total_reward:.2f}")


def main() -> None:
    """CLI entry: train DQN or PPO from YAML, or load a checkpoint and watch."""
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
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch trained agent play instead of training'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (required for --watch, e.g. checkpoints/snake/best_model.pth)'
    )
    args = parser.parse_args()


    project_root = Path(__file__).parent.parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)


    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


    checkpoint_dir = None
    if not args.watch:
        checkpoint_dir = create_checkpoint_dir(
            config['training']['checkpoint_dir'],
            config['training']['experiment_name']
        )
        print(f"Checkpoint directory: {checkpoint_dir}")


    env = snake_env_from_config(config['environment'])


    obs_space = env.observation_space
    if hasattr(obs_space, 'shape'):
        state_shape = obs_space.shape
    else:
        state_shape = (obs_space.n,)


    algorithm = config['training']['algorithm'].lower()
    device = get_device()

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

    if args.watch:
        if not args.checkpoint:
            print("Error: --watch requires --checkpoint with path to trained model")
            sys.exit(1)
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        if not ckpt_path.exists():
            print(f"Error: Checkpoint not found at {ckpt_path}")
            sys.exit(1)
        args.checkpoint = str(ckpt_path)
        agent.load(args.checkpoint)
        if hasattr(agent, 'eval'):
            agent.eval()
        print(f"Loaded {args.checkpoint}")
        watch_agent(env, agent, num_episodes=5)
        return


    metrics_tracker = MetricsTracker()
    if algorithm == "dqn":
        train_dqn(env, agent, config, metrics_tracker, checkpoint_dir)
    else:
        train_ppo(env, agent, config, metrics_tracker, checkpoint_dir)


if __name__ == "__main__":
    main()
