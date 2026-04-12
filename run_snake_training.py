#!/usr/bin/env python3
"""
Snake Game Deep RL - Training Script

Converts the Colab notebook into a runnable Python script.
Works locally or on Google Colab (set PROJECT_DIR and run setup steps as needed).

Usage:
    python run_snake_training.py                    # Run full training
    python run_snake_training.py --watch            # Watch trained agent play
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm




PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)


def setup_project_structure():
    """Create project directories."""
    os.makedirs(f"{PROJECT_DIR}/src/environments", exist_ok=True)
    os.makedirs(f"{PROJECT_DIR}/src/agents", exist_ok=True)
    os.makedirs(f"{PROJECT_DIR}/src/networks", exist_ok=True)
    os.makedirs(f"{PROJECT_DIR}/src/utils", exist_ok=True)
    os.makedirs(f"{PROJECT_DIR}/src/experiments", exist_ok=True)
    os.makedirs(f"{PROJECT_DIR}/configs", exist_ok=True)
    os.makedirs(f"{PROJECT_DIR}/logs/snake", exist_ok=True)
    os.makedirs(f"{PROJECT_DIR}/checkpoints/snake", exist_ok=True)
    print(f"Project directory: {PROJECT_DIR}")


def create_config():
    """Create configuration file."""
    config_content = {
        "environment": {
            "grid_size": 15,
            "state_representation": "feature",
            "initial_length": 3,
            "reward_food": 1.0,
            "reward_death": -1.0,
            "reward_step": -0.005,
            "reward_distance": 0.15,
        },
        "dqn": {
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.9995,
            "replay_buffer_size": 50000,
            "batch_size": 32,
            "target_update_frequency": 500,
            "network": [64, 64],
            "activation": "relu",
        },
        "ppo": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.02,
            "max_grad_norm": 0.5,
            "update_epochs": 10,
            "batch_size": 64,
            "network": [128, 128, 64],
            "activation": "relu",
        },
        "training": {
            "algorithm": "ppo",
            "total_episodes": 3000,
            "eval_frequency": 100,
            "save_frequency": 500,
            "update_frequency": 1,
            "ppo_rollout_steps": 512,
            "log_dir": f"{PROJECT_DIR}/logs/snake",
            "checkpoint_dir": f"{PROJECT_DIR}/checkpoints/snake",
            "experiment_name": "snake_ppo_colab",
        },
        "evaluation": {
            "num_episodes": 10,
            "render": False,
            "save_videos": False,
        },
    }

    config_path = f"{PROJECT_DIR}/configs/snake_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    print(f"Config created at: {config_path}")
    return config_path


def load_config(config_path):
    """Load config from file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(config_path: str):
    """Run the full training pipeline."""

    from src.environments.snake_env import snake_env_from_config
    from src.agents import DQNAgent, PPODiscreteAgent
    from src.utils.training import MetricsTracker, evaluate_agent

    config = load_config(config_path)


    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


    env = snake_env_from_config(config["environment"])

    state_shape = (12,)
    algorithm = config["training"]["algorithm"].lower()
    from src.utils.training import get_device
    device = get_device()


    if algorithm == "dqn":
        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=env.action_space.n,
            learning_rate=config["dqn"]["learning_rate"],
            gamma=config["dqn"]["gamma"],
            epsilon_start=config["dqn"]["epsilon_start"],
            epsilon_end=config["dqn"]["epsilon_end"],
            epsilon_decay=config["dqn"]["epsilon_decay"],
            replay_buffer_size=config["dqn"]["replay_buffer_size"],
            batch_size=config["dqn"]["batch_size"],
            target_update_frequency=config["dqn"]["target_update_frequency"],
            hidden_sizes=config["dqn"]["network"],
            activation=config["dqn"]["activation"],
            state_representation=config["environment"]["state_representation"],
            device=device,
            seed=42,
        )
    else:
        agent = PPODiscreteAgent(
            state_shape=state_shape,
            num_actions=env.action_space.n,
            learning_rate=config["ppo"]["learning_rate"],
            gamma=config["ppo"]["gamma"],
            gae_lambda=config["ppo"]["gae_lambda"],
            clip_epsilon=config["ppo"]["clip_epsilon"],
            value_coef=config["ppo"]["value_coef"],
            entropy_coef=config["ppo"]["entropy_coef"],
            max_grad_norm=config["ppo"]["max_grad_norm"],
            update_epochs=config["ppo"]["update_epochs"],
            batch_size=config["ppo"]["batch_size"],
            hidden_sizes=config["ppo"]["network"],
            activation=config["ppo"]["activation"],
            state_representation=config["environment"]["state_representation"],
            device=device,
            seed=42,
        )

    print(f"Algorithm: {algorithm.upper()}, Device: {device}")


    metrics_tracker = MetricsTracker()
    total_episodes = config["training"]["total_episodes"]
    eval_frequency = config["training"]["eval_frequency"]
    save_frequency = config["training"]["save_frequency"]
    update_frequency = config["training"]["update_frequency"]
    ppo_rollout_steps = config["training"].get("ppo_rollout_steps", 0)

    print(f"Starting training for {total_episodes} episodes...")
    if algorithm == "ppo" and ppo_rollout_steps > 0:
        print(f"PPO rollout: {ppo_rollout_steps} steps per update")

    best_score = -np.inf
    episode = 0
    pbar = tqdm(total=total_episodes, desc="Training")

    while episode < total_episodes:
        if algorithm == "dqn":
            state, info = env.reset()
            episode_reward, episode_length = 0, 0
            done = False
            while not done:
                action = agent.act(state, deterministic=False)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, next_state, done)
                episode_reward += reward
                episode_length += 1
                state = next_state

            if len(agent.replay_buffer) >= agent.batch_size and episode % update_frequency == 0:
                metrics = agent.train_step()
                metrics_tracker.record_episode(
                    reward=episode_reward,
                    score=env.score,
                    length=episode_length,
                    loss=metrics.get("loss"),
                    epsilon=metrics.get("epsilon"),
                )
            else:
                metrics_tracker.record_episode(
                    reward=episode_reward,
                    score=env.score,
                    length=episode_length,
                    epsilon=agent.epsilon,
                )
            episode += 1
            pbar.update(1)

        else:
            steps_collected = 0
            rollout_episodes = []

            while (ppo_rollout_steps <= 0 or steps_collected < ppo_rollout_steps) and episode < total_episodes:
                state, info = env.reset()
                ep_reward, ep_length = 0, 0
                done = False
                while not done:
                    action, log_prob, value = agent.act(state, deterministic=False)
                    next_state, reward, terminated, truncated, step_info = env.step(action)
                    done = terminated or truncated
                    agent.store_transition(state, action, reward, log_prob, value, done)
                    ep_reward += reward
                    ep_length += 1
                    steps_collected += 1
                    state = next_state
                    if ppo_rollout_steps > 0 and steps_collected >= ppo_rollout_steps:
                        break

                rollout_episodes.append((ep_reward, env.score, ep_length))
                episode += 1
                pbar.update(1)
                if ppo_rollout_steps > 0 and steps_collected >= ppo_rollout_steps:
                    break

            if len(agent.states) > 0:
                next_value = 0.0 if done else agent.act(state, deterministic=True)[2]
                metrics = agent.train_step(next_value=next_value)
                for r, s, l in rollout_episodes:
                    metrics_tracker.record_episode(reward=r, score=s, length=l, loss=metrics.get("loss"))

        if episode > 0 and episode % eval_frequency == 0:
            eval_results = evaluate_agent(env, agent, num_episodes=5, deterministic=True)
            stats = metrics_tracker.get_statistics(window=100)

            print(f"\nEpisode {episode}")
            print(f"  Recent Avg Reward: {stats.get('mean_reward', 0):.2f}")
            print(f"  Recent Avg Score: {stats.get('mean_score', 0):.2f}")
            print(f"  Eval Avg Score: {eval_results['mean_score']:.2f}")

            if eval_results["mean_score"] > best_score:
                best_score = eval_results["mean_score"]
                checkpoint_path = f"{PROJECT_DIR}/checkpoints/snake/best_model.pth"
                agent.save(checkpoint_path)
                print(f"  ✓ Saved best model (score: {best_score:.2f})")

        if episode > 0 and episode % save_frequency == 0:
            checkpoint_path = f"{PROJECT_DIR}/checkpoints/snake/checkpoint_ep{episode}.pth"
            agent.save(checkpoint_path)
            metrics_tracker.save(f"{PROJECT_DIR}/checkpoints/snake/metrics.json")
            print(f"  ✓ Saved checkpoint at episode {episode}")

    pbar.close()
    print("\n✓ Training complete!")


    eval_results = evaluate_agent(env, agent, num_episodes=10, deterministic=True)
    print("\n" + "=" * 50)
    print("Final Evaluation Results")
    print("=" * 50)
    print(f"Mean Score: {eval_results['mean_score']:.2f} ± {eval_results['std_score']:.2f}")
    print(f"Max Score: {eval_results['max_score']:.2f}")
    print("=" * 50)

    return agent, env, config


def watch_agent_play(env, agent, config, num_episodes=3, delay=0.1, max_steps=500):
    """Watch trained agent play Snake (text-based, no GUI)."""
    from src.agents import DQNAgent

    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        steps = 0
        total_reward = 0

        print(f"\n--- Episode {ep + 1} ---")

        while not done and steps < max_steps:
            result = agent.act(state, deterministic=True)
            action = result[0] if isinstance(result, tuple) else result

            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state

        print(f"  Score: {env.score} | Steps: {steps} | Reward: {total_reward:.2f}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Snake RL Training")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch trained agent play instead of training",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project directory (default: script location)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: configs/snake_config.yaml)",
    )
    args = parser.parse_args()

    global PROJECT_DIR
    if args.project_dir:
        PROJECT_DIR = args.project_dir
        sys.path.insert(0, PROJECT_DIR)

    setup_project_structure()

    config_path = args.config or f"{PROJECT_DIR}/configs/snake_config.yaml"
    if not os.path.exists(config_path):
        create_config()
    else:
        print(f"Using config: {config_path}")

    if args.watch:
        from src.environments.snake_env import snake_env_from_config
        from src.agents import DQNAgent, PPODiscreteAgent

        config = load_config(config_path)
        env = snake_env_from_config(config["environment"])

        obs_space = env.observation_space
        state_shape = obs_space.shape if hasattr(obs_space, "shape") else (obs_space.n,)
        from src.utils.training import get_device
        device = get_device()
        algorithm = config["training"]["algorithm"].lower()

        if algorithm == "dqn":
            agent = DQNAgent(
                state_shape=state_shape,
                num_actions=env.action_space.n,
                learning_rate=config["dqn"]["learning_rate"],
                gamma=config["dqn"]["gamma"],
                epsilon_start=config["dqn"]["epsilon_start"],
                epsilon_end=config["dqn"]["epsilon_end"],
                epsilon_decay=config["dqn"]["epsilon_decay"],
                replay_buffer_size=config["dqn"]["replay_buffer_size"],
                batch_size=config["dqn"]["batch_size"],
                target_update_frequency=config["dqn"]["target_update_frequency"],
                hidden_sizes=config["dqn"]["network"],
                activation=config["dqn"]["activation"],
                state_representation=config["environment"]["state_representation"],
                device=device,
            )
        else:
            agent = PPODiscreteAgent(
                state_shape=state_shape,
                num_actions=env.action_space.n,
                learning_rate=config["ppo"]["learning_rate"],
                gamma=config["ppo"]["gamma"],
                gae_lambda=config["ppo"]["gae_lambda"],
                clip_epsilon=config["ppo"]["clip_epsilon"],
                value_coef=config["ppo"]["value_coef"],
                entropy_coef=config["ppo"]["entropy_coef"],
                max_grad_norm=config["ppo"]["max_grad_norm"],
                update_epochs=config["ppo"]["update_epochs"],
                batch_size=config["ppo"]["batch_size"],
                hidden_sizes=config["ppo"]["network"],
                activation=config["ppo"]["activation"],
                state_representation=config["environment"]["state_representation"],
                device=device,
            )

        checkpoint_path = f"{PROJECT_DIR}/checkpoints/snake/best_model.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Error: No trained model at {checkpoint_path}")
            print("Run training first: python run_snake_training.py")
            sys.exit(1)

        agent.load(checkpoint_path)
        if hasattr(agent, "eval"):
            agent.eval()

        print(f"Loaded agent from {checkpoint_path}")
        watch_agent_play(env, agent, config, num_episodes=3, delay=0.05)
    else:
        run_training(config_path)


if __name__ == "__main__":
    main()
