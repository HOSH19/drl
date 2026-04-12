# Snake Game Deep RL Project

A deep reinforcement learning project for training agents to play Snake using DQN and PPO algorithms.

## Features

- **Custom Snake Environment**: Gymnasium-compatible with relative direction-to-food features for easier learning
- **Algorithms**: DQN and PPO (with rollout-based updates)
- **Apple Silicon (M1/M2/M3)**: Automatic MPS device detection for GPU acceleration
- **State Representations**: Feature vector, grid, or image

## Project Structure

```
drl/
├── src/
│   ├── environments/     # Snake game environment
│   ├── agents/           # DQN and PPO agents
│   ├── networks/         # Neural network architectures
│   ├── utils/            # Training utilities, visualization
│   └── experiments/      # train_snake.py, evaluate_snake.py
├── configs/              # snake_config.yaml
├── notebooks/
│   ├── snake_experiments.ipynb   # Interactive training & experiments
│   ├── watch_agent_play.ipynb    # Watch trained agent
│   └── colab_snake_setup.ipynb   # Google Colab setup
├── run_snake_training.py  # Quick-start script (flat checkpoints)
└── requirements.txt
```

## Installation

```bash
cd drl
pip install -r requirements.txt
```

**Apple Silicon (M2/M3)**: PyTorch will automatically use MPS when available. No extra setup needed.

## Quick Start (Local / M2 Mac)

### Option 1: Training script (recommended)

```bash
# Train (saves to checkpoints/snake/)
python run_snake_training.py

# Watch trained agent
python run_snake_training.py --watch --checkpoint checkpoints/snake/best_model.pth
```

### Option 2: Experiment script (timestamped checkpoints)

```bash
# Train (saves to checkpoints/snake/snake_ppo/YYYYMMDD_HHMMSS/)
python src/experiments/train_snake.py --config configs/snake_config.yaml

# Watch
python src/experiments/train_snake.py --watch --checkpoint checkpoints/snake/snake_ppo/20250217_120000/best_model.pth
```

### Option 3: Jupyter notebooks

```bash
jupyter notebook notebooks/snake_experiments.ipynb
# or
jupyter notebook notebooks/watch_agent_play.ipynb
```

Run from project root. Notebooks auto-detect project directory.

## Configuration

Edit `configs/snake_config.yaml`:

- **algorithm**: `"ppo"` or `"dqn"`
- **ppo_rollout_steps**: 512 (steps per PPO update; 0 = every episode)
- **reward_food** / **reward_death** / **reward_step**: terminal and per-step base rewards
- **reward_distance**: scales Manhattan improvement toward food each step (default in config; reduces circling)
- **training.algorithm**: `"ppo"` or `"dqn"` (project defaults to PPO in `configs/snake_config.yaml`)
- **grid_size**: 15 (smaller = easier learning)

## Colab

Use `notebooks/colab_snake_setup.ipynb` for Google Colab. Clone the repo and run the cells.

## License

MIT License
