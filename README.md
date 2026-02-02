# Snake Game Deep RL Project

A comprehensive deep reinforcement learning project for training agents to play Snake using DQN and PPO algorithms.

## Features

- **Custom Snake Environment**: Gymnasium-compatible environment with configurable state representations
- **Multiple Algorithms**: DQN (Deep Q-Network) and PPO (Proximal Policy Optimization) implementations
- **Flexible State Representations**: Support for grid, feature vector, and image-based observations
- **Comprehensive Visualization**: Training curves, policy heatmaps, and game replays
- **Experiment Notebook**: Interactive Jupyter notebook for exploration

## Project Structure

```
drl/
├── src/
│   ├── environments/
│   │   ├── snake_env.py          # Snake game environment
│   │   └── snake_renderer.py     # Visualization renderer
│   ├── agents/
│   │   ├── dqn_agent.py          # DQN agent implementation
│   │   └── ppo_discrete_agent.py # PPO agent for discrete actions
│   ├── networks/
│   │   └── dqn_network.py        # Q-network architectures
│   ├── utils/
│   │   ├── replay_buffer.py     # Experience replay buffer
│   │   ├── training.py           # Training utilities
│   │   └── visualization.py     # Visualization functions
│   └── experiments/
│       ├── train_snake.py        # Main training script
│       └── evaluate_snake.py     # Evaluation script
├── configs/
│   └── snake_config.yaml         # Configuration file
├── notebooks/
│   └── snake_experiments.ipynb   # Interactive experiments
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd drl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train a DQN agent:
```bash
python src/experiments/train_snake.py --config configs/snake_config.yaml
```

Train a PPO agent (modify config to set `algorithm: "ppo"`):
```bash
python src/experiments/train_snake.py --config configs/snake_config.yaml
```

### Evaluation

Evaluate a trained model:
```bash
python src/experiments/evaluate_snake.py \
    --checkpoint checkpoints/snake/snake_dqn/YYYYMMDD_HHMMSS/best_model.pth \
    --num_episodes 10 \
    --render
```

### Interactive Notebook

Open the Jupyter notebook for interactive exploration:
```bash
jupyter notebook notebooks/snake_experiments.ipynb
```

## Configuration

Edit `configs/snake_config.yaml` to customize:

- **Environment**: Grid size, state representation, reward shaping
- **DQN**: Learning rate, epsilon decay, network architecture
- **PPO**: Learning rate, GAE lambda, clipping parameter
- **Training**: Number of episodes, evaluation frequency

## Key RL Concepts Explored

1. **Exploration vs Exploitation**: Epsilon-greedy (DQN) vs entropy regularization (PPO)
2. **Reward Shaping**: Sparse vs dense rewards, distance-based shaping
3. **Value Functions**: Q-learning (DQN) vs policy gradients (PPO)
4. **Experience Replay**: Importance of replay buffer in DQN
5. **State Representation**: Impact of different state encodings
6. **Hyperparameter Sensitivity**: Learning rates, network sizes, reward scales

## Experiments

Try different experiments:

1. **Reward Function Comparison**: Modify reward_food, reward_death, reward_step
2. **State Representation**: Switch between "grid", "feature", and "image"
3. **Algorithm Comparison**: Train both DQN and PPO and compare performance
4. **Grid Size**: Experiment with different grid sizes (10x10, 20x20, 30x30)
5. **Network Architecture**: Adjust hidden layer sizes and activation functions

## Results

Training progress is automatically saved to:
- `checkpoints/snake/<experiment_name>/<timestamp>/`
- Training curves are saved as PNG files
- Metrics are saved as JSON files

## License

MIT License
