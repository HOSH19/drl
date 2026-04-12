"""Unit tests for :class:`environments.snake_env.SnakeEnv` rewards and termination."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from environments.snake_env import SnakeEnv


def test_reward_calculation() -> None:
    """Step, food, and wall-death rewards match configured constants."""
    env = SnakeEnv(
        grid_size=10,
        reward_food=1.0,
        reward_death=-1.0,
        reward_step=-0.01,
    )
    env.reset()
    env.snake = [(5, 5)]
    env.food = (3, 5)
    env.direction = env.RIGHT

    _, reward, _, _, step_info = env.step(env.UP)
    assert not step_info.get("food_eaten", False)
    assert abs(reward - env.reward_step) < 1e-6

    env.reset()
    env.snake = [(4, 5)]
    env.food = (3, 5)
    env.direction = env.RIGHT
    _, reward, _, _, step_info = env.step(env.UP)
    assert step_info.get("food_eaten", False)
    assert abs(reward - env.reward_food) < 1e-6

    env.reset()
    env.snake = [(0, 5)]
    env.food = (8, 8)
    env.direction = env.RIGHT
    _, reward, terminated, _, _ = env.step(env.UP)
    assert terminated
    assert abs(reward - env.reward_death) < 1e-6


def test_distance_shaping() -> None:
    """Distance shaping adds ``reward_distance * (old_dist - new_dist)`` on non-terminal steps."""
    env = SnakeEnv(
        grid_size=10,
        reward_food=1.0,
        reward_death=-1.0,
        reward_step=0.0,
        reward_distance=1.0,
    )
    env.reset()
    env.snake = [(5, 5)]
    env.food = (3, 5)
    env.direction = env.RIGHT
    _, reward, _, _, _ = env.step(env.UP)
    assert abs(reward - 1.0) < 1e-6
