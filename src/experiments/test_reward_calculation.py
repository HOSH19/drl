"""
Sanity checks for SnakeEnv rewards: food / death / step, and optional distance shaping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments import SnakeEnv


def test_reward_calculation():
    print("=" * 70)
    print("Bare reward checks (food / death / step)")
    print("=" * 70)

    env = SnakeEnv(
        grid_size=10,
        reward_food=1.0,
        reward_death=-1.0,
        reward_step=-0.01,
    )

    state, info = env.reset()
    env.snake = [(5, 5)]
    env.food = (3, 5)
    env.direction = env.RIGHT

    print("\nTest 1: Ordinary step (no food)")
    old_head = env.snake[0]
    action = env.UP
    _, reward, terminated, truncated, step_info = env.step(action)
    assert not step_info.get("food_eaten", False)
    assert abs(reward - env.reward_step) < 1e-6, (reward, env.reward_step)
    print(f"  reward={reward} (expected reward_step={env.reward_step}) OK")

    print("\nTest 2: Eat food")
    state, info = env.reset()
    env.snake = [(4, 5)]
    env.food = (3, 5)
    env.direction = env.RIGHT
    _, reward, terminated, truncated, step_info = env.step(env.UP)
    assert step_info.get("food_eaten", False)
    assert abs(reward - env.reward_food) < 1e-6, (reward, env.reward_food)
    print(f"  reward={reward} (expected reward_food={env.reward_food}) OK")

    print("\nTest 3: Death (wall)")
    state, info = env.reset()
    env.snake = [(0, 5)]
    env.food = (8, 8)
    env.direction = env.RIGHT
    _, reward, terminated, truncated, step_info = env.step(env.UP)
    assert terminated
    assert abs(reward - env.reward_death) < 1e-6, (reward, env.reward_death)
    print(f"  reward={reward} (expected reward_death={env.reward_death}) OK")

    print("\n" + "=" * 70)


def test_distance_shaping():
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
    assert abs(reward - 1.0) < 1e-6, reward


if __name__ == "__main__":
    test_reward_calculation()
    test_distance_shaping()
    print("distance shaping OK")
