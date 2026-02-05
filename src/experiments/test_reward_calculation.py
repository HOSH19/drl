"""
Test reward calculation step by step to find bugs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments import SnakeEnv
import numpy as np

def test_reward_calculation():
    """Test reward calculation in detail."""
    print("=" * 70)
    print("Testing Reward Calculation")
    print("=" * 70)
    
    env = SnakeEnv(
        grid_size=10,
        reward_food=50.0,
        reward_death=-10.0,
        reward_step=0.0,
        reward_distance=1.0
    )
    
    # Reset and manually set up scenario
    state, info = env.reset()
    env.snake = [(5, 5)]
    env.food = (3, 5)  # Food is 2 rows UP
    env.direction = env.RIGHT
    
    print("\nInitial Setup:")
    print(f"  Snake head: {env.snake[0]}")
    print(f"  Food: {env.food}")
    print(f"  Direction: {['UP', 'DOWN', 'LEFT', 'RIGHT'][env.direction]}")
    print(f"  Distance to food: {env._distance_to_food(env.snake[0])}")
    
    # Test 1: Move toward food (UP)
    print("\n" + "-" * 70)
    print("Test 1: Move UP (toward food)")
    print("-" * 70)
    
    old_head = env.snake[0]
    old_dist = env._distance_to_food(old_head)
    print(f"  Before step:")
    print(f"    Old head: {old_head}")
    print(f"    Old distance: {old_dist}")
    print(f"    Food position: {env.food}")
    
    action = env.UP
    print(f"  Action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")
    
    # Manually trace through step logic
    dx, dy = env.directions[action]
    print(f"  Direction vector: ({dx}, {dy})")
    new_head = (old_head[0] + dx, old_head[1] + dy)
    print(f"  Calculated new_head: {new_head}")
    
    # Execute step
    next_state, reward, terminated, truncated, step_info = env.step(action)
    
    print(f"  After step:")
    print(f"    Actual new head: {env.snake[0]}")
    print(f"    New distance: {env._distance_to_food(env.snake[0])}")
    print(f"    Food eaten: {step_info.get('food_eaten', False)}")
    print(f"    Reward received: {reward:.2f}")
    print(f"    Score: {env.score}")
    
    # Calculate expected reward
    if env.snake[0] == env.food:
        expected = env.reward_food + old_dist * env.reward_distance
        print(f"  Expected reward (food eaten): {env.reward_food} + {old_dist} * {env.reward_distance} = {expected:.2f}")
    else:
        new_dist = env._distance_to_food(env.snake[0])
        expected = env.reward_step + (old_dist - new_dist) * env.reward_distance
        print(f"  Expected reward (no food): {env.reward_step} + ({old_dist} - {new_dist}) * {env.reward_distance} = {expected:.2f}")
    
    if abs(reward - expected) < 0.01:
        print("  ✅ Reward matches expected!")
    else:
        print(f"  ❌ ERROR: Reward mismatch! Expected {expected:.2f}, got {reward:.2f}")
    
    # Test 2: Move away from food
    print("\n" + "-" * 70)
    print("Test 2: Move RIGHT (perpendicular, away from food)")
    print("-" * 70)
    
    state, info = env.reset()
    env.snake = [(5, 5)]
    env.food = (3, 5)  # Food is UP
    env.direction = env.UP
    
    old_head = env.snake[0]
    old_dist = env._distance_to_food(old_head)
    print(f"  Before step:")
    print(f"    Old head: {old_head}")
    print(f"    Old distance: {old_dist}")
    
    action = env.RIGHT
    print(f"  Action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")
    
    next_state, reward, terminated, truncated, step_info = env.step(action)
    
    print(f"  After step:")
    print(f"    New head: {env.snake[0]}")
    print(f"    New distance: {env._distance_to_food(env.snake[0])}")
    print(f"    Reward received: {reward:.2f}")
    
    new_dist = env._distance_to_food(env.snake[0])
    expected = env.reward_step + (old_dist - new_dist) * env.reward_distance
    print(f"  Expected reward: {env.reward_step} + ({old_dist} - {new_dist}) * {env.reward_distance} = {expected:.2f}")
    
    if abs(reward - expected) < 0.01:
        print("  ✅ Reward matches expected!")
    else:
        print(f"  ❌ ERROR: Reward mismatch! Expected {expected:.2f}, got {reward:.2f}")
    
    # Test 3: Actually eat food
    print("\n" + "-" * 70)
    print("Test 3: Move directly to food and eat it")
    print("-" * 70)
    
    state, info = env.reset()
    env.snake = [(4, 5)]
    env.food = (3, 5)  # Food is directly UP (1 step away)
    env.direction = env.RIGHT
    
    old_head = env.snake[0]
    old_dist = env._distance_to_food(old_head)
    print(f"  Before step:")
    print(f"    Old head: {old_head}")
    print(f"    Food: {env.food}")
    print(f"    Old distance: {old_dist}")
    
    action = env.UP
    print(f"  Action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")
    
    next_state, reward, terminated, truncated, step_info = env.step(action)
    
    print(f"  After step:")
    print(f"    New head: {env.snake[0]}")
    print(f"    Food eaten: {step_info.get('food_eaten', False)}")
    print(f"    Score: {env.score}")
    print(f"    Reward received: {reward:.2f}")
    
    if step_info.get('food_eaten', False):
        expected = env.reward_food + old_dist * env.reward_distance
        print(f"  Expected reward: {env.reward_food} + {old_dist} * {env.reward_distance} = {expected:.2f}")
        if abs(reward - expected) < 0.01:
            print("  ✅ Reward matches expected!")
        else:
            print(f"  ❌ ERROR: Reward mismatch! Expected {expected:.2f}, got {reward:.2f}")
    else:
        print("  ❌ ERROR: Food should have been eaten!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_reward_calculation()
