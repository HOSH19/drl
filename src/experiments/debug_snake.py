"""
Debug script to verify Snake environment and check if agent can learn.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from environments import SnakeEnv

def test_environment():
    """Test if environment works correctly."""
    print("="*50)
    print("Testing Snake Environment")
    print("="*50)
    
    env = SnakeEnv(grid_size=20, reward_step=0.0)
    
    # Test 1: Can random actions eat food?
    print("\n1. Testing random actions...")
    scores = []
    for episode in range(10):
        state, info = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = env.action_space.sample()
            state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            steps += 1
        scores.append(step_info.get("score", 0))
        print(f"  Episode {episode+1}: Score={step_info.get('score', 0)}, Steps={steps}")
    
    print(f"\n  Random agent average score: {np.mean(scores):.2f}")
    if np.mean(scores) > 0:
        print("  ✓ Environment works - random actions can eat food")
    else:
        print("  ✗ WARNING: Random actions never ate food - check environment!")
    
    # Test 2: Check state representation
    print("\n2. Testing state representation...")
    state, info = env.reset()
    print(f"  State shape: {state.shape}")
    print(f"  State type: {type(state)}")
    print(f"  State sample: {state[:5] if len(state) > 5 else state}")
    
    # Test 3: Check food spawning
    print("\n3. Testing food spawning...")
    for i in range(5):
        state, info = env.reset()
        # Get food position from state (for feature representation)
        if env.state_representation == "feature":
            food_x = int(state[2] * env.grid_size)
            food_y = int(state[3] * env.grid_size)
            print(f"  Episode {i+1}: Food at ({food_x}, {food_y}), Snake head at ({int(state[0]*env.grid_size)}, {int(state[1]*env.grid_size)})")
        print(f"  Score: {info.get('score', 0)}, Snake length: {info.get('snake_length', 0)}")
    
    # Test 4: Check rewards
    print("\n4. Testing rewards...")
    state, info = env.reset()
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        state, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: Reward={reward:.2f}, Score={step_info.get('score', 0)}, Done={terminated or truncated}")
        if terminated or truncated:
            break
    
    print(f"\n  Total reward over 10 steps: {total_reward:.2f}")
    print("\n" + "="*50)

if __name__ == "__main__":
    test_environment()
