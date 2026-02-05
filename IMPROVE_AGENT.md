# Why Your Agent Isn't Learning Well

## The Problem

Your agent can't move toward food because:

1. **No Distance Reward** (`reward_distance: 0.0`)
   - Agent only gets reward when it eats food (very rare)
   - No guidance about which direction to move
   - Like trying to find a needle in a haystack blindfolded

2. **Step Penalty** (`reward_step: -0.1`)
   - Penalizes survival, encouraging quick death
   - Agent learns to die fast rather than explore

3. **Sparse Rewards**
   - Only +10 for food (happens rarely)
   - -10 for death (happens often)
   - Net result: mostly negative rewards

## Solutions

### Solution 1: Use Improved Config (RECOMMENDED)

Use `snake_config_improved.yaml` which has:
- ✅ **Distance reward: 1.0** - Strong guidance toward food
- ✅ **No step penalty** - Let agent explore
- ✅ **Higher food reward: 50.0** - Bigger incentive
- ✅ **Slower epsilon decay** - Explore longer
- ✅ **More frequent updates** - Learn faster

### Solution 2: Update Your Current Config

In Colab, update your config:

```python
config['environment']['reward_distance'] = 1.0  # Enable strong distance reward
config['environment']['reward_step'] = 0.0  # Remove step penalty
config['environment']['reward_food'] = 50.0  # Increase food reward
config['dqn']['epsilon_decay'] = 0.9998  # Slower decay
config['dqn']['learning_rate'] = 3e-4  # Higher learning rate
config['training']['update_frequency'] = 2  # Update more often
config['training']['total_episodes'] = 5000  # More training
```

### Solution 3: Retrain with Better Settings

1. **Start fresh** with improved config
2. **Train for 5000+ episodes**
3. **Monitor progress** - scores should improve over time
4. **Be patient** - RL takes time to learn

## Expected Learning Curve

With improved config:
- **Episodes 0-1000**: Scores 0-1 (exploring, occasional food)
- **Episodes 1000-3000**: Scores 0-3 (learning to approach food)
- **Episodes 3000-5000**: Scores 1-5+ (consistently eating food)

## Diagnostic: Check What Agent Learned

Run this to see what the agent is doing:

```python
# Check agent's Q-values for different states
env = SnakeEnv(grid_size=15, reward_distance=1.0)
state, info = env.reset()

# Get Q-values for all actions
with torch.no_grad():
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    q_values = agent.q_network(state_tensor).cpu().numpy()[0]

print(f"Q-values for each action:")
print(f"  UP: {q_values[0]:.2f}")
print(f"  DOWN: {q_values[1]:.2f}")
print(f"  LEFT: {q_values[2]:.2f}")
print(f"  RIGHT: {q_values[3]:.2f}")
print(f"  Best action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][np.argmax(q_values)]}")

# Check if agent knows where food is
print(f"\nState features:")
print(f"  Head: ({state[0]*15:.1f}, {state[1]*15:.1f})")
print(f"  Food: ({state[2]*15:.1f}, {state[3]*15:.1f})")
print(f"  Direction: {['UP', 'DOWN', 'LEFT', 'RIGHT'][np.argmax(state[4:8])]}")
```

## Quick Fix for Current Model

If you want to improve your current model without retraining:

1. **Increase distance reward** in your config
2. **Remove step penalty**
3. **Continue training** for 2000+ more episodes
4. The agent should improve with better reward signal

## Why Distance Reward Works

Without distance reward:
- Agent gets: +50 (food) or -10 (death) or 0 (survive)
- No signal about which direction is better
- Random exploration until luck

With distance reward (1.0):
- Moving closer to food: +1 to +2 reward per step
- Moving away from food: -1 to -2 reward per step
- Clear signal: "go this way!"
- Agent learns direction quickly

## Training Tips

1. **Start with distance reward high** (1.0-2.0)
2. **Train until scores improve** (1000+ episodes minimum)
3. **Gradually reduce distance reward** as agent improves
4. **Use smaller grid initially** (15x15), then increase to 20x20

## Bottom Line

Your agent needs **distance reward** to learn. Without it, finding food is like random chance. With it, the agent gets clear feedback about which direction to move.
