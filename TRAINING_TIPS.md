# Training Tips for Snake RL

## Problem: Zero Scores (Agent Never Eats Food)

If your training shows scores of 0, the agent is dying before reaching food. Here are solutions:

## Quick Fixes

### 1. Use Easier Configuration
Use `snake_config_easy.yaml` which has:
- Smaller grid (15x15 instead of 20x20)
- No step penalty (reward_step: 0.0)
- Distance reward enabled (guides agent toward food)
- Slower epsilon decay (more exploration)
- Simpler network

### 2. Increase Episodes
- **Minimum**: 1000 episodes for basic learning
- **Recommended**: 2000-5000 episodes for good performance
- **200 episodes is too few** - agent hasn't learned yet

### 3. Adjust Rewards
```yaml
reward_food: 10.0      # Keep positive
reward_death: -5.0     # Less harsh (was -10.0)
reward_step: 0.0       # Remove penalty initially
reward_distance: 0.1   # Guide agent toward food
```

### 4. Slow Down Epsilon Decay
```yaml
epsilon_start: 1.0
epsilon_end: 0.05      # Keep exploring longer
epsilon_decay: 0.998   # Slower decay (was 0.995)
```

### 5. Check Training Progress
Add debug prints to see what's happening:

```python
# In training loop, add:
if episode % 10 == 0:
    print(f"Episode {episode}: Score={info.get('score', 0)}, "
          f"Length={episode_length}, Epsilon={agent.epsilon:.3f}")
```

## Common Issues

### Issue 1: Agent Dies Immediately
**Symptoms**: Episode lengths < 10 steps, scores always 0

**Solutions**:
- Remove step penalty: `reward_step: 0.0`
- Reduce death penalty: `reward_death: -5.0`
- Increase epsilon: `epsilon_end: 0.1`
- Use smaller grid: `grid_size: 15`

### Issue 2: Agent Moves Randomly Forever
**Symptoms**: Long episodes but no food eaten

**Solutions**:
- Enable distance reward: `reward_distance: 0.1`
- Increase food reward: `reward_food: 20.0`
- Check if state representation is correct

### Issue 3: Loss Not Decreasing
**Symptoms**: Volatile loss, no improvement

**Solutions**:
- Reduce learning rate: `learning_rate: 5e-5`
- Increase batch size: `batch_size: 128`
- Update more frequently: `update_frequency: 2`
- Check if replay buffer has enough samples

### Issue 4: Scores Stay at 0 After Many Episodes
**Symptoms**: 1000+ episodes, still 0 scores

**Solutions**:
- Use curriculum learning (start easy, increase difficulty)
- Check environment logic (is food spawning correctly?)
- Try different state representation
- Visualize agent behavior to see what it's doing

## Recommended Training Schedule

### Phase 1: Exploration (Episodes 0-500)
- High epsilon (0.8-1.0)
- No step penalty
- Distance reward enabled
- Small grid (15x15)

### Phase 2: Learning (Episodes 500-2000)
- Decreasing epsilon (0.5-0.1)
- Small step penalty (-0.05)
- Distance reward enabled
- Medium grid (20x20)

### Phase 3: Refinement (Episodes 2000+)
- Low epsilon (0.05-0.01)
- Step penalty (-0.1)
- No distance reward
- Full grid (20x20)

## Debugging Checklist

- [ ] Check if food is spawning: Print food position
- [ ] Check if agent can see food: Print state features
- [ ] Check episode lengths: Should be > 20 steps
- [ ] Check epsilon value: Should start at 1.0
- [ ] Check replay buffer size: Should have > batch_size samples
- [ ] Check rewards: Should see positive rewards occasionally
- [ ] Visualize: Watch agent play to see what it's doing

## Expected Progress

### After 200 Episodes
- Scores: 0-1 (agent might eat food occasionally)
- Episode length: 20-50 steps
- Reward: Mostly negative, some spikes

### After 1000 Episodes
- Scores: 0-3 (agent eats food sometimes)
- Episode length: 30-100 steps
- Reward: Improving trend

### After 5000 Episodes
- Scores: 2-10+ (agent consistently eats food)
- Episode length: 50-200+ steps
- Reward: Positive trend, less volatile

## Quick Test

Run this to verify environment works:

```python
env = SnakeEnv(grid_size=15, reward_step=0.0)
state, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random actions
    state, reward, done, _, info = env.step(action)
    if info.get('score', 0) > 0:
        print(f"Food eaten! Score: {info['score']}")
        break
    if done:
        state, info = env.reset()
```

If random actions can't eat food, there's an environment bug.
