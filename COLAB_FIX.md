# Fixing the Score Bug in Colab

## Option 1: Pull Latest Changes from GitHub (Recommended)

If you've pushed the fix to GitHub:

```python
# In a Colab cell, run:
%cd /content/drl
!git pull origin main

# Then re-run the import cell and training cell
```

## Option 2: Manually Edit the File in Colab

1. **Open the file editor**:
   - Click the folder icon (📁) on the left sidebar
   - Navigate to `src/experiments/train_snake.py`
   - Click on it to open

2. **Find the training loop** (around line 45-80)

3. **Change this**:
```python
# Collect episode
while not done:
    action = agent.act(state, deterministic=False)
    next_state, reward, terminated, truncated, step_info = env.step(action)
    done = terminated or truncated
    agent.store_transition(state, action, reward, next_state, done)
    episode_reward += reward
    episode_length += 1
    state = next_state

# Train agent
metrics_tracker.record_episode(
    reward=episode_reward,
    score=info.get("score", 0),  # ❌ WRONG
    ...
)
```

**To this**:
```python
# Collect episode
final_info = info  # Initialize with reset info
while not done:
    action = agent.act(state, deterministic=False)
    next_state, reward, terminated, truncated, step_info = env.step(action)
    done = terminated or truncated
    agent.store_transition(state, action, reward, next_state, done)
    episode_reward += reward
    episode_length += 1
    state = next_state
    final_info = step_info  # ✅ Update with latest step info

# Train agent
metrics_tracker.record_episode(
    reward=episode_reward,
    score=final_info.get("score", 0),  # ✅ CORRECT
    ...
)
```

4. **Save the file** (Ctrl+S or Cmd+S)

5. **Re-run the training cell** - no need to restart runtime!

## Option 3: Fix Directly in the Notebook Cell

If you're using the Colab notebook (`colab_snake_setup.ipynb`), edit the training cell directly:

1. Find the training loop cell (Step 9)
2. Add `final_info = info` before the while loop
3. Add `final_info = step_info` inside the while loop
4. Change all `info.get("score", 0)` to `final_info.get("score", 0)`

## Option 4: Quick Fix with Code Replacement

Run this in a Colab cell to automatically fix the file:

```python
import re

# Read the file
with open('/content/drl/src/experiments/train_snake.py', 'r') as f:
    content = f.read()

# Fix DQN training section
content = re.sub(
    r'(# Collect episode\n\s+)(while not done:)',
    r'\1final_info = info  # Initialize with reset info\n\2',
    content
)

content = re.sub(
    r'(state = next_state\n\s+)(# Train agent)',
    r'\1final_info = step_info  # Update with latest step info\n\2',
    content
)

# Replace all info.get("score", 0) with final_info.get("score", 0) in training sections
content = re.sub(
    r'score=info\.get\("score", 0\)',
    r'score=final_info.get("score", 0)',
    content
)

# Write back
with open('/content/drl/src/experiments/train_snake.py', 'w') as f:
    f.write(content)

print("✅ File fixed! Now re-run your training cell.")
```

## Option 5: Re-upload Fixed File

1. Download the fixed `train_snake.py` from your local machine
2. In Colab, click the folder icon
3. Navigate to `src/experiments/`
4. Click the upload icon (📤)
5. Upload the fixed file (it will replace the old one)
6. Re-run the training cell

## After Fixing

1. **No need to restart runtime** - just re-run the training cell
2. **Clear variables** (optional, if you want fresh start):
```python
# Clear previous training state
%reset -f
# Then re-run imports and training
```

3. **Verify the fix** - scores should now show correctly!

## Quick Verification

Add this debug print in your training loop to verify:

```python
if episode % 10 == 0:
    print(f"Episode {episode}: Score={final_info.get('score', 0)}, "
          f"Length={episode_length}, Epsilon={agent.epsilon:.3f}")
```

You should see scores > 0 when the agent eats food!
