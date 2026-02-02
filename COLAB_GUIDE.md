# Google Colab Setup Guide for Snake RL Project

This guide will help you train your Snake RL project on Google Colab.

## Quick Start Steps

### 1. Upload Files to Colab

**Option A: Upload via File Browser (Easiest)**
1. Open Google Colab: https://colab.research.google.com/
2. Upload the notebook: `notebooks/colab_snake_setup.ipynb`
3. Use the file browser (folder icon on left) to upload your entire `src/` folder
4. Upload `configs/snake_config.yaml` (or let the notebook create it)

**Option B: Clone from GitHub**
1. Push your project to GitHub
2. In Colab, use: `!git clone https://github.com/yourusername/drl.git /content/drl`

### 2. Run the Notebook

1. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)
2. Run cells sequentially:
   - Mount Google Drive (optional but recommended)
   - Install dependencies
   - Upload/create project files
   - Verify imports
   - Train agent
   - Visualize results
   - Save to Drive

### 3. File Structure in Colab

After setup, your Colab should have:
```
/content/drl/
├── src/
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── snake_env.py
│   │   └── snake_renderer.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dqn_agent.py
│   │   └── ppo_discrete_agent.py
│   ├── networks/
│   │   ├── __init__.py
│   │   └── dqn_network.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── replay_buffer.py
│   │   ├── training.py
│   │   └── visualization.py
│   └── experiments/
│       ├── __init__.py
│       ├── train_snake.py
│       └── evaluate_snake.py
├── configs/
│   └── snake_config.yaml
└── notebooks/
    └── colab_snake_setup.ipynb
```

## Important Notes

### GPU Usage
- **Always enable GPU** for faster training
- Free Colab provides T4 GPU (good for this project)
- GPU is automatically detected by PyTorch

### Saving Models
- **Mount Google Drive** to save models persistently
- Models are saved to: `/content/drive/MyDrive/drl_snake/`
- Without Drive, download files before session ends

### Training Time
- DQN: ~30-60 minutes for 2000 episodes (on GPU)
- PPO: ~45-90 minutes for 2000 episodes (on GPU)
- Adjust `total_episodes` in config for faster testing

### Memory Management
If you get "Out of Memory" errors:
- Reduce `batch_size` (64 → 32)
- Reduce `replay_buffer_size` (100000 → 50000)
- Use smaller networks: `[128, 128, 64]` → `[64, 64]`
- Use `feature` state representation (not `image`)

### Resuming Training
To resume from a checkpoint:
```python
# Load checkpoint
agent.load('/content/drive/MyDrive/drl_snake/snake_best_model.pth')

# Continue training from where you left off
# (modify episode range in training loop)
```

## Troubleshooting

### Import Errors
- Make sure all files in `src/` are uploaded
- Check that `__init__.py` files exist in each package
- Verify Python path: `sys.path.insert(0, '/content/drl/src')`

### CUDA Errors
- Restart runtime: Runtime → Restart runtime
- Check GPU availability: `torch.cuda.is_available()`
- Use CPU if GPU unavailable (slower but works)

### File Not Found
- Check file paths match your upload location
- Use absolute paths: `/content/drl/...`
- Verify files exist: `!ls /content/drl/src/`

## Recommended Settings for Colab

### For Quick Testing (Fast)
```yaml
training:
  total_episodes: 500
  eval_frequency: 50
  save_frequency: 250

dqn:
  batch_size: 32
  replay_buffer_size: 50000
  network: [64, 64]
```

### For Full Training (Better Results)
```yaml
training:
  total_episodes: 5000
  eval_frequency: 100
  save_frequency: 500

dqn:
  batch_size: 64
  replay_buffer_size: 100000
  network: [128, 128, 64]
```

## Next Steps After Training

1. **Download Models**: Save to Google Drive or download locally
2. **Evaluate**: Use evaluation script to test on more episodes
3. **Visualize**: Generate policy heatmaps and game replays
4. **Experiment**: Try different hyperparameters and algorithms

## Tips

- **Save frequently**: Colab sessions can disconnect
- **Use Drive**: Always mount Google Drive for important files
- **Monitor progress**: Check training curves regularly
- **Start small**: Test with 500 episodes first, then scale up
- **Compare algorithms**: Train both DQN and PPO to compare

Happy training! 🐍🤖
