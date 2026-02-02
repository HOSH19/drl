# GitHub Setup Guide

Follow these steps to create a new GitHub repository and push your Snake RL project.

## Step 1: Create Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in the details:
   - **Repository name**: `snake-rl` (or your preferred name)
   - **Description**: "Deep RL project for training agents to play Snake using DQN and PPO"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we'll add these)
4. Click **"Create repository"**

## Step 2: Initialize Git Locally

Open terminal in your project directory and run:

```bash
cd /Users/hoshuhan/Documents/passion/drl

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Snake RL project with DQN and PPO implementations"
```

## Step 3: Connect to GitHub and Push

```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/snake-rl.git

# Rename default branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Alternative: Using SSH (if you have SSH keys set up)

```bash
# Add remote with SSH URL
git remote add origin git@github.com:YOUR_USERNAME/snake-rl.git

# Push to GitHub
git push -u origin main
```

## Step 4: Verify

1. Go to your GitHub repository page
2. You should see all your files uploaded
3. The README.md will display on the repository homepage

## Common Issues

### Authentication Required
If you get authentication errors:
- **Option 1**: Use GitHub CLI (`gh auth login`)
- **Option 2**: Use Personal Access Token instead of password
  - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Generate new token with `repo` permissions
  - Use token as password when pushing

### Branch Name Issues
If you get branch name errors:
```bash
# Check current branch name
git branch

# Rename to main if needed
git branch -M main
```

### Large Files
If you have large model files (.pth), they're already in .gitignore. If you need to remove them from git history:
```bash
# Remove large files from git history (if already committed)
git rm --cached checkpoints/**/*.pth
git commit -m "Remove large model files"
```

## Future Updates

To push future changes:

```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Quick Reference Commands

```bash
# Check git status
git status

# See what files are tracked
git ls-files

# View commit history
git log --oneline

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout main
```

## Recommended: Add GitHub Actions (Optional)

Create `.github/workflows/python.yml` for CI/CD:

```yaml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Test imports
      run: |
        python -c "from src.environments import SnakeEnv; print('✓ Imports work')"
```

Happy coding! 🐍🤖
