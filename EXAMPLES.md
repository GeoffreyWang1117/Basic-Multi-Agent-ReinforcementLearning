# Examples - MultiAgent-CTDE-Lab

Quick reference examples for common use cases.

## Table of Contents

1. [Training Examples](#training-examples)
2. [Evaluation Examples](#evaluation-examples)
3. [Visualization Examples](#visualization-examples)
4. [Comparison Examples](#comparison-examples)
5. [Python API Examples](#python-api-examples)

---

## Training Examples

### Example 1: Quick MADDPG Training

```bash
# Train for 5000 episodes
python experiments/train.py \
    --env cooperative_navigation \
    --algorithm maddpg \
    --num_episodes 5000 \
    --exp_name quick_maddpg_test
```

### Example 2: CommMADDPG with TarMAC Communication

```bash
python experiments/train.py \
    --env cooperative_navigation \
    --algorithm comm_maddpg \
    --comm_type tarmac \
    --comm_rounds 2 \
    --num_agents 4 \
    --num_episodes 10000 \
    --exp_name tarmac_4agents
```

### Example 3: Predator-Prey Training

```bash
python experiments/train.py \
    --env predator_prey \
    --algorithm maddpg \
    --num_agents 4 \
    --num_episodes 15000 \
    --noise_start 0.6 \
    --exp_name predator_prey_4v1
```

### Example 4: Training with Config File

```bash
# Use predefined config
python experiments/train_from_config.py \
    --config configs/comm_maddpg_coop_nav.yaml
```

### Example 5: Config with Overrides

```bash
# Override specific parameters
python experiments/train_from_config.py \
    --config configs/maddpg_coop_nav.yaml \
    --override \
        training.num_episodes=20000 \
        algorithm.hidden_dim=512 \
        algorithm.lr_actor=0.0002 \
        env.num_agents=6
```

---

## Evaluation Examples

### Example 1: Evaluate Trained Model

```bash
python experiments/eval.py \
    --env cooperative_navigation \
    --algorithm maddpg \
    --model_path results/my_exp/best_model.pth \
    --num_episodes 20 \
    --render
```

### Example 2: Evaluate and Save Video

```bash
python experiments/eval.py \
    --env predator_prey \
    --algorithm comm_maddpg \
    --comm_type commnet \
    --model_path results/comm_exp/best_model.pth \
    --num_episodes 5 \
    --save_video \
    --video_path evaluation.gif
```

---

## Visualization Examples

### Example 1: Compare Multiple Experiments

```bash
python utils/compare_experiments.py \
    --result_dirs \
        results/maddpg_baseline \
        results/comm_maddpg_commnet \
        results/comm_maddpg_tarmac \
    --labels "MADDPG" "CommMADDPG-CommNet" "CommMADDPG-TarMAC" \
    --output my_comparison.png \
    --window 100
```

### Example 2: Custom Visualization Script

```python
# visualize_results.py
from utils.visualization import TrainingVisualizer
import json
import numpy as np

# Load training results
with open('results/my_exp/episodes.json', 'r') as f:
    episodes = json.load(f)

rewards = [ep['total_reward'] for ep in episodes]
steps = [ep['steps'] for ep in episodes]

# Create visualizer
viz = TrainingVisualizer(save_dir='my_plots')

# Plot dashboard
metrics = {
    'Episode Rewards': rewards,
    'Episode Length': steps
}
viz.plot_multi_metric_dashboard(metrics, 'training_metrics.png')

print("Visualizations saved!")
```

Run it:
```bash
python visualize_results.py
```

---

## Comparison Examples

### Example 1: Hyperparameter Comparison

Train multiple variants:

```bash
# Variant 1: Small learning rate
python experiments/train_from_config.py \
    --config configs/maddpg_coop_nav.yaml \
    --override \
        algorithm.lr_actor=0.00005 \
        output.exp_name=maddpg_lr_small

# Variant 2: Medium learning rate (baseline)
python experiments/train_from_config.py \
    --config configs/maddpg_coop_nav.yaml \
    --override output.exp_name=maddpg_lr_medium

# Variant 3: Large learning rate
python experiments/train_from_config.py \
    --config configs/maddpg_coop_nav.yaml \
    --override \
        algorithm.lr_actor=0.0005 \
        output.exp_name=maddpg_lr_large
```

Compare results:

```bash
python utils/compare_experiments.py \
    --result_dirs \
        results/maddpg_lr_small \
        results/maddpg_lr_medium \
        results/maddpg_lr_large \
    --labels "LR=5e-5" "LR=1e-4" "LR=5e-4" \
    --output lr_comparison.png
```

### Example 2: Algorithm Comparison

```bash
# Train MADDPG
python experiments/train.py \
    --env cooperative_navigation \
    --algorithm maddpg \
    --num_episodes 10000 \
    --exp_name maddpg_comparison

# Train CommMADDPG
python experiments/train.py \
    --env cooperative_navigation \
    --algorithm comm_maddpg \
    --comm_type commnet \
    --num_episodes 10000 \
    --exp_name comm_maddpg_comparison

# Compare
python utils/compare_experiments.py \
    --result_dirs \
        results/maddpg_comparison \
        results/comm_maddpg_comparison \
    --labels "MADDPG (no comm)" "CommMADDPG" \
    --output algorithm_comparison.png
```

---

## Python API Examples

### Example 1: Custom Training Loop

```python
import torch
import numpy as np
from environments.cooperative_navigation import CooperativeNavigation
from algorithms.maddpg import MADDPG
from utils.logger import Logger

# Setup
env = CooperativeNavigation(num_agents=3, num_landmarks=3)
env_info = env.get_env_info()

agent = MADDPG(
    num_agents=env_info['num_agents'],
    state_dim=env_info['state_dim'],
    action_dim=env_info['action_dim'],
    device='cpu'
)

logger = Logger('results', 'custom_training')

# Training loop
for episode in range(1000):
    states = env.reset()
    episode_reward = 0

    while True:
        actions = agent.act(states, noise=0.3)
        next_states, rewards, dones, info = env.step(actions)

        global_state = env.get_global_state()
        next_global_state = env.get_global_state()

        agent.step(states, actions, rewards, next_states, dones,
                  global_state, next_global_state)

        if len(agent.replay_buffer) > agent.batch_size:
            agent.update()

        states = next_states
        episode_reward += sum(rewards)

        if all(dones):
            break

    logger.log_episode(episode, episode_reward, 0, {})

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

# Save model
agent.save('my_custom_model.pth')
logger.plot_metrics()
```

### Example 2: Using Config Manager

```python
from utils.config import ConfigManager

# Load config
config = ConfigManager(config_path='configs/maddpg_coop_nav.yaml')

# Access values
num_episodes = config.get('training.num_episodes')
lr_actor = config.get('algorithm.lr_actor')

print(f"Training for {num_episodes} episodes")
print(f"Actor LR: {lr_actor}")

# Update config
config.update({
    'training': {'num_episodes': 5000},
    'algorithm': {'lr_actor': 0.0002}
})

# Save updated config
config.save('configs/my_custom_config.yaml')
```

### Example 3: Programmatic Evaluation

```python
import torch
import numpy as np
from environments.cooperative_navigation import CooperativeNavigation
from algorithms.maddpg import MADDPG

# Load environment and model
env = CooperativeNavigation(num_agents=3, num_landmarks=3)
env_info = env.get_env_info()

agent = MADDPG(
    num_agents=env_info['num_agents'],
    state_dim=env_info['state_dim'],
    action_dim=env_info['action_dim'],
    device='cpu'
)

# Load trained model
agent.load('results/my_exp/best_model.pth')

# Evaluate
episode_rewards = []

for episode in range(10):
    states = env.reset()
    episode_reward = 0

    while True:
        actions = agent.act(states, noise=0.0)  # No exploration
        next_states, rewards, dones, info = env.step(actions)

        states = next_states
        episode_reward += sum(rewards)

        if all(dones):
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
```

---

## Shell Script Examples

### Example: Batch Training

Create `train_all.sh`:

```bash
#!/bin/bash

# Train multiple configurations

echo "Starting batch training..."

# MADDPG variants
for lr in 0.0001 0.0002 0.0005; do
    python experiments/train_from_config.py \
        --config configs/maddpg_coop_nav.yaml \
        --override \
            algorithm.lr_actor=$lr \
            output.exp_name=maddpg_lr_$lr &
done

wait

echo "All training jobs completed!"

# Compare results
python utils/compare_experiments.py \
    --result_dirs results/maddpg_lr_* \
    --output batch_comparison.png
```

Run:
```bash
chmod +x train_all.sh
./train_all.sh
```

---

## Quick Reference Commands

```bash
# Quick test
python quick_start.py

# Run tests
python run_tests.py

# Train with config
python experiments/train_from_config.py --config configs/maddpg_coop_nav.yaml

# Compare experiments
python utils/compare_experiments.py --result_dirs results/exp1 results/exp2 --labels "Exp1" "Exp2"

# Evaluate model
python experiments/eval.py --model_path results/my_exp/best_model.pth --env cooperative_navigation --algorithm maddpg
```

---

**For more details, see USAGE_GUIDE.md**
