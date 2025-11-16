# Usage Guide - MultiAgent-CTDE-Lab

This guide provides detailed instructions for using all features of the Multi-Agent RL framework.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training with YAML Configs](#training-with-yaml-configs)
3. [Comparing Experiments](#comparing-experiments)
4. [Advanced Visualization](#advanced-visualization)
5. [Running Tests](#running-tests)
6. [Custom Environments](#custom-environments)
7. [Custom Algorithms](#custom-algorithms)

---

## Quick Start

### Basic Test

Verify everything works:

```bash
python quick_start.py
```

This runs a quick test of all components.

### Simple Training

Train MADDPG on cooperative navigation:

```bash
python experiments/train.py \
    --env cooperative_navigation \
    --algorithm maddpg \
    --num_episodes 1000
```

---

## Training with YAML Configs

### Using Pre-configured YAML Files

We provide several pre-configured YAML files in `configs/`:

**1. MADDPG on Cooperative Navigation:**
```bash
python experiments/train_from_config.py \
    --config configs/maddpg_coop_nav.yaml
```

**2. CommMADDPG with Communication:**
```bash
python experiments/train_from_config.py \
    --config configs/comm_maddpg_coop_nav.yaml
```

**3. Predator-Prey with MADDPG:**
```bash
python experiments/train_from_config.py \
    --config configs/predator_prey_maddpg.yaml
```

### Overriding Config Parameters

Override specific parameters from command line:

```bash
python experiments/train_from_config.py \
    --config configs/maddpg_coop_nav.yaml \
    --override \
        training.num_episodes=5000 \
        algorithm.lr_actor=0.0002 \
        env.num_agents=5
```

### Creating Custom Configs

Create your own YAML config file:

```yaml
# my_config.yaml
env:
  name: cooperative_navigation
  num_agents: 4
  num_landmarks: 4

algorithm:
  name: comm_maddpg
  comm_type: tarmac
  comm_rounds: 3
  hidden_dim: 512

training:
  num_episodes: 20000
  noise_start: 0.3

output:
  exp_name: my_experiment
```

Then run:
```bash
python experiments/train_from_config.py --config my_config.yaml
```

---

## Comparing Experiments

### Compare Multiple Training Runs

After training multiple experiments, compare them:

```bash
python utils/compare_experiments.py \
    --result_dirs \
        results/maddpg_exp1 \
        results/comm_maddpg_exp1 \
        results/maddpg_exp2 \
    --labels "MADDPG" "CommMADDPG" "MADDPG-v2" \
    --output comparison_plot.png \
    --window 100
```

This generates:
- Learning curve comparisons
- Episode length comparison
- Success rate comparison (if available)
- Cumulative reward comparison
- Summary table in markdown

### Programmatic Comparison

```python
from utils.compare_experiments import ExperimentComparator

# Create comparator
comparator = ExperimentComparator(
    result_dirs=['results/exp1', 'results/exp2'],
    labels=['Baseline', 'Improved']
)

# Print summary
comparator.print_summary()

# Generate comparison table
print(comparator.generate_comparison_table())

# Plot learning curves
comparator.plot_learning_curves(save_path='comparison.png', window=50)
```

---

## Advanced Visualization

### Using the Visualization Module

```python
from utils.visualization import TrainingVisualizer
import numpy as np

viz = TrainingVisualizer(save_dir='my_visualizations')

# 1. Plot multi-metric dashboard
metrics = {
    'Episode Reward': episode_rewards,
    'Critic Loss': critic_losses,
    'Actor Loss': actor_losses,
    'Success Rate': success_rates
}
viz.plot_multi_metric_dashboard(metrics, save_name='dashboard.png')

# 2. Plot agent trajectories
trajectories = [agent1_trajectory, agent2_trajectory, agent3_trajectory]
landmarks = np.array([[1.0, 1.0], [-1.0, -1.0]])
viz.plot_agent_trajectories(trajectories, landmarks, save_name='trajectories.png')

# 3. Create reward heatmap
reward_grid = np.array([...])  # Shape: (episodes, agents)
viz.plot_reward_heatmap(reward_grid, save_name='heatmap.png')

# 4. Plot cooperation matrix
cooperation_matrix = np.array([...])  # Shape: (n_agents, n_agents)
viz.plot_cooperation_matrix(cooperation_matrix, save_name='cooperation.png')
```

### Creating Episode Animations

```python
# During evaluation, collect episode data
episode_data = []
for step in range(num_steps):
    episode_data.append({
        'agent_pos': env.agent_pos.copy(),
        'agent_vel': env.agent_vel.copy(),
        'landmark_pos': env.landmark_pos.copy()
    })

# Create animation
viz.create_episode_animation(episode_data, env, save_name='episode.gif')
```

---

## Running Tests

### Run All Tests

```bash
python run_tests.py
```

### Run Specific Test Module

```bash
python -m unittest tests.test_environments
python -m unittest tests.test_algorithms
```

### Run Specific Test

```bash
python -m unittest tests.test_environments.TestCooperativeNavigation.test_reset
```

### Test with Coverage (optional)

```bash
pip install coverage
coverage run run_tests.py
coverage report
coverage html  # Generate HTML report
```

---

## Custom Environments

### Creating a Custom Environment

1. **Create new file** in `environments/`:

```python
# environments/my_custom_env.py
from environments.base_env import BaseMultiAgentEnv
import numpy as np

class MyCustomEnv(BaseMultiAgentEnv):
    def __init__(self, num_agents: int = 3):
        state_dim = 10  # Your state dimension
        action_dim = 2  # Your action dimension
        super().__init__(num_agents, state_dim, action_dim)

        # Initialize your environment

    def reset(self):
        # Reset environment
        states = [...]  # List of observations
        return states

    def step(self, actions):
        # Execute actions
        next_states = [...]
        rewards = [...]
        dones = [...]
        info = {}
        return next_states, rewards, dones, info

    def get_global_state(self):
        # Return global state for CTDE
        return np.concatenate([...])

    def render(self, mode='human'):
        # Render environment
        pass
```

2. **Register** in `environments/__init__.py`:

```python
from .my_custom_env import MyCustomEnv

__all__ = [
    'BaseMultiAgentEnv',
    'CooperativeNavigation',
    'PredatorPrey',
    'MyCustomEnv'  # Add here
]
```

3. **Use in training**:

```bash
# Modify train.py or create custom training script
```

---

## Custom Algorithms

### Implementing a Custom Algorithm

1. **Create algorithm file** in `algorithms/`:

```python
# algorithms/my_algorithm.py
import torch
import torch.nn as nn
from utils.replay_buffer import MultiAgentReplayBuffer

class MyAlgorithm:
    def __init__(self, num_agents, state_dim, action_dim, device='cpu'):
        self.num_agents = num_agents
        self.device = device

        # Initialize networks
        # ...

        self.replay_buffer = MultiAgentReplayBuffer(100000, num_agents)

    def act(self, states, noise=0.0):
        """Select actions for all agents"""
        # Implement action selection
        actions = [...]
        return actions

    def step(self, states, actions, rewards, next_states, dones,
             global_state=None, next_global_state=None):
        """Store transition"""
        self.replay_buffer.add(states, actions, rewards, next_states, dones,
                              global_state, next_global_state)

    def update(self):
        """Update algorithm"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample and update
        # ...

        return {'loss': loss_value}

    def save(self, filepath):
        """Save model"""
        torch.save(..., filepath)

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        # Load parameters
```

2. **Register in** `algorithms/__init__.py`

3. **Add to training script** options

---

## Best Practices

### 1. **Experiment Organization**

Use descriptive experiment names:
```bash
python experiments/train_from_config.py \
    --config configs/maddpg_coop_nav.yaml \
    --override output.exp_name=maddpg_lr1e-4_bs256_20250116
```

### 2. **Hyperparameter Tuning**

Create variants of config files:
```
configs/
  maddpg_baseline.yaml
  maddpg_large_lr.yaml
  maddpg_small_batch.yaml
```

### 3. **Reproducibility**

Always set seeds in config:
```yaml
system:
  seed: 42
```

### 4. **Monitoring Training**

Check logs regularly:
```bash
tail -f results/my_exp/episodes.json
```

### 5. **Model Checkpointing**

Save frequently during long training:
```yaml
training:
  save_freq: 500  # Save every 500 episodes
```

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory:**
- Reduce `batch_size` in config
- Use smaller `hidden_dim`
- Use CPU: `system.device: cpu`

**2. Training not converging:**
- Adjust learning rates (`lr_actor`, `lr_critic`)
- Increase `noise_start` for more exploration
- Check reward scaling

**3. Import errors:**
- Ensure you're in project root
- Check `__init__.py` files exist
- Verify Python path

**4. Slow training:**
- Enable CUDA if available
- Increase `batch_size`
- Reduce `update_freq`

---

## Additional Resources

- **README.md**: Project overview and quick start
- **configs/**: Example configuration files
- **tests/**: Unit tests for reference
- **examples/**: Additional example scripts

## Getting Help

- Check existing code in `experiments/` for examples
- Run `python script.py --help` for command-line options
- Review unit tests in `tests/` for API usage

---

**Happy Training! 🚀**
