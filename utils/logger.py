"""
Logging utilities for training
"""

import os
import json
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from datetime import datetime


class Logger:
    """Training logger"""

    def __init__(self, log_dir: str, exp_name: str = None):
        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics = {}
        self.episode_data = []

    def log_scalar(self, key: str, value: float, step: int):
        """Log a scalar value"""
        if key not in self.metrics:
            self.metrics[key] = {'steps': [], 'values': []}

        self.metrics[key]['steps'].append(step)
        self.metrics[key]['values'].append(value)

    def log_episode(self, episode: int, total_reward: float, steps: int, info: Dict = None):
        """Log episode information"""
        episode_info = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'timestamp': datetime.now().isoformat()
        }

        if info:
            episode_info.update(info)

        self.episode_data.append(episode_info)

        # Save to JSON
        with open(os.path.join(self.log_dir, 'episodes.json'), 'w') as f:
            json.dump(self.episode_data, f, indent=2)

    def plot_metrics(self, keys: List[str] = None, save: bool = True):
        """Plot training metrics"""
        if keys is None:
            keys = list(self.metrics.keys())

        num_plots = len(keys)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

        if num_plots == 1:
            axes = [axes]

        for ax, key in zip(axes, keys):
            if key in self.metrics:
                steps = self.metrics[key]['steps']
                values = self.metrics[key]['values']
                ax.plot(steps, values, alpha=0.6)

                # Smooth curve
                if len(values) > 10:
                    window = min(50, len(values) // 10)
                    smoothed = self._smooth(values, window)
                    ax.plot(steps, smoothed, linewidth=2, label='Smoothed')

                ax.set_xlabel('Step')
                ax.set_ylabel(key)
                ax.set_title(key)
                ax.legend()
                ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.log_dir, 'metrics.png'), dpi=150)

        plt.close()

    def _smooth(self, values: List[float], window: int) -> List[float]:
        """Smooth values using moving average"""
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        return smoothed

    def save_config(self, config: Dict):
        """Save training configuration"""
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def print_summary(self, window: int = 100):
        """Print training summary"""
        if len(self.episode_data) == 0:
            return

        recent = self.episode_data[-window:]
        avg_reward = np.mean([ep['total_reward'] for ep in recent])
        avg_steps = np.mean([ep['steps'] for ep in recent])

        print(f"\n{'='*50}")
        print(f"Training Summary (Last {len(recent)} episodes)")
        print(f"{'='*50}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Total Episodes: {len(self.episode_data)}")
        print(f"{'='*50}\n")
