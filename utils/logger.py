"""
Logging utilities for training with TensorBoard support
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class Logger:
    """Training logger with TensorBoard integration"""

    def __init__(self, log_dir: str, exp_name: str = None, use_tensorboard: bool = True):
        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics = {}
        self.episode_data = []

        # TensorBoard writer
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer: Optional[SummaryWriter] = None

        if self.use_tensorboard:
            tb_dir = os.path.join(self.log_dir, 'tensorboard')
            self.writer = SummaryWriter(tb_dir)
            print(f"TensorBoard logging to: {tb_dir}")
            print(f"  View with: tensorboard --logdir {tb_dir}")

    def log_scalar(self, key: str, value: float, step: int):
        """Log a scalar value"""
        if key not in self.metrics:
            self.metrics[key] = {'steps': [], 'values': []}

        self.metrics[key]['steps'].append(step)
        self.metrics[key]['values'].append(value)

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar(key, value, step)

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

    def log_histogram(self, key: str, values: np.ndarray, step: int):
        """Log histogram to TensorBoard"""
        if self.writer is not None:
            self.writer.add_histogram(key, values, step)

    def log_image(self, key: str, image: np.ndarray, step: int):
        """Log image to TensorBoard"""
        if self.writer is not None:
            self.writer.add_image(key, image, step, dataformats='HWC')

    def log_figure(self, key: str, figure: plt.Figure, step: int):
        """Log matplotlib figure to TensorBoard"""
        if self.writer is not None:
            self.writer.add_figure(key, figure, step)

    def log_text(self, key: str, text: str, step: int):
        """Log text to TensorBoard"""
        if self.writer is not None:
            self.writer.add_text(key, text, step)

    def log_hparams(self, hparams: Dict, metrics: Dict):
        """Log hyperparameters and metrics"""
        if self.writer is not None:
            self.writer.add_hparams(hparams, metrics)

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

    def close(self):
        """Close TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()
            print("TensorBoard writer closed")

    def __del__(self):
        """Cleanup"""
        self.close()
