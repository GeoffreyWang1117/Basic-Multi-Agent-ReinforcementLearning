"""
Experiment Comparison Tool
Compare multiple training runs and generate comparison plots
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import argparse


class ExperimentComparator:
    """Compare multiple experiment results"""

    def __init__(self, result_dirs: List[str], labels: List[str] = None):
        """
        Initialize comparator

        Args:
            result_dirs: List of result directory paths
            labels: Optional labels for each experiment
        """
        self.result_dirs = result_dirs
        self.labels = labels or [f"Exp {i+1}" for i in range(len(result_dirs))]
        self.experiments = []

        # Load experiments
        for result_dir in result_dirs:
            self.experiments.append(self._load_experiment(result_dir))

    def _load_experiment(self, result_dir: str) -> Dict:
        """Load experiment data"""
        episodes_file = os.path.join(result_dir, 'episodes.json')
        config_file = os.path.join(result_dir, 'config.json')

        exp_data = {
            'episodes': [],
            'config': {}
        }

        # Load episodes
        if os.path.exists(episodes_file):
            with open(episodes_file, 'r') as f:
                exp_data['episodes'] = json.load(f)

        # Load config
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                exp_data['config'] = json.load(f)

        return exp_data

    def plot_learning_curves(self, save_path: str = None, window: int = 50):
        """
        Plot learning curves for all experiments

        Args:
            save_path: Path to save the plot
            window: Smoothing window size
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # 1. Episode Rewards
        ax = axes[0]
        for exp, label in zip(self.experiments, self.labels):
            if exp['episodes']:
                episodes = [ep['episode'] for ep in exp['episodes']]
                rewards = [ep['total_reward'] for ep in exp['episodes']]

                # Plot raw
                ax.plot(episodes, rewards, alpha=0.3, linewidth=0.5)

                # Plot smoothed
                smoothed = self._smooth(rewards, window)
                ax.plot(episodes, smoothed, linewidth=2, label=label)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Episode Lengths
        ax = axes[1]
        for exp, label in zip(self.experiments, self.labels):
            if exp['episodes']:
                episodes = [ep['episode'] for ep in exp['episodes']]
                steps = [ep['steps'] for ep in exp['episodes']]

                smoothed = self._smooth(steps, window)
                ax.plot(episodes, smoothed, linewidth=2, label=label)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Success Rate (if available)
        ax = axes[2]
        for exp, label in zip(self.experiments, self.labels):
            if exp['episodes'] and any('success' in ep for ep in exp['episodes']):
                episodes = [ep['episode'] for ep in exp['episodes']]
                success = [ep.get('success', 0) for ep in exp['episodes']]

                # Compute rolling success rate
                success_rate = self._rolling_mean(success, window)
                ax.plot(episodes, success_rate, linewidth=2, label=label)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title(f'Success Rate (Rolling {window} episodes)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.1])

        # 4. Cumulative Reward
        ax = axes[3]
        for exp, label in zip(self.experiments, self.labels):
            if exp['episodes']:
                episodes = [ep['episode'] for ep in exp['episodes']]
                rewards = [ep['total_reward'] for ep in exp['episodes']]

                cumulative = np.cumsum(rewards)
                ax.plot(episodes, cumulative, linewidth=2, label=label)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Rewards')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to: {save_path}")

        plt.show()

    def generate_comparison_table(self) -> str:
        """Generate markdown comparison table"""
        table = "## Experiment Comparison\n\n"
        table += "| Experiment | Algorithm | Episodes | Avg Reward (last 100) | Best Reward | Avg Steps |\n"
        table += "|------------|-----------|----------|----------------------|-------------|----------|\n"

        for exp, label in zip(self.experiments, self.labels):
            if exp['episodes']:
                algo = exp['config'].get('algorithm', 'N/A')
                num_episodes = len(exp['episodes'])

                # Last 100 episodes stats
                recent = exp['episodes'][-100:]
                avg_reward = np.mean([ep['total_reward'] for ep in recent])
                best_reward = np.max([ep['total_reward'] for ep in exp['episodes']])
                avg_steps = np.mean([ep['steps'] for ep in recent])

                table += f"| {label} | {algo} | {num_episodes} | {avg_reward:.2f} | {best_reward:.2f} | {avg_steps:.1f} |\n"

        return table

    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*70)
        print("EXPERIMENT COMPARISON SUMMARY")
        print("="*70)

        for i, (exp, label) in enumerate(zip(self.experiments, self.labels)):
            print(f"\n{label}:")
            print(f"  Algorithm: {exp['config'].get('algorithm', 'N/A')}")
            print(f"  Environment: {exp['config'].get('env', 'N/A')}")
            print(f"  Num Agents: {exp['config'].get('num_agents', 'N/A')}")

            if exp['episodes']:
                print(f"  Episodes: {len(exp['episodes'])}")

                recent = exp['episodes'][-100:]
                avg_reward = np.mean([ep['total_reward'] for ep in recent])
                std_reward = np.std([ep['total_reward'] for ep in recent])
                best_reward = np.max([ep['total_reward'] for ep in exp['episodes']])

                print(f"  Avg Reward (last 100): {avg_reward:.2f} ± {std_reward:.2f}")
                print(f"  Best Reward: {best_reward:.2f}")

        print("\n" + "="*70 + "\n")

    def _smooth(self, values: List[float], window: int) -> List[float]:
        """Smooth values using moving average"""
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        return smoothed

    def _rolling_mean(self, values: List[float], window: int) -> List[float]:
        """Calculate rolling mean"""
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(np.mean(values[start:i+1]))
        return result


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--result_dirs', nargs='+', required=True,
                       help='Directories containing experiment results')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='Labels for each experiment')
    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output plot filename')
    parser.add_argument('--window', type=int, default=50,
                       help='Smoothing window size')

    args = parser.parse_args()

    # Create comparator
    comparator = ExperimentComparator(args.result_dirs, args.labels)

    # Print summary
    comparator.print_summary()

    # Generate table
    print(comparator.generate_comparison_table())

    # Plot comparison
    comparator.plot_learning_curves(save_path=args.output, window=args.window)


if __name__ == '__main__':
    main()
