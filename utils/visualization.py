"""
Advanced Visualization Tools for Multi-Agent RL
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow
from typing import List, Dict
import os


class TrainingVisualizer:
    """Visualize training progress and agent behaviors"""

    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_multi_metric_dashboard(self, metrics: Dict[str, List], save_name: str = 'dashboard.png'):
        """
        Create comprehensive dashboard with multiple metrics

        Args:
            metrics: Dictionary of metric_name -> list of values
            save_name: Filename to save the plot
        """
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (metric_name, values) in enumerate(metrics.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Plot raw data
            ax.plot(values, alpha=0.3, linewidth=0.5, color='blue')

            # Plot smoothed
            if len(values) > 10:
                window = min(50, len(values) // 10)
                smoothed = self._smooth(values, window)
                ax.plot(smoothed, linewidth=2, color='red', label='Smoothed')

            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Step/Episode')
            ax.set_ylabel('Value')
            ax.grid(alpha=0.3)
            ax.legend()

        # Remove empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Dashboard saved to: {save_path}")

    def plot_agent_trajectories(self, trajectories: List[np.ndarray], landmarks: np.ndarray = None,
                                save_name: str = 'trajectories.png'):
        """
        Plot agent trajectories in 2D space

        Args:
            trajectories: List of trajectories, each (T, 2) array
            landmarks: Optional landmark positions (N, 2)
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))

        # Plot trajectories
        for i, (traj, color) in enumerate(zip(trajectories, colors)):
            # Plot path
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6,
                   linewidth=2, label=f'Agent {i}')

            # Plot start and end
            ax.scatter(traj[0, 0], traj[0, 1], color=color, s=200,
                      marker='o', edgecolor='black', linewidth=2, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=200,
                      marker='s', edgecolor='black', linewidth=2, zorder=5)

            # Add arrow for direction
            if len(traj) > 1:
                for t in range(0, len(traj) - 1, max(1, len(traj) // 10)):
                    dx = traj[t + 1, 0] - traj[t, 0]
                    dy = traj[t + 1, 1] - traj[t, 1]
                    ax.arrow(traj[t, 0], traj[t, 1], dx, dy,
                            head_width=0.05, head_length=0.05,
                            fc=color, ec=color, alpha=0.4)

        # Plot landmarks if provided
        if landmarks is not None:
            ax.scatter(landmarks[:, 0], landmarks[:, 1], color='green',
                      s=300, marker='*', edgecolor='black', linewidth=2,
                      label='Landmarks', zorder=10)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Agent Trajectories', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Trajectories saved to: {save_path}")

    def create_episode_animation(self, episode_data: List[Dict], env, save_name: str = 'episode.gif'):
        """
        Create animated GIF of an episode

        Args:
            episode_data: List of dicts with 'agent_pos', 'landmark_pos', etc.
            env: Environment for rendering context
            save_name: Filename for the GIF
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            ax.clear()

            data = episode_data[frame]
            agent_pos = data['agent_pos']
            num_agents = len(agent_pos)

            # Set limits
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
            ax.set_title(f'Step {frame}/{len(episode_data)}', fontsize=14)

            # Plot landmarks
            if 'landmark_pos' in data:
                landmark_pos = data['landmark_pos']
                for lm in landmark_pos:
                    circle = Circle(lm, 0.1, color='green', alpha=0.3)
                    ax.add_patch(circle)
                    ax.plot(lm[0], lm[1], 'g*', markersize=20)

            # Plot agents
            colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
            for i, (pos, color) in enumerate(zip(agent_pos, colors)):
                circle = Circle(pos, 0.075, color=color, alpha=0.7, edgecolor='black', linewidth=2)
                ax.add_patch(circle)

                # Add velocity arrow if available
                if 'agent_vel' in data:
                    vel = data['agent_vel'][i]
                    ax.arrow(pos[0], pos[1], vel[0] * 0.3, vel[1] * 0.3,
                            head_width=0.05, head_length=0.05,
                            fc=color, ec=color, linewidth=2)

                ax.text(pos[0], pos[1], str(i), ha='center', va='center',
                       color='white', fontweight='bold', fontsize=10)

        anim = animation.FuncAnimation(fig, update, frames=len(episode_data),
                                      interval=100, repeat=True)

        save_path = os.path.join(self.save_dir, save_name)
        anim.save(save_path, writer='pillow', fps=10)
        plt.close()

        print(f"Animation saved to: {save_path}")

    def plot_reward_heatmap(self, reward_grid: np.ndarray, save_name: str = 'reward_heatmap.png'):
        """
        Plot heatmap of rewards across episodes/agents

        Args:
            reward_grid: 2D array (episodes x agents)
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(reward_grid.T, aspect='auto', cmap='RdYlGn',
                      interpolation='nearest')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Agent', fontsize=12)
        ax.set_title('Reward Heatmap', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Reward', fontsize=12)

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Heatmap saved to: {save_path}")

    def plot_cooperation_matrix(self, cooperation_matrix: np.ndarray,
                                agent_names: List[str] = None,
                                save_name: str = 'cooperation.png'):
        """
        Visualize cooperation/interaction matrix between agents

        Args:
            cooperation_matrix: Symmetric matrix (n_agents x n_agents)
            agent_names: Optional names for agents
            save_name: Filename to save
        """
        n_agents = cooperation_matrix.shape[0]

        if agent_names is None:
            agent_names = [f'Agent {i}' for i in range(n_agents)]

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cooperation_matrix, cmap='Blues', vmin=0)

        # Set ticks
        ax.set_xticks(np.arange(n_agents))
        ax.set_yticks(np.arange(n_agents))
        ax.set_xticklabels(agent_names)
        ax.set_yticklabels(agent_names)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Add values
        for i in range(n_agents):
            for j in range(n_agents):
                text = ax.text(j, i, f'{cooperation_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black')

        ax.set_title('Agent Cooperation Matrix', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cooperation Score', fontsize=12)

        save_path = os.path.join(self.save_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Cooperation matrix saved to: {save_path}")

    def _smooth(self, values: List, window: int) -> List:
        """Apply moving average smoothing"""
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        return smoothed
