"""
Cooperative Navigation Environment
Multiple agents must navigate to cover all landmarks without collisions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from .base_env import BaseMultiAgentEnv


class CooperativeNavigation(BaseMultiAgentEnv):
    """
    Cooperative navigation task where agents must cover landmarks

    State Space (per agent):
        - Own position (2D)
        - Own velocity (2D)
        - Relative positions to all landmarks (2D * num_landmarks)
        - Relative positions to other agents (2D * (num_agents-1))

    Action Space: Continuous 2D force (x, y)

    Rewards:
        - Global reward: negative sum of distances to nearest landmarks
        - Collision penalty: -10 for agent-agent collision
    """

    def __init__(self, num_agents: int = 3, num_landmarks: int = 3,
                 world_size: float = 2.0, max_steps: int = 100):
        # Agent observation: pos(2) + vel(2) + landmarks(2*n_l) + agents(2*(n_a-1))
        state_dim = 2 + 2 + 2 * num_landmarks + 2 * (num_agents - 1)
        action_dim = 2  # 2D continuous force

        super().__init__(num_agents, state_dim, action_dim)

        self.num_landmarks = num_landmarks
        self.world_size = world_size
        self.max_steps = max_steps

        # Physical parameters
        self.agent_size = 0.05
        self.landmark_size = 0.05
        self.dt = 0.1
        self.damping = 0.25

        # State variables
        self.agent_pos = None
        self.agent_vel = None
        self.landmark_pos = None

    def reset(self) -> List[np.ndarray]:
        """Reset environment"""
        self.episode_step = 0

        # Randomly initialize agent positions and velocities
        self.agent_pos = np.random.uniform(-self.world_size, self.world_size,
                                          (self.num_agents, 2))
        self.agent_vel = np.zeros((self.num_agents, 2))

        # Randomly initialize landmark positions
        self.landmark_pos = np.random.uniform(-self.world_size, self.world_size,
                                             (self.num_landmarks, 2))

        return self._get_observations()

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Execute one step"""
        self.episode_step += 1
        actions = np.array(actions)

        # Update physics
        self._update_physics(actions)

        # Calculate rewards
        rewards, info = self._calculate_rewards()

        # Check termination
        dones = [self.episode_step >= self.max_steps] * self.num_agents

        # Get observations
        observations = self._get_observations()

        return observations, rewards, dones, info

    def _update_physics(self, actions: np.ndarray):
        """Update agent positions based on actions"""
        # Apply forces (actions are forces)
        forces = np.clip(actions, -1, 1)

        # Update velocities with damping
        self.agent_vel = self.agent_vel * (1 - self.damping) + forces * self.dt

        # Update positions
        self.agent_pos = self.agent_pos + self.agent_vel * self.dt

        # Boundary constraints
        self.agent_pos = np.clip(self.agent_pos, -self.world_size, self.world_size)

    def _calculate_rewards(self) -> Tuple[List[float], Dict]:
        """Calculate rewards for all agents"""
        rewards = np.zeros(self.num_agents)

        # Global reward: minimum distance from each landmark to nearest agent
        landmark_dists = []
        for lm_pos in self.landmark_pos:
            dists = np.linalg.norm(self.agent_pos - lm_pos, axis=1)
            landmark_dists.append(np.min(dists))

        global_reward = -np.sum(landmark_dists)
        rewards += global_reward

        # Collision penalty
        collision_count = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist < 2 * self.agent_size:
                    rewards[i] -= 10
                    rewards[j] -= 10
                    collision_count += 1

        info = {
            'global_reward': global_reward,
            'avg_landmark_dist': np.mean(landmark_dists),
            'collision_count': collision_count
        }

        return rewards.tolist(), info

    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents"""
        observations = []

        for i in range(self.num_agents):
            obs = []

            # Own position and velocity
            obs.append(self.agent_pos[i])
            obs.append(self.agent_vel[i])

            # Relative positions to landmarks
            for lm_pos in self.landmark_pos:
                obs.append(lm_pos - self.agent_pos[i])

            # Relative positions to other agents
            for j in range(self.num_agents):
                if i != j:
                    obs.append(self.agent_pos[j] - self.agent_pos[i])

            observations.append(np.concatenate(obs))

        return observations

    def get_global_state(self) -> np.ndarray:
        """Return global state for centralized critic"""
        # Global state: all agent positions, velocities, and landmark positions
        global_state = np.concatenate([
            self.agent_pos.flatten(),
            self.agent_vel.flatten(),
            self.landmark_pos.flatten()
        ])
        return global_state

    def render(self, mode: str = 'human', save_path: str = None):
        """Render the environment"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set world boundaries
        ax.set_xlim(-self.world_size - 0.5, self.world_size + 0.5)
        ax.set_ylim(-self.world_size - 0.5, self.world_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Cooperative Navigation (Step {self.episode_step})')

        # Draw landmarks
        for lm_pos in self.landmark_pos:
            circle = plt.Circle(lm_pos, self.landmark_size, color='green', alpha=0.5)
            ax.add_patch(circle)
            ax.plot(lm_pos[0], lm_pos[1], 'g*', markersize=15)

        # Draw agents
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_agents))
        for i, (pos, vel) in enumerate(zip(self.agent_pos, self.agent_vel)):
            circle = plt.Circle(pos, self.agent_size, color=colors[i], alpha=0.7)
            ax.add_patch(circle)

            # Draw velocity vector
            ax.arrow(pos[0], pos[1], vel[0]*0.5, vel[1]*0.5,
                    head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i])

            ax.text(pos[0], pos[1], str(i), ha='center', va='center',
                   fontsize=10, color='white', weight='bold')

        ax.legend(['Landmarks', 'Agents'], loc='upper right')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if mode == 'human':
            plt.show()

        plt.close()
