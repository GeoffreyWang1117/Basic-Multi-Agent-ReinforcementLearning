"""
Predator-Prey Environment
Predators (cooperative) must catch prey (adversarial) in a continuous space
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from .base_env import BaseMultiAgentEnv


class PredatorPrey(BaseMultiAgentEnv):
    """
    Predator-Prey environment with cooperative predators and adversarial prey

    State Space (per predator):
        - Own position (2D)
        - Own velocity (2D)
        - Relative position to prey (2D)
        - Relative positions to other predators (2D * (num_predators-1))

    State Space (prey):
        - Own position (2D)
        - Own velocity (2D)
        - Relative positions to all predators (2D * num_predators)

    Action Space: Continuous 2D force (x, y)

    Rewards:
        - Predators: +100 for catching prey, -distance_to_prey
        - Prey: +1 for each step survived, -100 if caught
    """

    def __init__(self, num_predators: int = 3, num_prey: int = 1,
                 world_size: float = 2.0, max_steps: int = 100):

        # Predator observation size
        pred_state_dim = 2 + 2 + 2 * num_prey + 2 * (num_predators - 1)

        super().__init__(num_predators + num_prey, pred_state_dim, 2)

        self.num_predators = num_predators
        self.num_prey = num_prey
        self.world_size = world_size
        self.max_steps = max_steps

        # Physical parameters
        self.predator_size = 0.075
        self.prey_size = 0.05
        self.catch_radius = 0.15
        self.dt = 0.1
        self.damping = 0.25

        # Different max speeds
        self.predator_max_speed = 0.5
        self.prey_max_speed = 0.6  # Prey slightly faster

        # State variables
        self.predator_pos = None
        self.predator_vel = None
        self.prey_pos = None
        self.prey_vel = None
        self.prey_caught = None

    def reset(self) -> List[np.ndarray]:
        """Reset environment"""
        self.episode_step = 0

        # Initialize predators
        self.predator_pos = np.random.uniform(-self.world_size, self.world_size,
                                             (self.num_predators, 2))
        self.predator_vel = np.zeros((self.num_predators, 2))

        # Initialize prey (start far from predators)
        self.prey_pos = np.random.uniform(-self.world_size, self.world_size,
                                         (self.num_prey, 2))
        self.prey_vel = np.zeros((self.num_prey, 2))

        self.prey_caught = np.zeros(self.num_prey, dtype=bool)

        return self._get_observations()

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Execute one step"""
        self.episode_step += 1
        actions = np.array(actions)

        # Split actions for predators and prey
        predator_actions = actions[:self.num_predators]
        prey_actions = actions[self.num_predators:] if len(actions) > self.num_predators else None

        # Update physics
        self._update_physics(predator_actions, prey_actions)

        # Check for catches
        self._check_catches()

        # Calculate rewards
        rewards, info = self._calculate_rewards()

        # Check termination
        all_caught = np.all(self.prey_caught)
        dones = [self.episode_step >= self.max_steps or all_caught] * self.num_agents

        # Get observations
        observations = self._get_observations()

        return observations, rewards, dones, info

    def _update_physics(self, predator_actions: np.ndarray, prey_actions: np.ndarray = None):
        """Update positions based on actions"""
        # Update predators
        forces = np.clip(predator_actions, -1, 1)
        self.predator_vel = self.predator_vel * (1 - self.damping) + forces * self.dt

        # Limit predator speed
        pred_speeds = np.linalg.norm(self.predator_vel, axis=1, keepdims=True)
        pred_speeds = np.maximum(pred_speeds, 1e-6)
        self.predator_vel = np.where(
            pred_speeds > self.predator_max_speed,
            self.predator_vel * self.predator_max_speed / pred_speeds,
            self.predator_vel
        )

        self.predator_pos = self.predator_pos + self.predator_vel * self.dt
        self.predator_pos = np.clip(self.predator_pos, -self.world_size, self.world_size)

        # Update prey (if actions provided, otherwise use simple evasion policy)
        if prey_actions is not None:
            prey_forces = np.clip(prey_actions, -1, 1)
        else:
            # Simple evasion: move away from nearest predator
            prey_forces = np.zeros((self.num_prey, 2))
            for i in range(self.num_prey):
                if not self.prey_caught[i]:
                    dists = np.linalg.norm(self.predator_pos - self.prey_pos[i], axis=1)
                    nearest_pred = np.argmin(dists)
                    direction = self.prey_pos[i] - self.predator_pos[nearest_pred]
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    prey_forces[i] = direction

        self.prey_vel = self.prey_vel * (1 - self.damping) + prey_forces * self.dt

        # Limit prey speed
        prey_speeds = np.linalg.norm(self.prey_vel, axis=1, keepdims=True)
        prey_speeds = np.maximum(prey_speeds, 1e-6)
        self.prey_vel = np.where(
            prey_speeds > self.prey_max_speed,
            self.prey_vel * self.prey_max_speed / prey_speeds,
            self.prey_vel
        )

        self.prey_pos = self.prey_pos + self.prey_vel * self.dt
        self.prey_pos = np.clip(self.prey_pos, -self.world_size, self.world_size)

    def _check_catches(self):
        """Check if any prey is caught"""
        for i in range(self.num_prey):
            if not self.prey_caught[i]:
                # Check if enough predators are close
                dists = np.linalg.norm(self.predator_pos - self.prey_pos[i], axis=1)
                close_predators = np.sum(dists < self.catch_radius)

                # Need at least 2 predators to catch prey
                if close_predators >= 2:
                    self.prey_caught[i] = True

    def _calculate_rewards(self) -> Tuple[List[float], Dict]:
        """Calculate rewards"""
        rewards = []

        # Predator rewards
        for i in range(self.num_predators):
            reward = 0.0

            # Distance-based reward to nearest uncaught prey
            uncaught_prey = np.where(~self.prey_caught)[0]
            if len(uncaught_prey) > 0:
                prey_dists = np.linalg.norm(
                    self.prey_pos[uncaught_prey] - self.predator_pos[i], axis=1
                )
                reward -= np.min(prey_dists)

            # Catch bonus
            if np.any(self.prey_caught):
                reward += 100

            rewards.append(reward)

        # Prey rewards
        for i in range(self.num_prey):
            if self.prey_caught[i]:
                rewards.append(-100.0)
            else:
                rewards.append(1.0)  # Survival bonus

        info = {
            'prey_caught': np.sum(self.prey_caught),
            'total_prey': self.num_prey
        }

        return rewards, info

    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents"""
        observations = []

        # Predator observations
        for i in range(self.num_predators):
            obs = []
            obs.append(self.predator_pos[i])
            obs.append(self.predator_vel[i])

            # Relative positions to prey
            for j in range(self.num_prey):
                if not self.prey_caught[j]:
                    obs.append(self.prey_pos[j] - self.predator_pos[i])
                else:
                    obs.append(np.zeros(2))  # Caught prey

            # Relative positions to other predators
            for j in range(self.num_predators):
                if i != j:
                    obs.append(self.predator_pos[j] - self.predator_pos[i])

            observations.append(np.concatenate(obs))

        # Prey observations
        for i in range(self.num_prey):
            obs = []
            obs.append(self.prey_pos[i])
            obs.append(self.prey_vel[i])

            # Relative positions to all predators
            for j in range(self.num_predators):
                obs.append(self.predator_pos[j] - self.prey_pos[i])

            observations.append(np.concatenate(obs))

        return observations

    def get_global_state(self) -> np.ndarray:
        """Return global state for centralized critic"""
        global_state = np.concatenate([
            self.predator_pos.flatten(),
            self.predator_vel.flatten(),
            self.prey_pos.flatten(),
            self.prey_vel.flatten(),
            self.prey_caught.astype(float)
        ])
        return global_state

    def render(self, mode: str = 'human', save_path: str = None):
        """Render the environment"""
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xlim(-self.world_size - 0.5, self.world_size + 0.5)
        ax.set_ylim(-self.world_size - 0.5, self.world_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Predator-Prey (Step {self.episode_step})')

        # Draw predators
        for i, (pos, vel) in enumerate(zip(self.predator_pos, self.predator_vel)):
            circle = plt.Circle(pos, self.predator_size, color='red', alpha=0.7)
            ax.add_patch(circle)
            ax.arrow(pos[0], pos[1], vel[0]*0.3, vel[1]*0.3,
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
            ax.text(pos[0], pos[1], f'P{i}', ha='center', va='center',
                   fontsize=9, color='white', weight='bold')

        # Draw prey
        for i, (pos, vel) in enumerate(zip(self.prey_pos, self.prey_vel)):
            if self.prey_caught[i]:
                circle = plt.Circle(pos, self.prey_size, color='gray', alpha=0.3)
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], 'X', ha='center', va='center',
                       fontsize=12, color='black', weight='bold')
            else:
                circle = plt.Circle(pos, self.prey_size, color='blue', alpha=0.7)
                ax.add_patch(circle)
                ax.arrow(pos[0], pos[1], vel[0]*0.3, vel[1]*0.3,
                        head_width=0.05, head_length=0.05, fc='blue', ec='blue')
                ax.text(pos[0], pos[1], f'R{i}', ha='center', va='center',
                       fontsize=9, color='white', weight='bold')

        ax.legend(['Predators', 'Prey'], loc='upper right')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if mode == 'human':
            plt.show()

        plt.close()
