"""
Replay Buffer for Multi-Agent RL
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict


class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent reinforcement learning
    Stores transitions: (states, actions, rewards, next_states, dones, global_state, next_global_state)
    """

    def __init__(self, buffer_size: int = 100000, num_agents: int = 2):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)

    def add(self, states: List[np.ndarray], actions: List[np.ndarray],
            rewards: List[float], next_states: List[np.ndarray],
            dones: List[bool], global_state: np.ndarray = None,
            next_global_state: np.ndarray = None):
        """
        Add a transition to the buffer

        Args:
            states: List of states for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_states: List of next states for each agent
            dones: List of done flags for each agent
            global_state: Optional global state for centralized critic
            next_global_state: Optional next global state
        """
        experience = {
            'states': [np.array(s, dtype=np.float32) for s in states],
            'actions': [np.array(a, dtype=np.float32) for a in actions],
            'rewards': np.array(rewards, dtype=np.float32),
            'next_states': [np.array(s, dtype=np.float32) for s in next_states],
            'dones': np.array(dones, dtype=np.float32),
            'global_state': np.array(global_state, dtype=np.float32) if global_state is not None else None,
            'next_global_state': np.array(next_global_state, dtype=np.float32) if next_global_state is not None else None
        }
        self.memory.append(experience)

    def sample(self, batch_size: int) -> Dict:
        """
        Sample a batch of experiences

        Returns:
            batch: Dictionary containing batched tensors
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        experiences = random.sample(self.memory, batch_size)

        # Stack experiences
        batch = {
            'states': [
                np.stack([exp['states'][i] for exp in experiences])
                for i in range(self.num_agents)
            ],
            'actions': [
                np.stack([exp['actions'][i] for exp in experiences])
                for i in range(self.num_agents)
            ],
            'rewards': np.stack([exp['rewards'] for exp in experiences]),
            'next_states': [
                np.stack([exp['next_states'][i] for exp in experiences])
                for i in range(self.num_agents)
            ],
            'dones': np.stack([exp['dones'] for exp in experiences])
        }

        # Add global states if available
        if experiences[0]['global_state'] is not None:
            batch['global_states'] = np.stack([exp['global_state'] for exp in experiences])
            batch['next_global_states'] = np.stack([exp['next_global_state'] for exp in experiences])

        return batch

    def __len__(self):
        return len(self.memory)

    def clear(self):
        """Clear the buffer"""
        self.memory.clear()


class PrioritizedReplayBuffer(MultiAgentReplayBuffer):
    """
    Prioritized Experience Replay for Multi-Agent RL
    Uses TD-error for prioritization
    """

    def __init__(self, buffer_size: int = 100000, num_agents: int = 2,
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        super().__init__(buffer_size, num_agents)

        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling weight
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0

    def add(self, states, actions, rewards, next_states, dones,
            global_state=None, next_global_state=None):
        """Add with max priority for new experiences"""
        super().add(states, actions, rewards, next_states, dones,
                   global_state, next_global_state)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[Dict, np.ndarray, List[int]]:
        """
        Sample with prioritization

        Returns:
            batch: Dictionary of batched experiences
            weights: Importance sampling weights
            indices: Indices of sampled experiences
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Get experiences
        experiences = [self.memory[idx] for idx in indices]

        # Stack experiences
        batch = {
            'states': [
                np.stack([exp['states'][i] for exp in experiences])
                for i in range(self.num_agents)
            ],
            'actions': [
                np.stack([exp['actions'][i] for exp in experiences])
                for i in range(self.num_agents)
            ],
            'rewards': np.stack([exp['rewards'] for exp in experiences]),
            'next_states': [
                np.stack([exp['next_states'][i] for exp in experiences])
                for i in range(self.num_agents)
            ],
            'dones': np.stack([exp['dones'] for exp in experiences])
        }

        if experiences[0]['global_state'] is not None:
            batch['global_states'] = np.stack([exp['global_state'] for exp in experiences])
            batch['next_global_states'] = np.stack([exp['next_global_state'] for exp in experiences])

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, weights, indices.tolist()

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
