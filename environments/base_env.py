"""
Base Multi-Agent Environment Interface
Defines the standard interface for all multi-agent environments
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any


class BaseMultiAgentEnv(ABC):
    """Abstract base class for multi-agent environments"""

    def __init__(self, num_agents: int, state_dim: int, action_dim: int):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_step = 0
        self.max_steps = 100

    @abstractmethod
    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial observations for all agents"""
        pass

    @abstractmethod
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """
        Execute one step in the environment

        Args:
            actions: List of actions for each agent

        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            dones: List of done flags for each agent
            info: Dictionary with additional information
        """
        pass

    @abstractmethod
    def get_global_state(self) -> np.ndarray:
        """Return global state for centralized critic (CTDE)"""
        pass

    @abstractmethod
    def render(self, mode: str = 'human'):
        """Render the environment"""
        pass

    def get_env_info(self) -> Dict[str, Any]:
        """Return environment configuration information"""
        return {
            'num_agents': self.num_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_steps': self.max_steps
        }
