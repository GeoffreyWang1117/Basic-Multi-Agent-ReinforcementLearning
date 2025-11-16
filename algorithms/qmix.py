"""
QMIX: Monotonic Value Function Factorisation for Multi-Agent RL

Reference: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning" (2018)

Key Features:
- Value decomposition: Q_tot(s, a) = f(Q_1, Q_2, ..., Q_n, s)
- Monotonicity constraint: dQ_tot/dQ_i >= 0 for all agents
- Centralized training, decentralized execution (CTDE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
import copy

from networks.qmix import QNetwork, MixingNetwork
from utils.replay_buffer import MultiAgentReplayBuffer


class QMIX:
    """
    QMIX Algorithm for Multi-Agent Reinforcement Learning
    Uses value decomposition with monotonicity constraint
    """

    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 global_state_dim: int = None, hidden_dim: int = 128,
                 mixing_embed_dim: int = 32, lr: float = 5e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 buffer_size: int = 100000, batch_size: int = 256,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.9995, device: str = 'cpu'):

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Global state dimension
        if global_state_dim is None:
            global_state_dim = num_agents * state_dim

        self.global_state_dim = global_state_dim

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Create Q-Networks for each agent
        self.q_networks = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim).to(device)
            for _ in range(num_agents)
        ])

        self.q_networks_target = nn.ModuleList([
            copy.deepcopy(q_net) for q_net in self.q_networks
        ])

        # Create Mixing Network
        self.mixing_network = MixingNetwork(num_agents, global_state_dim, mixing_embed_dim).to(device)
        self.mixing_network_target = copy.deepcopy(self.mixing_network)

        # Optimizer for all networks
        params = list(self.mixing_network.parameters())
        for q_net in self.q_networks:
            params += list(q_net.parameters())

        self.optimizer = optim.Adam(params, lr=lr)

        # Replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(buffer_size, num_agents)

    def act(self, states: List[np.ndarray], epsilon: float = None) -> List[int]:
        """
        Select actions for all agents using epsilon-greedy policy

        Args:
            states: List of observations for each agent
            epsilon: Exploration rate (if None, uses self.epsilon)

        Returns:
            actions: List of discrete action indices
        """
        if epsilon is None:
            epsilon = self.epsilon

        actions = []

        for i, state in enumerate(states):
            if np.random.rand() < epsilon:
                # Random action
                action = np.random.randint(self.action_dim)
            else:
                # Greedy action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                self.q_networks[i].eval()
                with torch.no_grad():
                    q_values = self.q_networks[i](state_tensor)
                    action = q_values.argmax(dim=1).item()
                self.q_networks[i].train()

            actions.append(action)

        return actions

    def step(self, states: List[np.ndarray], actions: List[int],
             rewards: List[float], next_states: List[np.ndarray],
             dones: List[bool], global_state: np.ndarray = None,
             next_global_state: np.ndarray = None):
        """Store transition in replay buffer"""
        # Convert discrete actions to one-hot for storage (or just keep as indices)
        self.replay_buffer.add(states, actions, rewards, next_states, dones,
                              global_state, next_global_state)

    def update(self) -> dict:
        """
        Update QMIX networks

        Returns:
            info: Dictionary with loss information
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = [torch.FloatTensor(s).to(self.device) for s in batch['states']]
        actions = [torch.LongTensor(a).to(self.device) for a in batch['actions']]
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = [torch.FloatTensor(s).to(self.device) for s in batch['next_states']]
        dones = torch.FloatTensor(batch['dones']).to(self.device)

        # Global states
        if 'global_states' in batch:
            global_states = torch.FloatTensor(batch['global_states']).to(self.device)
            next_global_states = torch.FloatTensor(batch['next_global_states']).to(self.device)
        else:
            # Concatenate all observations
            global_states = torch.cat(states, dim=1)
            next_global_states = torch.cat(next_states, dim=1)

        # Get current Q-values for chosen actions
        chosen_action_qvals = []
        for i in range(self.num_agents):
            q_vals = self.q_networks[i](states[i])  # (batch, action_dim)
            chosen_q = q_vals.gather(1, actions[i].unsqueeze(1))  # (batch, 1)
            chosen_action_qvals.append(chosen_q)

        chosen_action_qvals = torch.cat(chosen_action_qvals, dim=1)  # (batch, num_agents)

        # Mix Q-values using mixing network
        q_total = self.mixing_network(chosen_action_qvals, global_states)  # (batch, 1)

        # Get target Q-values
        with torch.no_grad():
            # Get next Q-values from target networks
            target_next_qvals = []
            for i in range(self.num_agents):
                next_q_vals = self.q_networks_target[i](next_states[i])  # (batch, action_dim)
                max_next_q = next_q_vals.max(dim=1, keepdim=True)[0]  # (batch, 1)
                target_next_qvals.append(max_next_q)

            target_next_qvals = torch.cat(target_next_qvals, dim=1)  # (batch, num_agents)

            # Mix target Q-values
            q_total_target = self.mixing_network_target(target_next_qvals, next_global_states)

            # Compute TD target
            # Use global reward (sum of all agents' rewards)
            global_rewards = rewards.sum(dim=1, keepdim=True)  # (batch, 1)
            global_dones = dones.max(dim=1, keepdim=True)[0]  # (batch, 1)

            td_target = global_rewards + self.gamma * q_total_target * (1 - global_dones)

        # Compute loss
        loss = nn.MSELoss()(q_total, td_target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 10)
        for q_net in self.q_networks:
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10)
        self.optimizer.step()

        # Soft update target networks
        self.soft_update_targets()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            'loss': loss.item(),
            'q_total_mean': q_total.mean().item(),
            'epsilon': self.epsilon
        }

    def soft_update_targets(self):
        """Soft update of target networks"""
        # Update mixing network
        for target_param, param in zip(self.mixing_network_target.parameters(),
                                       self.mixing_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update Q-networks
        for i in range(self.num_agents):
            for target_param, param in zip(self.q_networks_target[i].parameters(),
                                          self.q_networks[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save all models"""
        checkpoint = {
            'mixing_network': self.mixing_network.state_dict(),
            **{f'q_network_{i}': q_net.state_dict() for i, q_net in enumerate(self.q_networks)},
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load all models"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        self.mixing_network_target = copy.deepcopy(self.mixing_network)

        for i in range(self.num_agents):
            self.q_networks[i].load_state_dict(checkpoint[f'q_network_{i}'])
            self.q_networks_target[i] = copy.deepcopy(self.q_networks[i])

        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
