"""
QMIX Networks: Mixing Network and Q-Networks for Value Decomposition

Reference: "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning" (Rashid et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """
    Individual Q-Network for each agent
    Takes individual observation and outputs Q-values for each action
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: (batch_size, state_dim) or (batch_size, num_agents, state_dim)

        Returns:
            q_values: (batch_size, action_dim) or (batch_size, num_agents, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class MixingNetwork(nn.Module):
    """
    QMIX Mixing Network
    Combines individual Q-values into a global Q-value using hypernetworks
    Ensures monotonicity: dQ_tot/dQ_i >= 0 for all agents i
    """

    def __init__(self, num_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        super(MixingNetwork, self).__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        # Hypernetwork for first layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, num_agents * mixing_embed_dim)
        )

        # Hypernetwork for first layer biases
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )

        # Hypernetwork for second layer weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )

        # Hypernetwork for second layer bias (single scalar)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Mix agent Q-values into global Q-value

        Args:
            agent_q_values: (batch_size, num_agents) - Q-values for each agent's selected action
            global_state: (batch_size, state_dim) - global state

        Returns:
            q_total: (batch_size, 1) - global Q-value
        """
        batch_size = agent_q_values.size(0)

        # Ensure agent_q_values has shape (batch_size, num_agents)
        if agent_q_values.dim() == 1:
            agent_q_values = agent_q_values.unsqueeze(0)

        agent_q_values = agent_q_values.view(batch_size, -1, 1)  # (batch, num_agents, 1)

        # Generate weights and biases from global state using hypernetworks
        # First layer
        w1 = torch.abs(self.hyper_w1(global_state))  # (batch, num_agents * mixing_embed_dim)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)  # (batch, num_agents, mixing_embed_dim)

        b1 = self.hyper_b1(global_state)  # (batch, mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)  # (batch, 1, mixing_embed_dim)

        # First layer computation: (batch, 1, num_agents) @ (batch, num_agents, mixing_embed_dim)
        hidden = torch.bmm(agent_q_values.permute(0, 2, 1), w1) + b1  # (batch, 1, mixing_embed_dim)
        hidden = F.elu(hidden)

        # Second layer
        w2 = torch.abs(self.hyper_w2(global_state))  # (batch, mixing_embed_dim)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)  # (batch, mixing_embed_dim, 1)

        b2 = self.hyper_b2(global_state)  # (batch, 1)
        b2 = b2.view(batch_size, 1, 1)  # (batch, 1, 1)

        # Second layer computation
        q_total = torch.bmm(hidden, w2) + b2  # (batch, 1, 1)
        q_total = q_total.view(batch_size, 1)  # (batch, 1)

        return q_total


class RecurrentQNetwork(nn.Module):
    """
    Recurrent Q-Network using GRU for partial observability
    Useful for environments where agents need memory
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, gru_hidden_dim: int = 64):
        super(RecurrentQNetwork, self).__init__()

        self.gru_hidden_dim = gru_hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(gru_hidden_dim, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, state: torch.Tensor, hidden_state: torch.Tensor = None):
        """
        Forward pass with recurrent state

        Args:
            state: (batch_size, seq_len, state_dim) or (batch_size, state_dim)
            hidden_state: (1, batch_size, gru_hidden_dim)

        Returns:
            q_values: (batch_size, action_dim)
            new_hidden: (1, batch_size, gru_hidden_dim)
        """
        # Handle different input shapes
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension

        batch_size = state.size(0)

        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.gru_hidden_dim, device=state.device)

        # Encode state
        x = F.relu(self.fc1(state))

        # GRU
        gru_out, new_hidden = self.gru(x, hidden_state)

        # Take last output
        x = gru_out[:, -1, :]

        # Q-values
        q_values = self.fc2(x)

        return q_values, new_hidden

    def init_hidden(self, batch_size: int, device: str = 'cpu'):
        """Initialize hidden state"""
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
