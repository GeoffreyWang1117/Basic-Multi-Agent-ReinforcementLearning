"""
Critic Networks for Multi-Agent RL (CTDE paradigm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralizedCritic(nn.Module):
    """
    Centralized Critic for CTDE (Centralized Training, Decentralized Execution)
    Takes global state and all agents' actions as input
    """

    def __init__(self, global_state_dim: int, total_action_dim: int,
                 hidden_dim: int = 256):
        super(CentralizedCritic, self).__init__()

        self.fc1 = nn.Linear(global_state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, global_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            global_state: (batch_size, global_state_dim)
            actions: (batch_size, total_action_dim) - all agents' actions concatenated

        Returns:
            q_value: (batch_size, 1)
        """
        x = torch.cat([global_state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class AttentionCritic(nn.Module):
    """
    Attention-based Critic that can handle variable number of agents
    Based on "Actor-Attention-Critic for Multi-Agent Reinforcement Learning"
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, attention_dim: int = 128):
        super(AttentionCritic, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim)
        )

        # Attention mechanism
        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)
        self.value = nn.Linear(attention_dim, attention_dim)

        # Output layers
        self.fc1 = nn.Linear(attention_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.attention_dim = attention_dim

        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.state_encoder, self.action_encoder,
                      self.query, self.key, self.value, self.fc1, self.fc2]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, own_state: torch.Tensor, own_action: torch.Tensor,
                other_states: torch.Tensor, other_actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention over other agents

        Args:
            own_state: (batch_size, state_dim)
            own_action: (batch_size, action_dim)
            other_states: (batch_size, num_other_agents, state_dim)
            other_actions: (batch_size, num_other_agents, action_dim)

        Returns:
            q_value: (batch_size, 1)
        """
        batch_size = own_state.size(0)
        num_others = other_states.size(1) if other_states.dim() > 2 else 0

        # Encode own state and action
        own_state_emb = self.state_encoder(own_state)  # (batch, attention_dim)
        own_action_emb = self.action_encoder(own_action)  # (batch, attention_dim)
        own_emb = own_state_emb + own_action_emb

        if num_others > 0:
            # Encode other agents
            other_state_emb = self.state_encoder(
                other_states.view(-1, other_states.size(-1))
            ).view(batch_size, num_others, self.attention_dim)

            other_action_emb = self.action_encoder(
                other_actions.view(-1, other_actions.size(-1))
            ).view(batch_size, num_others, self.attention_dim)

            other_emb = other_state_emb + other_action_emb  # (batch, num_others, attention_dim)

            # Attention mechanism
            query = self.query(own_emb).unsqueeze(1)  # (batch, 1, attention_dim)
            keys = self.key(other_emb)  # (batch, num_others, attention_dim)
            values = self.value(other_emb)  # (batch, num_others, attention_dim)

            # Scaled dot-product attention
            attention_scores = torch.bmm(query, keys.transpose(1, 2))  # (batch, 1, num_others)
            attention_scores = attention_scores / (self.attention_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)

            attended = torch.bmm(attention_weights, values).squeeze(1)  # (batch, attention_dim)

            # Combine own embedding with attended others
            combined = torch.cat([own_emb, attended], dim=1)
        else:
            # No other agents, use zero padding
            combined = torch.cat([own_emb, torch.zeros_like(own_emb)], dim=1)

        # Output Q-value
        x = F.relu(self.fc1(combined))
        q_value = self.fc2(x)

        return q_value
