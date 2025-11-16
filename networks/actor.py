"""
Actor Networks for Multi-Agent RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPActor(nn.Module):
    """Standard MLP Actor network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(MLPActor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: (batch_size, state_dim)

        Returns:
            action: (batch_size, action_dim) in range [-1, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


class RecurrentActor(nn.Module):
    """Recurrent Actor with GRU for partial observability"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, gru_hidden_dim: int = 128):
        super(RecurrentActor, self).__init__()

        self.gru_hidden_dim = gru_hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(gru_hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, hidden_state: torch.Tensor = None):
        """
        Forward pass with recurrent state

        Args:
            state: (batch_size, seq_len, state_dim) or (batch_size, state_dim)
            hidden_state: (1, batch_size, gru_hidden_dim)

        Returns:
            action: (batch_size, action_dim)
            new_hidden: (1, batch_size, gru_hidden_dim)
        """
        # Handle different input shapes
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension

        batch_size = state.size(0)

        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.gru_hidden_dim,
                                      device=state.device)

        # Encode state
        x = F.relu(self.fc1(state))

        # GRU
        gru_out, new_hidden = self.gru(x, hidden_state)

        # Take last output
        x = gru_out[:, -1, :]

        # Decode to action
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))

        return action, new_hidden

    def init_hidden(self, batch_size: int, device: str = 'cpu'):
        """Initialize hidden state"""
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
