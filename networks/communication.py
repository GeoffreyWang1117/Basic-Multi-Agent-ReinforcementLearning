"""
Communication Modules for Multi-Agent RL
Implements CommNet and TarMAC-style communication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommNet(nn.Module):
    """
    CommNet: Communication Neural Network
    Allows agents to communicate through a shared communication channel

    Reference: "Learning Multiagent Communication with Backpropagation" (Sukhbaatar et al., 2016)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_comm_rounds: int = 1):
        super(CommNet, self).__init__()

        self.num_comm_rounds = num_comm_rounds
        self.hidden_dim = hidden_dim

        # Encoding layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Communication layers (applied iteratively)
        self.comm_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.skip_connection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, observations: torch.Tensor):
        """
        Forward pass with communication

        Args:
            observations: (batch_size, num_agents, input_dim)

        Returns:
            hidden_states: (batch_size, num_agents, hidden_dim)
        """
        batch_size, num_agents, _ = observations.shape

        # Encode observations
        h = self.encoder(observations)  # (batch, num_agents, hidden)

        # Communication rounds
        for _ in range(self.num_comm_rounds):
            # Average communication from other agents
            comm_sum = h.sum(dim=1, keepdim=True)  # (batch, 1, hidden)
            comm_avg = (comm_sum - h) / (num_agents - 1)  # Exclude self

            # Update hidden states with communication
            h_new = self.comm_layer(h + comm_avg)

            # Skip connection
            h = h_new + self.skip_connection(h)

        return h


class TarMACAttention(nn.Module):
    """
    TarMAC: Targeted Multi-Agent Communication
    Uses attention mechanism for selective communication

    Reference: "TarMAC: Targeted Multi-Agent Communication" (Das et al., 2019)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_comm_rounds: int = 1):
        super(TarMACAttention, self).__init__()

        self.num_comm_rounds = num_comm_rounds
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Encoding
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # Multi-head attention for communication
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Message processing
        self.message_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, observations: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass with targeted communication

        Args:
            observations: (batch_size, num_agents, input_dim)
            mask: (batch_size, num_agents, num_agents) attention mask

        Returns:
            hidden_states: (batch_size, num_agents, hidden_dim)
        """
        batch_size, num_agents, _ = observations.shape

        # Encode observations
        h = F.relu(self.encoder(observations))  # (batch, num_agents, hidden)

        # Communication rounds
        for _ in range(self.num_comm_rounds):
            # Self-attention for communication
            h_residual = h

            # Multi-head attention
            attn_output, attn_weights = self.attention(
                query=h,
                key=h,
                value=h,
                attn_mask=mask,
                need_weights=True
            )

            # Add & Norm
            h = self.layer_norm1(h_residual + attn_output)

            # Feed-forward
            h_residual = h
            h_processed = self.message_processor(h)
            h = self.layer_norm2(h_residual + h_processed)

        return h


class MessagePool(nn.Module):
    """
    Simple message pooling mechanism
    Each agent broadcasts a message, others aggregate
    """

    def __init__(self, input_dim: int, message_dim: int = 64,
                 hidden_dim: int = 128, pooling: str = 'mean'):
        super(MessagePool, self).__init__()

        self.pooling = pooling  # 'mean', 'max', or 'sum'

        # Message generation
        self.message_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

        # Message integration
        self.message_integrator = nn.Sequential(
            nn.Linear(input_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, observations: torch.Tensor):
        """
        Forward pass

        Args:
            observations: (batch_size, num_agents, input_dim)

        Returns:
            integrated: (batch_size, num_agents, hidden_dim)
        """
        batch_size, num_agents, input_dim = observations.shape

        # Generate messages from all agents
        messages = self.message_encoder(observations)  # (batch, num_agents, message_dim)

        # Pool messages (exclude self message)
        if self.pooling == 'mean':
            message_sum = messages.sum(dim=1, keepdim=True)
            pooled = (message_sum - messages) / max(num_agents - 1, 1)
        elif self.pooling == 'max':
            # For max pooling, we need to handle self-exclusion differently
            pooled = []
            for i in range(num_agents):
                others_mask = torch.ones(num_agents, dtype=torch.bool)
                others_mask[i] = False
                other_messages = messages[:, others_mask, :]
                pooled_i = other_messages.max(dim=1, keepdim=False)[0]
                pooled.append(pooled_i)
            pooled = torch.stack(pooled, dim=1)
        else:  # sum
            message_sum = messages.sum(dim=1, keepdim=True)
            pooled = message_sum - messages

        # Integrate pooled messages with own observation
        combined = torch.cat([observations, pooled], dim=-1)
        integrated = self.message_integrator(combined)

        return integrated
