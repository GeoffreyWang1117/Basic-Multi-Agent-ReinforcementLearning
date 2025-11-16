"""
CommMADDPG: MADDPG with Communication
Adds CommNet-style communication to enhance coordination

This extends MADDPG by allowing agents to exchange information through a communication module
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
import copy

from networks.actor import MLPActor
from networks.critic import CentralizedCritic
from networks.communication import CommNet, TarMACAttention
from utils.replay_buffer import MultiAgentReplayBuffer


class CommActor(nn.Module):
    """Actor with communication module"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 comm_type: str = 'commnet', num_comm_rounds: int = 1):
        super(CommActor, self).__init__()

        self.comm_type = comm_type

        # Communication module
        if comm_type == 'commnet':
            self.comm_module = CommNet(state_dim, hidden_dim, num_comm_rounds)
        elif comm_type == 'tarmac':
            self.comm_module = TarMACAttention(state_dim, hidden_dim, num_heads=4,
                                              num_comm_rounds=num_comm_rounds)
        else:
            raise ValueError(f"Unknown comm_type: {comm_type}")

        # Policy head (takes communicated hidden state to action)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.policy_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(self, observations: torch.Tensor, agent_idx: int = None):
        """
        Forward pass with communication

        Args:
            observations: (batch_size, num_agents, state_dim) or (batch_size, state_dim)
            agent_idx: If observations is batched per-agent, specify which agent

        Returns:
            actions: (batch_size, num_agents, action_dim) or (batch_size, action_dim)
        """
        if observations.dim() == 2:
            # Single agent observation, expand to (batch, 1, state_dim)
            observations = observations.unsqueeze(1)
            single_agent = True
        else:
            single_agent = False

        # Communication
        hidden = self.comm_module(observations)  # (batch, num_agents, hidden_dim)

        # Generate actions
        actions = self.policy_head(hidden)  # (batch, num_agents, action_dim)

        if single_agent:
            actions = actions.squeeze(1)
        elif agent_idx is not None:
            # Return only specific agent's action
            actions = actions[:, agent_idx, :]

        return actions


class CommMADDPGAgent:
    """MADDPG Agent with communication"""

    def __init__(self, agent_id: int, state_dim: int, action_dim: int,
                 global_state_dim: int, total_action_dim: int,
                 num_agents: int, hidden_dim: int = 256,
                 comm_type: str = 'commnet', num_comm_rounds: int = 1,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.01, device: str = 'cpu'):

        self.agent_id = agent_id
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Shared communication actor (all agents share this)
        # Will be set externally in CommMADDPG
        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None

        # Individual critic (centralized)
        self.critic = CentralizedCritic(global_state_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, all_states: np.ndarray, noise: float = 0.0) -> np.ndarray:
        """
        Select action using communication actor

        Args:
            all_states: (num_agents, state_dim) - all agents' observations
            noise: Exploration noise level

        Returns:
            action: (action_dim,) - this agent's action
        """
        if self.actor is None:
            raise ValueError("Actor not set. Use set_actor() first.")

        all_states_tensor = torch.FloatTensor(all_states).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            # Get all actions from communication
            all_actions = self.actor(all_states_tensor)  # (1, num_agents, action_dim)
            action = all_actions[0, self.agent_id].cpu().numpy()
        self.actor.train()

        # Add exploration noise
        if noise > 0:
            action += noise * np.random.randn(len(action))
            action = np.clip(action, -1, 1)

        return action

    def set_actor(self, actor: CommActor, actor_target: CommActor, optimizer: optim.Optimizer):
        """Set shared actor network"""
        self.actor = actor
        self.actor_target = actor_target
        self.actor_optimizer = optimizer

    def update_critic(self, global_state: torch.Tensor, all_actions: torch.Tensor,
                     rewards: torch.Tensor, next_global_state: torch.Tensor,
                     next_all_actions: torch.Tensor, dones: torch.Tensor) -> float:
        """Update critic (same as MADDPG)"""
        with torch.no_grad():
            q_next = self.critic_target(next_global_state, next_all_actions)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        q_current = self.critic(global_state, all_actions)
        critic_loss = nn.MSELoss()(q_current, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.item()

    def soft_update_critic(self):
        """Soft update critic target network"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class CommMADDPG:
    """
    MADDPG with Communication
    Uses a shared communication-based actor for all agents
    """

    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 global_state_dim: int = None, hidden_dim: int = 256,
                 comm_type: str = 'commnet', num_comm_rounds: int = 1,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.01,
                 buffer_size: int = 100000, batch_size: int = 256,
                 device: str = 'cpu'):

        self.num_agents = num_agents
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.tau = tau
        self.device = device

        if global_state_dim is None:
            global_state_dim = num_agents * state_dim

        total_action_dim = num_agents * action_dim

        # Shared communication actor for all agents
        self.actor = CommActor(state_dim, action_dim, hidden_dim, comm_type, num_comm_rounds).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Create agents with individual critics
        self.agents = [
            CommMADDPGAgent(
                agent_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                global_state_dim=global_state_dim,
                total_action_dim=total_action_dim,
                num_agents=num_agents,
                hidden_dim=hidden_dim,
                comm_type=comm_type,
                num_comm_rounds=num_comm_rounds,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                device=device
            )
            for i in range(num_agents)
        ]

        # Set shared actor for all agents
        for agent in self.agents:
            agent.set_actor(self.actor, self.actor_target, self.actor_optimizer)

        # Replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(buffer_size, num_agents)

    def act(self, states: List[np.ndarray], noise: float = 0.0) -> List[np.ndarray]:
        """
        Get actions for all agents through communication

        Args:
            states: List of observations for each agent
            noise: Exploration noise

        Returns:
            actions: List of actions for each agent
        """
        all_states = np.array(states)  # (num_agents, state_dim)
        all_states_tensor = torch.FloatTensor(all_states).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            all_actions = self.actor(all_states_tensor)  # (1, num_agents, action_dim)
            all_actions = all_actions[0].cpu().numpy()  # (num_agents, action_dim)
        self.actor.train()

        # Add exploration noise
        if noise > 0:
            all_actions += noise * np.random.randn(*all_actions.shape)
            all_actions = np.clip(all_actions, -1, 1)

        return [all_actions[i] for i in range(self.num_agents)]

    def step(self, states: List[np.ndarray], actions: List[np.ndarray],
             rewards: List[float], next_states: List[np.ndarray],
             dones: List[bool], global_state: np.ndarray = None,
             next_global_state: np.ndarray = None):
        """Store transition in replay buffer"""
        self.replay_buffer.add(states, actions, rewards, next_states, dones,
                              global_state, next_global_state)

    def update(self) -> dict:
        """Update all agents"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = [torch.FloatTensor(s).to(self.device) for s in batch['states']]
        actions = [torch.FloatTensor(a).to(self.device) for a in batch['actions']]
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = [torch.FloatTensor(s).to(self.device) for s in batch['next_states']]
        dones = torch.FloatTensor(batch['dones']).to(self.device)

        # Global states
        if 'global_states' in batch:
            global_states = torch.FloatTensor(batch['global_states']).to(self.device)
            next_global_states = torch.FloatTensor(batch['next_global_states']).to(self.device)
        else:
            global_states = torch.cat(states, dim=1)
            next_global_states = torch.cat(next_states, dim=1)

        # Stack observations for communication: (batch, num_agents, state_dim)
        all_states = torch.stack(states, dim=1)
        all_next_states = torch.stack(next_states, dim=1)

        # Current actions
        all_actions = torch.cat(actions, dim=1)

        # Get next actions from target actor with communication
        with torch.no_grad():
            next_all_actions_raw = self.actor_target(all_next_states)  # (batch, num_agents, action_dim)
            next_all_actions = next_all_actions_raw.view(self.batch_size, -1)  # Flatten to (batch, total_action_dim)

        # Update critics
        critic_losses = []
        for i, agent in enumerate(self.agents):
            critic_loss = agent.update_critic(
                global_states,
                all_actions,
                rewards[:, i].unsqueeze(1),
                next_global_states,
                next_all_actions,
                dones[:, i].unsqueeze(1)
            )
            critic_losses.append(critic_loss)
            agent.soft_update_critic()

        # Update shared actor using all critics
        # Policy gradient: maximize average Q-value across all agents
        pred_all_actions_raw = self.actor(all_states)  # (batch, num_agents, action_dim)
        pred_all_actions = pred_all_actions_raw.view(self.batch_size, -1)  # Flatten

        # Compute actor loss as average of all agents' Q-values
        actor_loss = 0
        for agent in self.agents:
            q_val = agent.critic(global_states, pred_all_actions)
            actor_loss -= q_val.mean()
        actor_loss /= self.num_agents

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Soft update actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        info = {
            'actor_loss': actor_loss.item(),
            'avg_critic_loss': np.mean(critic_losses)
        }

        for i, loss in enumerate(critic_losses):
            info[f'agent_{i}_critic_loss'] = loss

        return info

    def save(self, filepath: str):
        """Save models"""
        checkpoint = {
            'actor': self.actor.state_dict(),
            **{f'agent_{i}_critic': agent.critic.state_dict() for i, agent in enumerate(self.agents)}
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load models"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target = copy.deepcopy(self.actor)

        for i, agent in enumerate(self.agents):
            agent.critic.load_state_dict(checkpoint[f'agent_{i}_critic'])
            agent.critic_target = copy.deepcopy(agent.critic)
