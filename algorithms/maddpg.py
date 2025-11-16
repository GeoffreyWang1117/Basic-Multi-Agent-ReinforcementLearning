"""
MADDPG: Multi-Agent Deep Deterministic Policy Gradient
Implements classic CTDE paradigm with centralized critic

Reference: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (Lowe et al., 2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import copy

from networks.actor import MLPActor
from networks.critic import CentralizedCritic
from utils.replay_buffer import MultiAgentReplayBuffer


class MADDPGAgent:
    """Single agent in MADDPG framework"""

    def __init__(self, agent_id: int, state_dim: int, action_dim: int,
                 global_state_dim: int, total_action_dim: int,
                 hidden_dim: int = 256, lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3, gamma: float = 0.99,
                 tau: float = 0.01, device: str = 'cpu'):

        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Actor network (decentralized - only uses local observation)
        self.actor = MLPActor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic network (centralized - uses global state and all actions)
        self.critic = CentralizedCritic(global_state_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        """
        Select action using actor network

        Args:
            state: Local observation
            noise: Noise level for exploration

        Returns:
            action: Action in range [-1, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()

        # Add exploration noise
        if noise > 0:
            action += noise * np.random.randn(len(action))
            action = np.clip(action, -1, 1)

        return action

    def update_critic(self, global_state: torch.Tensor, all_actions: torch.Tensor,
                     rewards: torch.Tensor, next_global_state: torch.Tensor,
                     next_all_actions: torch.Tensor, dones: torch.Tensor) -> float:
        """
        Update critic network

        Args:
            global_state: (batch_size, global_state_dim)
            all_actions: (batch_size, total_action_dim)
            rewards: (batch_size, 1) - rewards for this agent
            next_global_state: (batch_size, global_state_dim)
            next_all_actions: (batch_size, total_action_dim)
            dones: (batch_size, 1)

        Returns:
            critic_loss: Scalar loss value
        """
        # Compute target Q-value
        with torch.no_grad():
            q_next = self.critic_target(next_global_state, next_all_actions)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Compute current Q-value
        q_current = self.critic(global_state, all_actions)

        # Critic loss
        critic_loss = nn.MSELoss()(q_current, q_target)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, states: torch.Tensor, all_actions: torch.Tensor,
                    global_state: torch.Tensor, agent_idx: int) -> float:
        """
        Update actor network using policy gradient

        Args:
            states: (batch_size, state_dim) - local observations
            all_actions: (batch_size, total_action_dim) - all agents' actions
            global_state: (batch_size, global_state_dim)
            agent_idx: Index to replace this agent's actions

        Returns:
            actor_loss: Scalar loss value
        """
        # Get this agent's actions from actor
        actions_pred = self.actor(states)

        # Replace this agent's actions in all_actions
        all_actions_clone = all_actions.clone()
        action_dim = actions_pred.size(1)
        all_actions_clone[:, agent_idx * action_dim:(agent_idx + 1) * action_dim] = actions_pred

        # Actor loss: maximize Q-value
        actor_loss = -self.critic(global_state, all_actions_clone).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return actor_loss.item()

    def soft_update(self):
        """Soft update of target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class MADDPG:
    """
    MADDPG: Multi-Agent DDPG
    Centralized Training with Decentralized Execution (CTDE)
    """

    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 global_state_dim: int = None, hidden_dim: int = 256,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.01,
                 buffer_size: int = 100000, batch_size: int = 256,
                 device: str = 'cpu'):

        self.num_agents = num_agents
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.device = device

        # If global state dim not provided, concatenate all local observations
        if global_state_dim is None:
            global_state_dim = num_agents * state_dim

        total_action_dim = num_agents * action_dim

        # Create agents
        self.agents = [
            MADDPGAgent(
                agent_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                global_state_dim=global_state_dim,
                total_action_dim=total_action_dim,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                device=device
            )
            for i in range(num_agents)
        ]

        # Shared replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(buffer_size, num_agents)

    def act(self, states: List[np.ndarray], noise: float = 0.0) -> List[np.ndarray]:
        """Get actions for all agents"""
        return [agent.act(state, noise) for agent, state in zip(self.agents, states)]

    def step(self, states: List[np.ndarray], actions: List[np.ndarray],
             rewards: List[float], next_states: List[np.ndarray],
             dones: List[bool], global_state: np.ndarray = None,
             next_global_state: np.ndarray = None):
        """Store transition in replay buffer"""
        self.replay_buffer.add(states, actions, rewards, next_states, dones,
                              global_state, next_global_state)

    def update(self) -> dict:
        """
        Update all agents

        Returns:
            info: Dictionary with loss information
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample from replay buffer
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
            # Concatenate all observations
            global_states = torch.cat(states, dim=1)
            next_global_states = torch.cat(next_states, dim=1)

        # Concatenate all actions
        all_actions = torch.cat(actions, dim=1)

        # Get next actions from target actors
        with torch.no_grad():
            next_actions = [agent.actor_target(next_s) for agent, next_s in zip(self.agents, next_states)]
            next_all_actions = torch.cat(next_actions, dim=1)

        # Update each agent
        info = {}
        for i, agent in enumerate(self.agents):
            # Update critic
            critic_loss = agent.update_critic(
                global_states,
                all_actions,
                rewards[:, i].unsqueeze(1),
                next_global_states,
                next_all_actions,
                dones[:, i].unsqueeze(1)
            )

            # Update actor
            actor_loss = agent.update_actor(
                states[i],
                all_actions,
                global_states,
                agent_idx=i
            )

            # Soft update target networks
            agent.soft_update()

            info[f'agent_{i}_critic_loss'] = critic_loss
            info[f'agent_{i}_actor_loss'] = actor_loss

        return info

    def save(self, filepath: str):
        """Save all agent models"""
        checkpoint = {
            f'agent_{i}_actor': agent.actor.state_dict(),
            f'agent_{i}_critic': agent.critic.state_dict()
            for i, agent in enumerate(self.agents)
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load all agent models"""
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint[f'agent_{i}_actor'])
            agent.critic.load_state_dict(checkpoint[f'agent_{i}_critic'])
            agent.actor_target = copy.deepcopy(agent.actor)
            agent.critic_target = copy.deepcopy(agent.critic)
