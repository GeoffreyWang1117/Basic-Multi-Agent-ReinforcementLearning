import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

# 定义 Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = np.random.choice(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# 定义 Actor 网络
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# 定义 Critic 网络
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义 Agent
class Agent:
    def __init__(self, state_size, action_size, num_agents, lr_actor=1e-4, lr_critic=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.actor_local = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.critic_local = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)

        self.memory = ReplayBuffer(buffer_size=int(1e6), batch_size=64)

        self.gamma = 0.99
        self.tau = 1e-3

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, noise=0.1):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += noise * np.random.randn(self.action_size)
        return np.clip(action, -1, 1)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # 更新 Critic
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# 定义 MADDPG
class MADDPG:
    def __init__(self, state_size, action_size, num_agents):
        self.agents = [Agent(state_size, action_size, num_agents) for _ in range(num_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        for i, agent in enumerate(self.agents):
            agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def act(self, states, noise=0.1):
        return [agent.act(state, noise) for agent in self.agents]

    def save(self, filename):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'{filename}_agent{i}.pth')

    def load(self, filename):
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load(f'{filename}_agent{i}.pth'))
