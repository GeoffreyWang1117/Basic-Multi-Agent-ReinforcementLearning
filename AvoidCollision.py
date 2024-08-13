import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random

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
        experiences = random.sample(self.memory, k=self.batch_size)  # 使用 random.sample

        states = torch.tensor(np.array([e.state for e in experiences])).float()
        actions = torch.tensor(np.array([e.action for e in experiences])).float()
        rewards = torch.tensor(np.array([e.reward for e in experiences])).float()
        next_states = torch.tensor(np.array([e.next_state for e in experiences])).float()
        dones = torch.tensor(np.array([e.done for e in experiences]).astype(np.uint8)).float()

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

        # 调整 actions 的形状
        actions = actions.view(actions.size(0), -1)  # 展平 actions 到 (batch_size, action_size * num_agents)

        # 更新 Critic
        actions_next = self.actor_target(next_states).view(actions.size(0), -1)  # 调整下一步的动作形状
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        actions_pred = self.actor_local(states).view(actions.size(0), -1)  # 调整预测动作的形状
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
        return [agent.act(state, noise) for agent, state in zip(self.agents, states)]

    def save(self, filename):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'{filename}_agent{i}.pth')

    def load(self, filename):
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load(f'{filename}_agent{i}.pth'))

# 自定义的简单多智能体环境
class SimpleMultiAgentEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.num_agents = 2
        self.reset()

    def reset(self):
        self.agents_positions = [np.array([0, 0]), np.array([self.grid_size - 1, self.grid_size - 1])]
        self.goals_positions = [np.array([self.grid_size - 1, self.grid_size - 1]), np.array([0, 0])]
        return self._get_observation()

    def _get_observation(self):
        return [self.agents_positions[0] - self.goals_positions[0],
                self.agents_positions[1] - self.goals_positions[1]]

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        dones = np.zeros(self.num_agents, dtype=bool)
        for i in range(self.num_agents):
            action = np.argmax(actions[i]) if actions[i].ndim > 0 else int(actions[i])
            if action == 0:  # Move up
                self.agents_positions[i][1] = max(0, self.agents_positions[i][1] - 1)
            elif action == 1:  # Move down
                self.agents_positions[i][1] = min(self.grid_size - 1, self.agents_positions[i][1] + 1)
            elif action == 2:  # Move left
                self.agents_positions[i][0] = max(0, self.agents_positions[i][0] - 1)
            elif action == 3:  # Move right
                self.agents_positions[i][0] = min(self.grid_size - 1, self.agents_positions[i][0] + 1)

        # Check for collisions
        if np.array_equal(self.agents_positions[0], self.agents_positions[1]):
            rewards -= 10  # Negative reward for collision

        for i in range(self.num_agents):
            distance = np.linalg.norm(self.agents_positions[i] - self.goals_positions[i])
            rewards[i] += -distance  # Negative reward for being far from goal
            dones[i] = np.array_equal(self.agents_positions[i], self.goals_positions[i])

        next_observation = self._get_observation()
        return next_observation, rewards, dones



    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid.fill('-')
        for pos in self.agents_positions:
            grid[tuple(pos)] = 'A'
        for pos in self.goals_positions:
            grid[tuple(pos)] = 'G'
        print(grid)

# 初始化环境和MADDPG算法
env = SimpleMultiAgentEnv(grid_size=5)
maddpg = MADDPG(state_size=2, action_size=4, num_agents=2)

# 训练过程
for episode in range(1000):
    states = env.reset()
    done = False
    total_rewards = np.zeros(env.num_agents)
    
    while not done:
        actions = maddpg.act(states)
        next_states, rewards, dones = env.step(actions)
        maddpg.step(states, actions, rewards, next_states, dones)
        states = next_states
        total_rewards += rewards
        done = np.all(dones)  # 如果所有智能体都达到目标位置，则结束当前回合

    print(f'Episode {episode}, Total Rewards: {total_rewards}')
    env.render()

# 保存模型
maddpg.save('maddpg_agents')
