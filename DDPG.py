import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action

        self.replay_buffer = deque(maxlen=2000000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, next_state, action, reward, done = zip(*batch)

        state = torch.FloatTensor(np.stack(state)).to(device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(device)
        action = torch.FloatTensor(np.stack(action)).to(device)
        reward = torch.FloatTensor(np.stack(reward)).to(device)
        done = torch.FloatTensor(np.stack(done)).to(device)

        target_q = self.critic_target(next_state, self.actor_target(next_state)).squeeze()
        target_q = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action).squeeze()

        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, next_state, action, reward, done))

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))

# 主函数部分
if __name__ == "__main__":
    import gym

    env = gym.make("Pendulum-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    ddpg = DDPG(state_dim, action_dim, max_action)

    episodes = 500
for episode in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # 处理元组情况
    episode_reward = 0

    while True:
        action = ddpg.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)  # 处理五个返回值
        done = done or truncated  # 处理完整的结束条件
        ddpg.store_transition(state, action, reward, next_state, done)

        ddpg.train()

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}: Reward: {episode_reward}")

    ddpg.save("ddpg_pendulum")
