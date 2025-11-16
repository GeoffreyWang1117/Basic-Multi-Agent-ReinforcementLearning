"""
Training script for multi-agent RL algorithms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from tqdm import tqdm

from environments.cooperative_navigation import CooperativeNavigation
from environments.predator_prey import PredatorPrey
from algorithms.maddpg import MADDPG
from algorithms.comm_maddpg import CommMADDPG
from utils.logger import Logger


def train(args):
    """Main training loop"""

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    if args.env == 'cooperative_navigation':
        env = CooperativeNavigation(
            num_agents=args.num_agents,
            num_landmarks=args.num_agents,
            max_steps=args.max_steps
        )
    elif args.env == 'predator_prey':
        env = PredatorPrey(
            num_predators=args.num_agents,
            num_prey=1,
            max_steps=args.max_steps
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    env_info = env.get_env_info()
    print(f"\nEnvironment: {args.env}")
    print(f"Number of agents: {env_info['num_agents']}")
    print(f"State dimension: {env_info['state_dim']}")
    print(f"Action dimension: {env_info['action_dim']}")

    # Create algorithm
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"Using device: {device}\n")

    if args.algorithm == 'maddpg':
        agent = MADDPG(
            num_agents=env_info['num_agents'],
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            hidden_dim=args.hidden_dim,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            device=device
        )
    elif args.algorithm == 'comm_maddpg':
        agent = CommMADDPG(
            num_agents=env_info['num_agents'],
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            hidden_dim=args.hidden_dim,
            comm_type=args.comm_type,
            num_comm_rounds=args.comm_rounds,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            device=device
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Create logger
    logger = Logger(args.log_dir, args.exp_name)
    logger.save_config(vars(args))

    # Training loop
    print("Starting training...")
    global_step = 0
    best_reward = -float('inf')

    for episode in tqdm(range(args.num_episodes), desc="Training"):
        states = env.reset()
        episode_reward = np.zeros(env_info['num_agents'])
        episode_step = 0

        while True:
            # Select actions
            noise = args.noise_start * (args.noise_decay ** episode)
            actions = agent.act(states, noise=noise)

            # Execute actions
            global_state = env.get_global_state()
            next_states, rewards, dones, info = env.step(actions)
            next_global_state = env.get_global_state()

            # Store transition
            agent.step(states, actions, rewards, next_states, dones,
                      global_state, next_global_state)

            # Update agents
            if global_step % args.update_freq == 0:
                update_info = agent.update()
                if update_info:
                    for key, value in update_info.items():
                        logger.log_scalar(key, value, global_step)

            states = next_states
            episode_reward += np.array(rewards)
            episode_step += 1
            global_step += 1

            if all(dones):
                break

        # Log episode
        total_reward = np.sum(episode_reward)
        logger.log_episode(episode, total_reward, episode_step, info)
        logger.log_scalar('episode_reward', total_reward, episode)
        logger.log_scalar('episode_steps', episode_step, episode)

        # Print progress
        if (episode + 1) % args.print_freq == 0:
            recent_rewards = [ep['total_reward'] for ep in logger.episode_data[-args.print_freq:]]
            avg_reward = np.mean(recent_rewards)
            print(f"\nEpisode {episode + 1}/{args.num_episodes}")
            print(f"  Avg Reward (last {args.print_freq}): {avg_reward:.2f}")
            print(f"  Noise: {noise:.4f}")
            if info:
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.2f}")

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(logger.log_dir, 'best_model.pth'))

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            agent.save(os.path.join(logger.log_dir, f'checkpoint_{episode + 1}.pth'))

        # Render
        if args.render and (episode + 1) % args.render_freq == 0:
            env.render(save_path=os.path.join(logger.log_dir, f'render_{episode + 1}.png'))

    # Final save
    agent.save(os.path.join(logger.log_dir, 'final_model.pth'))
    logger.plot_metrics()
    logger.print_summary()

    print(f"\nTraining completed! Results saved to: {logger.log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Agent RL')

    # Environment
    parser.add_argument('--env', type=str, default='cooperative_navigation',
                       choices=['cooperative_navigation', 'predator_prey'],
                       help='Environment name')
    parser.add_argument('--num_agents', type=int, default=3,
                       help='Number of agents')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum steps per episode')

    # Algorithm
    parser.add_argument('--algorithm', type=str, default='maddpg',
                       choices=['maddpg', 'comm_maddpg'],
                       help='Algorithm to use')
    parser.add_argument('--comm_type', type=str, default='commnet',
                       choices=['commnet', 'tarmac'],
                       help='Communication type for comm_maddpg')
    parser.add_argument('--comm_rounds', type=int, default=1,
                       help='Number of communication rounds')

    # Network
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')

    # Training
    parser.add_argument('--num_episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--buffer_size', type=int, default=100000,
                       help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr_actor', type=float, default=1e-4,
                       help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                       help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.01,
                       help='Soft update parameter')
    parser.add_argument('--noise_start', type=float, default=0.5,
                       help='Initial exploration noise')
    parser.add_argument('--noise_decay', type=float, default=0.9999,
                       help='Noise decay rate')
    parser.add_argument('--update_freq', type=int, default=1,
                       help='Update frequency')

    # Logging
    parser.add_argument('--log_dir', type=str, default='results',
                       help='Log directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--print_freq', type=int, default=100,
                       help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                       help='Model save frequency')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--render_freq', type=int, default=500,
                       help='Render frequency')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')

    args = parser.parse_args()
    train(args)
