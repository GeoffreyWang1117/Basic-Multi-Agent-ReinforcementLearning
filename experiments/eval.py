"""
Evaluation script for trained multi-agent models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environments.cooperative_navigation import CooperativeNavigation
from environments.predator_prey import PredatorPrey
from algorithms.maddpg import MADDPG
from algorithms.comm_maddpg import CommMADDPG


def evaluate(args):
    """Evaluate a trained model"""

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

    # Create algorithm
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    if args.algorithm == 'maddpg':
        agent = MADDPG(
            num_agents=env_info['num_agents'],
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            device=device
        )
    elif args.algorithm == 'comm_maddpg':
        agent = CommMADDPG(
            num_agents=env_info['num_agents'],
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            comm_type=args.comm_type,
            device=device
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Load model
    print(f"Loading model from: {args.model_path}")
    agent.load(args.model_path)

    # Evaluation
    print(f"\nEvaluating for {args.num_episodes} episodes...")
    episode_rewards = []
    episode_steps = []

    for episode in range(args.num_episodes):
        states = env.reset()
        total_reward = 0
        step = 0
        trajectory = []

        while True:
            # Select actions (no noise)
            actions = agent.act(states, noise=0.0)

            # Store trajectory for rendering
            if args.render or args.save_video:
                trajectory.append({
                    'states': [s.copy() for s in states],
                    'actions': [a.copy() for a in actions]
                })

            # Execute actions
            next_states, rewards, dones, info = env.step(actions)

            states = next_states
            total_reward += np.sum(rewards)
            step += 1

            if all(dones):
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step)

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {step}")

        # Render last episode
        if args.render and episode == args.num_episodes - 1:
            env.render()

        # Save video
        if args.save_video and episode == args.num_episodes - 1:
            save_video(env, trajectory, args.video_path)

    # Statistics
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"Min/Max Reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"{'='*50}\n")


def save_video(env, trajectory, output_path):
    """Save trajectory as video"""
    print(f"Saving video to: {output_path}")

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        # Here you would render each frame
        # This is a placeholder - implement based on your env
        ax.set_title(f"Step {frame}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                  interval=100, blit=True)
    anim.save(output_path, writer='pillow', fps=10)
    plt.close()

    print("Video saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi-Agent RL')

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
                       help='Communication type')

    # Evaluation
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--save_video', action='store_true',
                       help='Save video of last episode')
    parser.add_argument('--video_path', type=str, default='eval_video.gif',
                       help='Video output path')

    # Misc
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')

    args = parser.parse_args()
    evaluate(args)
