"""
Quick Start Example - Test the Multi-Agent RL framework

This script demonstrates basic usage of the framework:
1. Create an environment
2. Initialize MADDPG or CommMADDPG
3. Run a few training episodes
4. Visualize results

Usage:
    python quick_start.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from environments.cooperative_navigation import CooperativeNavigation
from algorithms.maddpg import MADDPG
from algorithms.comm_maddpg import CommMADDPG


def test_environment():
    """Test environment setup"""
    print("="*60)
    print("1. Testing Environment Setup")
    print("="*60)

    env = CooperativeNavigation(num_agents=3, num_landmarks=3)
    env_info = env.get_env_info()

    print(f"\nEnvironment: Cooperative Navigation")
    print(f"  - Number of agents: {env_info['num_agents']}")
    print(f"  - State dimension: {env_info['state_dim']}")
    print(f"  - Action dimension: {env_info['action_dim']}")
    print(f"  - Max steps: {env_info['max_steps']}")

    # Test reset
    states = env.reset()
    print(f"\nInitial states shape: {[s.shape for s in states]}")

    # Test step
    actions = [np.random.uniform(-1, 1, env_info['action_dim']) for _ in range(env_info['num_agents'])]
    next_states, rewards, dones, info = env.step(actions)

    print(f"Rewards: {rewards}")
    print(f"Info: {info}")

    print("\n✓ Environment test passed!")
    return env


def test_maddpg(env):
    """Test MADDPG algorithm"""
    print("\n" + "="*60)
    print("2. Testing MADDPG Algorithm")
    print("="*60)

    env_info = env.get_env_info()

    agent = MADDPG(
        num_agents=env_info['num_agents'],
        state_dim=env_info['state_dim'],
        action_dim=env_info['action_dim'],
        batch_size=32,
        device='cpu'
    )

    print("\nMADDPG initialized with:")
    print(f"  - {env_info['num_agents']} agents")
    print(f"  - State dim: {env_info['state_dim']}")
    print(f"  - Action dim: {env_info['action_dim']}")

    # Run a few episodes
    print("\nRunning 5 training episodes...")

    for episode in range(5):
        states = env.reset()
        episode_reward = 0

        for step in range(50):
            actions = agent.act(states, noise=0.3)
            next_states, rewards, dones, info = env.step(actions)

            global_state = env.get_global_state()
            next_global_state = env.get_global_state()

            agent.step(states, actions, rewards, next_states, dones,
                      global_state, next_global_state)

            if len(agent.replay_buffer) > agent.batch_size:
                update_info = agent.update()

            states = next_states
            episode_reward += np.sum(rewards)

            if all(dones):
                break

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")

    print("\n✓ MADDPG test passed!")


def test_comm_maddpg(env):
    """Test CommMADDPG algorithm"""
    print("\n" + "="*60)
    print("3. Testing CommMADDPG Algorithm (with Communication)")
    print("="*60)

    env_info = env.get_env_info()

    agent = CommMADDPG(
        num_agents=env_info['num_agents'],
        state_dim=env_info['state_dim'],
        action_dim=env_info['action_dim'],
        comm_type='commnet',
        num_comm_rounds=1,
        batch_size=32,
        device='cpu'
    )

    print("\nCommMADDPG initialized with:")
    print(f"  - {env_info['num_agents']} agents")
    print(f"  - Communication type: CommNet")
    print(f"  - Communication rounds: 1")

    # Run a few episodes
    print("\nRunning 5 training episodes...")

    for episode in range(5):
        states = env.reset()
        episode_reward = 0

        for step in range(50):
            actions = agent.act(states, noise=0.3)
            next_states, rewards, dones, info = env.step(actions)

            global_state = env.get_global_state()
            next_global_state = env.get_global_state()

            agent.step(states, actions, rewards, next_states, dones,
                      global_state, next_global_state)

            if len(agent.replay_buffer) > agent.batch_size:
                update_info = agent.update()

            states = next_states
            episode_reward += np.sum(rewards)

            if all(dones):
                break

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")

    print("\n✓ CommMADDPG test passed!")


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("MultiAgent-CTDE-Lab - Quick Start Test")
    print("="*60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Test environment
    env = test_environment()

    # Test MADDPG
    test_maddpg(env)

    # Test CommMADDPG
    test_comm_maddpg(env)

    print("\n" + "="*60)
    print("All Tests Passed! 🎉")
    print("="*60)
    print("\nNext Steps:")
    print("1. Train full model:")
    print("   python experiments/train.py --algorithm maddpg --num_episodes 10000")
    print("\n2. Train with communication:")
    print("   python experiments/train.py --algorithm comm_maddpg --comm_type commnet")
    print("\n3. Evaluate trained model:")
    print("   python experiments/eval.py --model_path results/your_model/best_model.pth")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
