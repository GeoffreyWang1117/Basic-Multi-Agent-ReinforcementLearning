"""
Training script that uses YAML configuration files
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
from utils.config import ConfigManager


def create_environment(config: ConfigManager):
    """Create environment from config"""
    env_name = config.get('env.name')

    if env_name == 'cooperative_navigation':
        env = CooperativeNavigation(
            num_agents=config.get('env.num_agents', 3),
            num_landmarks=config.get('env.num_landmarks', 3),
            world_size=config.get('env.world_size', 2.0),
            max_steps=config.get('env.max_steps', 100)
        )
    elif env_name == 'predator_prey':
        env = PredatorPrey(
            num_predators=config.get('env.num_predators', 3),
            num_prey=config.get('env.num_prey', 1),
            world_size=config.get('env.world_size', 2.0),
            max_steps=config.get('env.max_steps', 100)
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env


def create_algorithm(config: ConfigManager, env_info: dict, device: str):
    """Create algorithm from config"""
    algo_name = config.get('algorithm.name')

    if algo_name == 'maddpg':
        agent = MADDPG(
            num_agents=env_info['num_agents'],
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            hidden_dim=config.get('algorithm.hidden_dim', 256),
            lr_actor=config.get('algorithm.lr_actor', 1e-4),
            lr_critic=config.get('algorithm.lr_critic', 1e-3),
            gamma=config.get('algorithm.gamma', 0.99),
            tau=config.get('algorithm.tau', 0.01),
            buffer_size=config.get('algorithm.buffer_size', 100000),
            batch_size=config.get('algorithm.batch_size', 256),
            device=device
        )
    elif algo_name == 'comm_maddpg':
        agent = CommMADDPG(
            num_agents=env_info['num_agents'],
            state_dim=env_info['state_dim'],
            action_dim=env_info['action_dim'],
            hidden_dim=config.get('algorithm.hidden_dim', 256),
            comm_type=config.get('algorithm.comm_type', 'commnet'),
            num_comm_rounds=config.get('algorithm.comm_rounds', 1),
            lr_actor=config.get('algorithm.lr_actor', 1e-4),
            lr_critic=config.get('algorithm.lr_critic', 1e-3),
            gamma=config.get('algorithm.gamma', 0.99),
            tau=config.get('algorithm.tau', 0.01),
            buffer_size=config.get('algorithm.buffer_size', 100000),
            batch_size=config.get('algorithm.batch_size', 256),
            device=device
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return agent


def train(config: ConfigManager):
    """Main training loop using config"""

    # Set random seeds
    seed = config.get('system.seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    device_setting = config.get('system.device', 'auto')
    if device_setting == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_setting

    print(f"Using device: {device}")

    # Create environment
    env = create_environment(config)
    env_info = env.get_env_info()

    print(f"\nEnvironment: {config.get('env.name')}")
    print(f"  - Num agents: {env_info['num_agents']}")
    print(f"  - State dim: {env_info['state_dim']}")
    print(f"  - Action dim: {env_info['action_dim']}")

    # Create algorithm
    agent = create_algorithm(config, env_info, device)

    print(f"\nAlgorithm: {config.get('algorithm.name')}")

    # Create logger
    log_dir = config.get('output.log_dir', 'results')
    exp_name = config.get('output.exp_name', None)
    logger = Logger(log_dir, exp_name)

    # Save config
    logger.save_config(config.config)

    # Training parameters
    num_episodes = config.get('training.num_episodes', 10000)
    noise_start = config.get('training.noise_start', 0.5)
    noise_decay = config.get('training.noise_decay', 0.9999)
    noise_min = config.get('training.noise_min', 0.05)
    update_freq = config.get('training.update_freq', 1)
    print_freq = config.get('training.print_freq', 100)
    save_freq = config.get('training.save_freq', 1000)
    render = config.get('training.render', False)
    render_freq = config.get('training.render_freq', 500)

    # Training loop
    print("\nStarting training...")
    global_step = 0
    best_reward = -float('inf')

    for episode in tqdm(range(num_episodes), desc="Training"):
        states = env.reset()
        episode_reward = np.zeros(env_info['num_agents'])
        episode_step = 0

        while True:
            # Select actions with decaying noise
            noise = max(noise_min, noise_start * (noise_decay ** episode))
            actions = agent.act(states, noise=noise)

            # Execute actions
            global_state = env.get_global_state()
            next_states, rewards, dones, info = env.step(actions)
            next_global_state = env.get_global_state()

            # Store transition
            agent.step(states, actions, rewards, next_states, dones,
                      global_state, next_global_state)

            # Update agents
            if global_step % update_freq == 0:
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
        logger.log_scalar('noise', noise, episode)

        # Print progress
        if (episode + 1) % print_freq == 0:
            recent_rewards = [ep['total_reward'] for ep in logger.episode_data[-print_freq:]]
            avg_reward = np.mean(recent_rewards)
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last {print_freq}): {avg_reward:.2f}")
            print(f"  Noise: {noise:.4f}")
            if info:
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.2f}")

        # Save best model
        if config.get('output.save_best', True) and total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(logger.log_dir, 'best_model.pth'))

        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(logger.log_dir, f'checkpoint_{episode + 1}.pth'))

        # Render
        if render and (episode + 1) % render_freq == 0:
            env.render(save_path=os.path.join(logger.log_dir, f'render_{episode + 1}.png'))

    # Final save
    agent.save(os.path.join(logger.log_dir, 'final_model.pth'))
    logger.plot_metrics()
    logger.print_summary()

    print(f"\nTraining completed! Results saved to: {logger.log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train from YAML config')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--override', nargs='*', default=[],
                       help='Override config values (e.g., training.num_episodes=5000)')

    args = parser.parse_args()

    # Load config
    config = ConfigManager(config_path=args.config)

    # Apply overrides
    for override in args.override:
        key, value = override.split('=')
        # Try to parse as number
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

        # Set nested key
        keys = key.split('.')
        temp = config.config
        for k in keys[:-1]:
            temp = temp.setdefault(k, {})
        temp[keys[-1]] = value

    print("Configuration:")
    print(config)

    train(config)
