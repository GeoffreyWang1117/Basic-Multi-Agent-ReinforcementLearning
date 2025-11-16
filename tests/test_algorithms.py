"""
Unit tests for multi-agent RL algorithms
"""

import unittest
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.maddpg import MADDPG
from algorithms.comm_maddpg import CommMADDPG


class TestMADDPG(unittest.TestCase):
    """Test MADDPG algorithm"""

    def setUp(self):
        """Set up test algorithm"""
        self.num_agents = 3
        self.state_dim = 10
        self.action_dim = 2
        self.global_state_dim = 30

        self.maddpg = MADDPG(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            global_state_dim=self.global_state_dim,
            batch_size=32,
            device='cpu'
        )

    def test_initialization(self):
        """Test algorithm initialization"""
        self.assertEqual(len(self.maddpg.agents), self.num_agents)
        self.assertEqual(self.maddpg.num_agents, self.num_agents)

    def test_act(self):
        """Test action selection"""
        states = [np.random.randn(self.state_dim) for _ in range(self.num_agents)]
        actions = self.maddpg.act(states, noise=0.1)

        self.assertEqual(len(actions), self.num_agents)
        for action in actions:
            self.assertEqual(len(action), self.action_dim)
            # Actions should be in [-1, 1]
            self.assertTrue(np.all(action >= -1.0))
            self.assertTrue(np.all(action <= 1.0))

    def test_step(self):
        """Test storing transitions"""
        states = [np.random.randn(self.state_dim) for _ in range(self.num_agents)]
        actions = [np.random.randn(self.action_dim) for _ in range(self.num_agents)]
        rewards = [np.random.randn() for _ in range(self.num_agents)]
        next_states = [np.random.randn(self.state_dim) for _ in range(self.num_agents)]
        dones = [False] * self.num_agents
        global_state = np.random.randn(self.global_state_dim)
        next_global_state = np.random.randn(self.global_state_dim)

        initial_size = len(self.maddpg.replay_buffer)

        self.maddpg.step(states, actions, rewards, next_states, dones,
                        global_state, next_global_state)

        self.assertEqual(len(self.maddpg.replay_buffer), initial_size + 1)

    def test_update(self):
        """Test algorithm update"""
        # Fill buffer with random data
        for _ in range(100):
            states = [np.random.randn(self.state_dim) for _ in range(self.num_agents)]
            actions = [np.random.randn(self.action_dim) for _ in range(self.num_agents)]
            rewards = [np.random.randn() for _ in range(self.num_agents)]
            next_states = [np.random.randn(self.state_dim) for _ in range(self.num_agents)]
            dones = [False] * self.num_agents
            global_state = np.random.randn(self.global_state_dim)
            next_global_state = np.random.randn(self.global_state_dim)

            self.maddpg.step(states, actions, rewards, next_states, dones,
                           global_state, next_global_state)

        # Update should work
        info = self.maddpg.update()

        self.assertIsInstance(info, dict)
        # Should have loss values for each agent
        for i in range(self.num_agents):
            self.assertIn(f'agent_{i}_critic_loss', info)
            self.assertIn(f'agent_{i}_actor_loss', info)

    def test_save_load(self):
        """Test model save and load"""
        import tempfile

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            save_path = f.name

        self.maddpg.save(save_path)

        # Create new instance and load
        new_maddpg = MADDPG(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            global_state_dim=self.global_state_dim,
            device='cpu'
        )

        new_maddpg.load(save_path)

        # Compare weights
        for i in range(self.num_agents):
            original_actor = self.maddpg.agents[i].actor.state_dict()
            loaded_actor = new_maddpg.agents[i].actor.state_dict()

            for key in original_actor:
                self.assertTrue(torch.allclose(original_actor[key], loaded_actor[key]))

        # Cleanup
        os.remove(save_path)


class TestCommMADDPG(unittest.TestCase):
    """Test CommMADDPG algorithm"""

    def setUp(self):
        """Set up test algorithm"""
        self.num_agents = 3
        self.state_dim = 10
        self.action_dim = 2

        self.comm_maddpg = CommMADDPG(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            comm_type='commnet',
            batch_size=32,
            device='cpu'
        )

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(len(self.comm_maddpg.agents), self.num_agents)
        self.assertIsNotNone(self.comm_maddpg.actor)

    def test_act_with_communication(self):
        """Test action selection with communication"""
        states = [np.random.randn(self.state_dim) for _ in range(self.num_agents)]
        actions = self.comm_maddpg.act(states, noise=0.1)

        self.assertEqual(len(actions), self.num_agents)
        for action in actions:
            self.assertEqual(len(action), self.action_dim)

    def test_shared_actor(self):
        """Test that all agents share the same actor"""
        # All agents should reference the same actor
        for agent in self.comm_maddpg.agents:
            self.assertIs(agent.actor, self.comm_maddpg.actor)


class TestAlgorithmInterface(unittest.TestCase):
    """Test that all algorithms follow the same interface"""

    def test_maddpg_interface(self):
        """Test MADDPG interface"""
        algo = MADDPG(num_agents=2, state_dim=4, action_dim=2, device='cpu')

        # Test required methods
        self.assertTrue(hasattr(algo, 'act'))
        self.assertTrue(hasattr(algo, 'step'))
        self.assertTrue(hasattr(algo, 'update'))
        self.assertTrue(hasattr(algo, 'save'))
        self.assertTrue(hasattr(algo, 'load'))

    def test_comm_maddpg_interface(self):
        """Test CommMADDPG interface"""
        algo = CommMADDPG(num_agents=2, state_dim=4, action_dim=2, device='cpu')

        # Test required methods
        self.assertTrue(hasattr(algo, 'act'))
        self.assertTrue(hasattr(algo, 'step'))
        self.assertTrue(hasattr(algo, 'update'))
        self.assertTrue(hasattr(algo, 'save'))
        self.assertTrue(hasattr(algo, 'load'))


if __name__ == '__main__':
    unittest.main()
