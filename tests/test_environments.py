"""
Unit tests for multi-agent environments
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.cooperative_navigation import CooperativeNavigation
from environments.predator_prey import PredatorPrey


class TestCooperativeNavigation(unittest.TestCase):
    """Test Cooperative Navigation environment"""

    def setUp(self):
        """Set up test environment"""
        self.env = CooperativeNavigation(num_agents=3, num_landmarks=3)

    def test_reset(self):
        """Test environment reset"""
        states = self.env.reset()

        self.assertEqual(len(states), 3, "Should return 3 agent states")
        for state in states:
            self.assertEqual(state.shape[0], self.env.state_dim,
                           f"State dim should be {self.env.state_dim}")

    def test_step(self):
        """Test environment step"""
        self.env.reset()

        # Random actions
        actions = [np.random.uniform(-1, 1, 2) for _ in range(3)]

        next_states, rewards, dones, info = self.env.step(actions)

        # Check outputs
        self.assertEqual(len(next_states), 3)
        self.assertEqual(len(rewards), 3)
        self.assertEqual(len(dones), 3)
        self.assertIsInstance(info, dict)

    def test_global_state(self):
        """Test global state retrieval"""
        self.env.reset()
        global_state = self.env.get_global_state()

        self.assertIsInstance(global_state, np.ndarray)
        self.assertGreater(len(global_state), 0)

    def test_collision_penalty(self):
        """Test collision detection"""
        self.env.reset()

        # Force agents to same position
        self.env.agent_pos[0] = np.array([0.0, 0.0])
        self.env.agent_pos[1] = np.array([0.0, 0.0])

        actions = [np.zeros(2) for _ in range(3)]
        _, rewards, _, info = self.env.step(actions)

        # Should have collision penalty
        self.assertGreater(info['collision_count'], 0)

    def test_state_consistency(self):
        """Test state consistency across steps"""
        states = self.env.reset()
        actions = [np.zeros(2) for _ in range(3)]

        for _ in range(10):
            states, _, _, _ = self.env.step(actions)
            self.assertEqual(len(states), 3)


class TestPredatorPrey(unittest.TestCase):
    """Test Predator-Prey environment"""

    def setUp(self):
        """Set up test environment"""
        self.env = PredatorPrey(num_predators=3, num_prey=1)

    def test_reset(self):
        """Test environment reset"""
        states = self.env.reset()

        expected_agents = 3 + 1  # predators + prey
        self.assertEqual(len(states), expected_agents)

    def test_step(self):
        """Test environment step"""
        self.env.reset()

        # Random actions for all agents
        actions = [np.random.uniform(-1, 1, 2) for _ in range(4)]

        next_states, rewards, dones, info = self.env.step(actions)

        self.assertEqual(len(next_states), 4)
        self.assertEqual(len(rewards), 4)
        self.assertEqual(len(dones), 4)

    def test_catch_mechanism(self):
        """Test prey catching mechanism"""
        self.env.reset()

        # Position predators around prey
        prey_pos = np.array([0.0, 0.0])
        self.env.prey_pos[0] = prey_pos

        # Two predators close to prey
        self.env.predator_pos[0] = prey_pos + np.array([0.1, 0.0])
        self.env.predator_pos[1] = prey_pos + np.array([-0.1, 0.0])

        self.env._check_catches()

        self.assertTrue(self.env.prey_caught[0], "Prey should be caught")

    def test_reward_structure(self):
        """Test reward structure"""
        self.env.reset()

        actions = [np.zeros(2) for _ in range(4)]
        _, rewards, _, _ = self.env.step(actions)

        # Predators should have rewards
        self.assertIsInstance(rewards[0], (int, float))

        # Prey should have survival bonus if not caught
        if not self.env.prey_caught[0]:
            self.assertGreater(rewards[3], 0)

    def test_global_state(self):
        """Test global state"""
        self.env.reset()
        global_state = self.env.get_global_state()

        self.assertIsInstance(global_state, np.ndarray)
        self.assertGreater(len(global_state), 0)


class TestEnvironmentInterface(unittest.TestCase):
    """Test that all environments follow the same interface"""

    def test_cooperative_nav_interface(self):
        """Test CooperativeNavigation interface"""
        env = CooperativeNavigation(num_agents=2, num_landmarks=2)

        # Test required methods
        self.assertTrue(hasattr(env, 'reset'))
        self.assertTrue(hasattr(env, 'step'))
        self.assertTrue(hasattr(env, 'get_global_state'))
        self.assertTrue(hasattr(env, 'get_env_info'))
        self.assertTrue(hasattr(env, 'render'))

    def test_predator_prey_interface(self):
        """Test PredatorPrey interface"""
        env = PredatorPrey(num_predators=2, num_prey=1)

        # Test required methods
        self.assertTrue(hasattr(env, 'reset'))
        self.assertTrue(hasattr(env, 'step'))
        self.assertTrue(hasattr(env, 'get_global_state'))
        self.assertTrue(hasattr(env, 'get_env_info'))
        self.assertTrue(hasattr(env, 'render'))


if __name__ == '__main__':
    unittest.main()
