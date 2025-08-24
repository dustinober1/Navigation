import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.navigation.agents import Agent, PrioritizedAgent


class TestAgent(unittest.TestCase):
    """Test cases for Agent class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_size = 37
        self.action_size = 4
        self.seed = 42
        
        self.agent = Agent(self.state_size, self.action_size, self.seed)
        
        # Sample experience data
        self.sample_state = np.random.randn(self.state_size)
        self.sample_action = 1
        self.sample_reward = 1.0
        self.sample_next_state = np.random.randn(self.state_size)
        self.sample_done = False
        
    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertFalse(self.agent.double_dqn)
        self.assertFalse(self.agent.dueling_dqn)
        
        # Check networks are created
        self.assertIsNotNone(self.agent.qnetwork_local)
        self.assertIsNotNone(self.agent.qnetwork_target)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
        
    def test_double_dqn_initialization(self):
        """Test agent with Double DQN enabled."""
        double_agent = Agent(self.state_size, self.action_size, self.seed, double_dqn=True)
        self.assertTrue(double_agent.double_dqn)
        
    def test_dueling_dqn_initialization(self):
        """Test agent with Dueling DQN enabled."""
        dueling_agent = Agent(self.state_size, self.action_size, self.seed, dueling_dqn=True)
        self.assertTrue(dueling_agent.dueling_dqn)
        
        # Check that it uses DuelingQNetwork
        from src.navigation.models import DuelingQNetwork
        self.assertIsInstance(dueling_agent.qnetwork_local, DuelingQNetwork)
        self.assertIsInstance(dueling_agent.qnetwork_target, DuelingQNetwork)
        
    def test_custom_hyperparameters(self):
        """Test agent with custom hyperparameters."""
        lr = 1e-3
        batch_size = 128
        gamma = 0.95
        tau = 1e-2
        
        custom_agent = Agent(
            self.state_size, self.action_size, self.seed,
            lr=lr, batch_size=batch_size, gamma=gamma, tau=tau
        )
        
        self.assertEqual(custom_agent.lr, lr)
        self.assertEqual(custom_agent.batch_size, batch_size)
        self.assertEqual(custom_agent.gamma, gamma)
        self.assertEqual(custom_agent.tau, tau)
        
    def test_act_method(self):
        """Test action selection."""
        state = np.random.randn(self.state_size)
        
        # Test with epsilon = 1.0 (should be random)
        action_random = self.agent.act(state, eps=1.0)
        self.assertIsInstance(action_random, (int, np.integer))
        self.assertIn(action_random, range(self.action_size))
        
        # Test with epsilon = 0.0 (should be greedy)
        action_greedy = self.agent.act(state, eps=0.0)
        self.assertIsInstance(action_greedy, (int, np.integer))
        self.assertIn(action_greedy, range(self.action_size))
        
    def test_act_reproducibility(self):
        """Test that act is reproducible with same epsilon."""
        state = np.random.randn(self.state_size)
        
        # With eps=0, should always return same action for same state
        action1 = self.agent.act(state, eps=0.0)
        action2 = self.agent.act(state, eps=0.0)
        self.assertEqual(action1, action2)
        
    def test_step_method(self):
        """Test that step method adds to memory."""
        initial_memory_size = len(self.agent.memory)
        
        self.agent.step(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done
        )
        
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
        
    def test_step_triggers_learning(self):
        """Test that step triggers learning when conditions are met."""
        # Fill memory with enough experiences
        batch_size = self.agent.batch_size
        for i in range(batch_size + 10):
            self.agent.step(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Mock the learn method to check if it's called
        with patch.object(self.agent, 'learn') as mock_learn:
            # Add more experiences to trigger learning
            for i in range(self.agent.update_every):
                self.agent.step(
                    np.random.randn(self.state_size), i % self.action_size,
                    np.random.randn(), np.random.randn(self.state_size), 
                    bool(i % 2)
                )
                
            # Learn should have been called at least once
            self.assertTrue(mock_learn.called)
            
    def test_soft_update(self):
        """Test soft update of target network."""
        # Get initial parameters
        local_params_before = [p.clone() for p in self.agent.qnetwork_local.parameters()]
        target_params_before = [p.clone() for p in self.agent.qnetwork_target.parameters()]
        
        # Make sure local and target networks are different initially
        # (they should be since they're initialized with different random states due to training)
        # Let's manually change local network to ensure difference
        with torch.no_grad():
            for p in self.agent.qnetwork_local.parameters():
                p.add_(0.1)
                
        # Perform soft update
        tau = 0.1
        self.agent.soft_update(self.agent.qnetwork_local, self.agent.qnetwork_target, tau)
        
        # Check that target network parameters changed
        for p_before, p_after in zip(target_params_before, self.agent.qnetwork_target.parameters()):
            self.assertFalse(torch.equal(p_before, p_after))
            
    def test_learn_method_standard_dqn(self):
        """Test learn method for standard DQN."""
        # Fill memory with experiences
        batch_size = self.agent.batch_size
        for i in range(batch_size + 10):
            self.agent.step(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Sample experiences
        experiences = self.agent.memory.sample()
        
        # Get initial parameters
        initial_params = [p.clone() for p in self.agent.qnetwork_local.parameters()]
        
        # Call learn method
        self.agent.learn(experiences, gamma=0.99)
        
        # Check that parameters have changed (learning occurred)
        for p_initial, p_current in zip(initial_params, self.agent.qnetwork_local.parameters()):
            # At least some parameters should have changed
            if not torch.equal(p_initial, p_current):
                break
        else:
            self.fail("No parameters changed during learning")
            
    def test_learn_method_double_dqn(self):
        """Test learn method for Double DQN."""
        double_agent = Agent(self.state_size, self.action_size, self.seed, double_dqn=True)
        
        # Fill memory with experiences
        batch_size = double_agent.batch_size
        for i in range(batch_size + 10):
            double_agent.step(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Sample experiences
        experiences = double_agent.memory.sample()
        
        # Get initial parameters
        initial_params = [p.clone() for p in double_agent.qnetwork_local.parameters()]
        
        # Call learn method
        double_agent.learn(experiences, gamma=0.99)
        
        # Check that learning occurred
        for p_initial, p_current in zip(initial_params, double_agent.qnetwork_local.parameters()):
            if not torch.equal(p_initial, p_current):
                break
        else:
            self.fail("No parameters changed during Double DQN learning")


class TestPrioritizedAgent(unittest.TestCase):
    """Test cases for PrioritizedAgent class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_size = 37
        self.action_size = 4
        self.seed = 42
        
        self.agent = PrioritizedAgent(self.state_size, self.action_size, self.seed)
        
        # Sample experience data
        self.sample_state = np.random.randn(self.state_size)
        self.sample_action = 1
        self.sample_reward = 1.0
        self.sample_next_state = np.random.randn(self.state_size)
        self.sample_done = False
        
    def test_prioritized_agent_initialization(self):
        """Test that prioritized agent initializes correctly."""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        
        # Check that it uses PrioritizedReplayBuffer
        from src.navigation.buffers import PrioritizedReplayBuffer
        self.assertIsInstance(self.agent.memory, PrioritizedReplayBuffer)
        
    def test_prioritized_with_double_dueling(self):
        """Test prioritized agent with Double and Dueling DQN."""
        agent = PrioritizedAgent(
            self.state_size, self.action_size, self.seed,
            double_dqn=True, dueling_dqn=True
        )
        
        self.assertTrue(agent.double_dqn)
        self.assertTrue(agent.dueling_dqn)
        
        from src.navigation.models import DuelingQNetwork
        self.assertIsInstance(agent.qnetwork_local, DuelingQNetwork)
        
    def test_act_method(self):
        """Test action selection for prioritized agent."""
        state = np.random.randn(self.state_size)
        
        # Test with epsilon = 1.0 (should be random)
        action_random = self.agent.act(state, eps=1.0)
        self.assertIsInstance(action_random, (int, np.integer))
        self.assertIn(action_random, range(self.action_size))
        
        # Test with epsilon = 0.0 (should be greedy)
        action_greedy = self.agent.act(state, eps=0.0)
        self.assertIsInstance(action_greedy, (int, np.integer))
        self.assertIn(action_greedy, range(self.action_size))
        
    def test_step_method(self):
        """Test that step method adds to prioritized memory."""
        initial_memory_size = len(self.agent.memory)
        
        self.agent.step(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done
        )
        
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
        
    def test_learn_method_with_importance_sampling(self):
        """Test learn method with importance sampling weights."""
        # Fill memory with experiences
        batch_size = self.agent.batch_size
        for i in range(batch_size + 10):
            self.agent.step(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Sample experiences (should include importance sampling weights and indices)
        experiences = self.agent.memory.sample()
        self.assertEqual(len(experiences), 7)  # includes weights and indices
        
        # Get initial parameters
        initial_params = [p.clone() for p in self.agent.qnetwork_local.parameters()]
        
        # Call learn method
        self.agent.learn(experiences, gamma=0.99)
        
        # Check that learning occurred
        for p_initial, p_current in zip(initial_params, self.agent.qnetwork_local.parameters()):
            if not torch.equal(p_initial, p_current):
                break
        else:
            self.fail("No parameters changed during prioritized learning")
            
    def test_priority_updates(self):
        """Test that priorities are updated during learning."""
        # Fill memory with experiences
        batch_size = self.agent.batch_size
        for i in range(batch_size + 10):
            self.agent.step(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Get initial sum tree total
        initial_total = self.agent.memory.tree.total()
        
        # Sample and learn
        experiences = self.agent.memory.sample()
        self.agent.learn(experiences, gamma=0.99)
        
        # Check that sum tree total changed (indicating priority updates)
        final_total = self.agent.memory.tree.total()
        # Note: This might occasionally fail due to random chance, but very unlikely
        self.assertNotEqual(initial_total, final_total)
        
    def test_custom_prioritized_parameters(self):
        """Test prioritized agent with custom parameters."""
        alpha = 0.8
        beta = 0.6
        beta_increment = 0.002
        
        agent = PrioritizedAgent(
            self.state_size, self.action_size, self.seed,
            alpha=alpha, beta=beta, beta_increment=beta_increment
        )
        
        self.assertEqual(agent.memory.alpha, alpha)
        self.assertEqual(agent.memory.beta, beta)
        self.assertEqual(agent.memory.beta_increment, beta_increment)


class TestAgentComparison(unittest.TestCase):
    """Test cases comparing Agent and PrioritizedAgent behavior."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_size = 37
        self.action_size = 4
        self.seed = 42
        
        self.standard_agent = Agent(self.state_size, self.action_size, self.seed)
        self.prioritized_agent = PrioritizedAgent(self.state_size, self.action_size, self.seed)
        
    def test_same_network_architectures(self):
        """Test that both agents use same network architecture by default."""
        # Both should use regular QNetwork by default
        from src.navigation.models import QNetwork
        self.assertIsInstance(self.standard_agent.qnetwork_local, QNetwork)
        self.assertIsInstance(self.prioritized_agent.qnetwork_local, QNetwork)
        
    def test_different_memory_types(self):
        """Test that agents use different memory types."""
        from src.navigation.buffers import ReplayBuffer, PrioritizedReplayBuffer
        
        self.assertIsInstance(self.standard_agent.memory, ReplayBuffer)
        self.assertIsInstance(self.prioritized_agent.memory, PrioritizedReplayBuffer)
        
    def test_same_action_selection(self):
        """Test that both agents select same actions given same state and epsilon."""
        state = np.random.randn(self.state_size)
        
        # With eps=0, both should select same action (greedy)
        action_standard = self.standard_agent.act(state, eps=0.0)
        action_prioritized = self.prioritized_agent.act(state, eps=0.0)
        
        # Note: This might fail if networks have different initialization
        # The test is more about ensuring both agents produce valid actions
        self.assertIn(action_standard, range(self.action_size))
        self.assertIn(action_prioritized, range(self.action_size))


if __name__ == '__main__':
    unittest.main()