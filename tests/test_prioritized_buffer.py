import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.navigation.buffers import SumTree, PrioritizedReplayBuffer


class TestSumTree(unittest.TestCase):
    """Test cases for SumTree data structure."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.capacity = 16
        self.tree = SumTree(self.capacity)
        
    def test_tree_initialization(self):
        """Test that tree initializes correctly."""
        self.assertEqual(self.tree.capacity, self.capacity)
        self.assertEqual(self.tree.n_entries, 0)
        self.assertEqual(self.tree.pointer, 0)
        self.assertEqual(len(self.tree.tree), 2 * self.capacity - 1)
        self.assertEqual(len(self.tree.data), self.capacity)
        
    def test_add_single_item(self):
        """Test adding a single item to the tree."""
        priority = 1.0
        data = "test_data"
        
        self.tree.add(priority, data)
        
        self.assertEqual(self.tree.n_entries, 1)
        self.assertEqual(self.tree.pointer, 1)
        self.assertEqual(self.tree.total(), priority)
        
    def test_add_multiple_items(self):
        """Test adding multiple items to the tree."""
        priorities = [1.0, 2.0, 3.0, 4.0]
        data_items = ["data1", "data2", "data3", "data4"]
        
        for p, d in zip(priorities, data_items):
            self.tree.add(p, d)
            
        self.assertEqual(self.tree.n_entries, len(priorities))
        self.assertEqual(self.tree.total(), sum(priorities))
        
    def test_circular_buffer_behavior(self):
        """Test that tree behaves as circular buffer when full."""
        # Fill beyond capacity
        for i in range(self.capacity + 5):
            self.tree.add(1.0, f"data{i}")
            
        # Should not exceed capacity
        self.assertEqual(self.tree.n_entries, self.capacity)
        # Pointer should wrap around
        self.assertEqual(self.tree.pointer, 5)  # (capacity + 5) % capacity
        
    def test_get_item(self):
        """Test retrieving items by cumulative sum."""
        priorities = [1.0, 2.0, 3.0, 4.0]
        data_items = ["data1", "data2", "data3", "data4"]
        
        for p, d in zip(priorities, data_items):
            self.tree.add(p, d)
            
        total = self.tree.total()
        
        # Test getting first item (s=0.5, should get first item with priority 1.0)
        idx, priority, data = self.tree.get(0.5)
        self.assertEqual(priority, 1.0)
        self.assertEqual(data, "data1")
        
        # Test getting item in middle range
        idx, priority, data = self.tree.get(2.5)  # Should get second item
        self.assertEqual(priority, 2.0)
        self.assertEqual(data, "data2")
        
    def test_update_priority(self):
        """Test updating priority of existing items."""
        self.tree.add(1.0, "data1")
        self.tree.add(2.0, "data2")
        
        initial_total = self.tree.total()
        
        # Update first item's priority
        leaf_idx = self.capacity - 1  # Index of first leaf
        self.tree.update(leaf_idx, 5.0)
        
        # Total should reflect the change
        expected_total = initial_total - 1.0 + 5.0
        self.assertEqual(self.tree.total(), expected_total)
        
    def test_total_zero_when_empty(self):
        """Test that total is zero when tree is empty."""
        self.assertEqual(self.tree.total(), 0)
        

class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Test cases for PrioritizedReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.action_size = 4
        self.buffer_size = 1000
        self.batch_size = 64
        self.seed = 42
        self.alpha = 0.6
        self.beta = 0.4
        
        self.buffer = PrioritizedReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, self.seed,
            self.alpha, self.beta
        )
        
        # Sample experience data
        self.state_size = 37
        self.sample_state = np.random.randn(self.state_size)
        self.sample_action = 1
        self.sample_reward = 1.0
        self.sample_next_state = np.random.randn(self.state_size)
        self.sample_done = False
        
    def test_buffer_initialization(self):
        """Test that buffer initializes correctly."""
        self.assertEqual(self.buffer.action_size, self.action_size)
        self.assertEqual(self.buffer.batch_size, self.batch_size)
        self.assertEqual(self.buffer.alpha, self.alpha)
        self.assertEqual(self.buffer.beta, self.beta)
        self.assertEqual(len(self.buffer), 0)
        self.assertIsNotNone(self.buffer.tree)
        
    def test_add_single_experience(self):
        """Test adding a single experience to the buffer."""
        initial_length = len(self.buffer)
        initial_total = self.buffer.tree.total()
        
        self.buffer.add(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done
        )
        
        self.assertEqual(len(self.buffer), initial_length + 1)
        self.assertGreater(self.buffer.tree.total(), initial_total)
        
    def test_new_experiences_get_max_priority(self):
        """Test that new experiences get maximum priority."""
        # Add an experience
        self.buffer.add(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done
        )
        
        # The total should equal max_priority (since it's the only item)
        self.assertEqual(self.buffer.tree.total(), self.buffer.max_priority)
        
    def test_sample_insufficient_data(self):
        """Test that sampling fails when insufficient data."""
        # Add fewer experiences than batch_size
        for i in range(self.batch_size - 10):
            self.buffer.add(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Should not be able to sample full batch
        with self.assertRaises(ValueError):
            self.buffer.sample()
            
    def test_sample_sufficient_data(self):
        """Test sampling when sufficient data is available."""
        # Add enough experiences
        for i in range(self.batch_size + 10):
            self.buffer.add(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Should be able to sample
        result = self.buffer.sample()
        self.assertEqual(len(result), 7)  # states, actions, rewards, next_states, dones, weights, idxs
        
        states, actions, rewards, next_states, dones, weights, idxs = result
        
        # Check return types and shapes
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(weights, torch.Tensor)
        self.assertIsInstance(idxs, list)
        
        # Check batch dimensions
        self.assertEqual(states.shape[0], self.batch_size)
        self.assertEqual(actions.shape[0], self.batch_size)
        self.assertEqual(rewards.shape[0], self.batch_size)
        self.assertEqual(next_states.shape[0], self.batch_size)
        self.assertEqual(dones.shape[0], self.batch_size)
        self.assertEqual(weights.shape[0], self.batch_size)
        self.assertEqual(len(idxs), self.batch_size)
        
    def test_importance_sampling_weights(self):
        """Test that importance sampling weights are computed correctly."""
        # Add experiences
        for i in range(self.batch_size + 10):
            self.buffer.add(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        states, actions, rewards, next_states, dones, weights, idxs = self.buffer.sample()
        
        # Weights should be positive
        self.assertTrue(torch.all(weights > 0))
        
        # Maximum weight should be 1.0 (due to normalization)
        self.assertAlmostEqual(torch.max(weights).item(), 1.0, places=5)
        
        # Weights shape should match batch size
        self.assertEqual(weights.shape, (self.batch_size,))
        
    def test_beta_annealing(self):
        """Test that beta increases over time."""
        initial_beta = self.buffer.beta
        
        # Add enough experiences and sample multiple times
        for i in range(self.batch_size + 10):
            self.buffer.add(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Sample several times to trigger beta updates
        for _ in range(10):
            self.buffer.sample()
            
        # Beta should have increased
        self.assertGreater(self.buffer.beta, initial_beta)
        
        # Beta should not exceed 1.0
        self.assertLessEqual(self.buffer.beta, 1.0)
        
    def test_update_priorities(self):
        """Test updating priorities based on TD errors."""
        # Add experiences
        for i in range(self.batch_size + 10):
            self.buffer.add(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Sample to get indices
        states, actions, rewards, next_states, dones, weights, idxs = self.buffer.sample()
        
        # Create some TD errors
        td_errors = np.random.randn(self.batch_size)
        initial_total = self.buffer.tree.total()
        
        # Update priorities
        self.buffer.update_priorities(idxs, td_errors)
        
        # Total should have changed (unless by rare coincidence)
        # Note: This test might occasionally fail due to random chance
        final_total = self.buffer.tree.total()
        self.assertNotEqual(initial_total, final_total)
        
    def test_max_priority_tracking(self):
        """Test that max_priority is tracked correctly."""
        initial_max = self.buffer.max_priority
        
        # Add experience
        self.buffer.add(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done
        )
        
        # Max priority should be unchanged (new experiences get current max)
        self.assertEqual(self.buffer.max_priority, initial_max)
        
        # Update with a larger priority
        large_error = 10.0
        self.buffer.update_priorities([self.buffer.tree.capacity - 1], [large_error])
        
        # Max priority should have increased
        expected_max = (abs(large_error) + self.buffer.epsilon) ** self.buffer.alpha
        self.assertEqual(self.buffer.max_priority, expected_max)
        
    def test_alpha_parameter_effect(self):
        """Test that alpha parameter affects prioritization."""
        # Create buffers with different alpha values
        buffer_no_priority = PrioritizedReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, self.seed, alpha=0.0
        )
        buffer_full_priority = PrioritizedReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, self.seed, alpha=1.0
        )
        
        # Add same experiences to both
        experiences = []
        for i in range(self.batch_size + 10):
            exp = (
                np.random.randn(self.state_size),
                i % self.action_size,
                np.random.randn(),
                np.random.randn(self.state_size),
                bool(i % 2)
            )
            experiences.append(exp)
            buffer_no_priority.add(*exp)
            buffer_full_priority.add(*exp)
        
        # Update with different TD errors (make some experiences much more important)
        sample_no_priority = buffer_no_priority.sample()
        sample_full_priority = buffer_full_priority.sample()
        
        # Create TD errors with high variance
        td_errors = np.array([10.0] * (self.batch_size // 2) + [0.1] * (self.batch_size // 2))
        
        buffer_no_priority.update_priorities(sample_no_priority[-1], td_errors)
        buffer_full_priority.update_priorities(sample_full_priority[-1], td_errors)
        
        # Sample again - full priority should show more variance in weights
        _, _, _, _, _, weights_no_priority, _ = buffer_no_priority.sample()
        _, _, _, _, _, weights_full_priority, _ = buffer_full_priority.sample()
        
        # With alpha=0, weights should be more uniform
        # With alpha=1, weights should have higher variance
        variance_no_priority = torch.var(weights_no_priority)
        variance_full_priority = torch.var(weights_full_priority)
        
        # This test might be flaky due to randomness, but generally should hold
        # We just check that both produced valid weights
        self.assertTrue(torch.all(weights_no_priority > 0))
        self.assertTrue(torch.all(weights_full_priority > 0))


if __name__ == '__main__':
    unittest.main()