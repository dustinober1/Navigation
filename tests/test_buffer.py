import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.navigation.buffers import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.action_size = 4
        self.buffer_size = 1000
        self.batch_size = 64
        self.seed = 42
        
        self.buffer = ReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, self.seed
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
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(len(self.buffer.memory), 0)
        
    def test_add_single_experience(self):
        """Test adding a single experience to the buffer."""
        initial_length = len(self.buffer)
        
        self.buffer.add(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done
        )
        
        self.assertEqual(len(self.buffer), initial_length + 1)
        
    def test_add_multiple_experiences(self):
        """Test adding multiple experiences to the buffer."""
        num_experiences = 10
        
        for i in range(num_experiences):
            self.buffer.add(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        self.assertEqual(len(self.buffer), num_experiences)
        
    def test_buffer_overflow(self):
        """Test that buffer respects maximum capacity."""
        # Fill buffer beyond capacity
        for i in range(self.buffer_size + 100):
            self.buffer.add(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        # Should not exceed buffer_size
        self.assertEqual(len(self.buffer), self.buffer_size)
        
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
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        # Check return types and shapes
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        
        # Check batch dimensions
        self.assertEqual(states.shape[0], self.batch_size)
        self.assertEqual(actions.shape[0], self.batch_size)
        self.assertEqual(rewards.shape[0], self.batch_size)
        self.assertEqual(next_states.shape[0], self.batch_size)
        self.assertEqual(dones.shape[0], self.batch_size)
        
        # Check feature dimensions
        self.assertEqual(states.shape[1], self.state_size)
        self.assertEqual(actions.shape[1], 1)
        self.assertEqual(rewards.shape[1], 1)
        self.assertEqual(next_states.shape[1], self.state_size)
        self.assertEqual(dones.shape[1], 1)
        
    def test_sample_data_types(self):
        """Test that sampled data has correct types and devices."""
        # Add experiences
        for i in range(self.batch_size + 10):
            self.buffer.add(
                np.random.randn(self.state_size), i % self.action_size,
                np.random.randn(), np.random.randn(self.state_size), 
                bool(i % 2)
            )
            
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        # Check tensor types
        self.assertEqual(states.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.int64)  # long tensor for actions
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(next_states.dtype, torch.float32)
        self.assertEqual(dones.dtype, torch.float32)
        
        # Check device (should be on the same device as defined in buffer.py)
        expected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.assertEqual(states.device, expected_device)
        self.assertEqual(actions.device, expected_device)
        self.assertEqual(rewards.device, expected_device)
        self.assertEqual(next_states.device, expected_device)
        self.assertEqual(dones.device, expected_device)
        
    def test_sample_randomness(self):
        """Test that sampling is random."""
        # Add more experiences than batch size
        experiences_data = []
        for i in range(200):
            state = np.random.randn(self.state_size)
            action = i % self.action_size
            reward = float(i)  # Use index as reward for tracking
            next_state = np.random.randn(self.state_size)
            done = bool(i % 2)
            
            experiences_data.append((state, action, reward, next_state, done))
            self.buffer.add(state, action, reward, next_state, done)
            
        # Sample twice
        _, _, rewards1, _, _ = self.buffer.sample()
        _, _, rewards2, _, _ = self.buffer.sample()
        
        # The rewards should be different (indicating different samples)
        # Note: There's a tiny chance they could be the same, but very unlikely
        self.assertFalse(torch.equal(rewards1, rewards2))
        
    def test_experience_namedtuple(self):
        """Test that experiences are stored as namedtuples correctly."""
        self.buffer.add(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done
        )
        
        # Access the stored experience
        stored_experience = self.buffer.memory[0]
        
        # Check namedtuple fields
        self.assertTrue(hasattr(stored_experience, 'state'))
        self.assertTrue(hasattr(stored_experience, 'action'))
        self.assertTrue(hasattr(stored_experience, 'reward'))
        self.assertTrue(hasattr(stored_experience, 'next_state'))
        self.assertTrue(hasattr(stored_experience, 'done'))
        
        # Check values
        np.testing.assert_array_equal(stored_experience.state, self.sample_state)
        self.assertEqual(stored_experience.action, self.sample_action)
        self.assertEqual(stored_experience.reward, self.sample_reward)
        np.testing.assert_array_equal(stored_experience.next_state, self.sample_next_state)
        self.assertEqual(stored_experience.done, self.sample_done)
        
    def test_seed_reproducibility(self):
        """Test that same seed produces reproducible sampling."""
        # Create two identical buffers with same seed
        buffer1 = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        buffer2 = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        
        # Add identical experiences to both
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
            buffer1.add(*exp)
            buffer2.add(*exp)
        
        # Reset random seed for both samplers
        buffer1.seed = self.seed
        buffer2.seed = self.seed
        
        # Sample from both - they should be identical
        sample1 = buffer1.sample()
        sample2 = buffer2.sample()
        
        # Compare samples (they should be the same due to same seed)
        for tensor1, tensor2 in zip(sample1, sample2):
            self.assertTrue(torch.equal(tensor1, tensor2))


if __name__ == '__main__':
    unittest.main()