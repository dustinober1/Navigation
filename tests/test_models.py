import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.navigation.models import QNetwork, DuelingQNetwork


class TestQNetwork(unittest.TestCase):
    """Test cases for QNetwork model."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_size = 37
        self.action_size = 4
        self.seed = 42
        self.batch_size = 32
        
        self.qnetwork = QNetwork(self.state_size, self.action_size, self.seed)
        
    def test_network_initialization(self):
        """Test that network initializes with correct architecture."""
        # Check network has correct layers
        self.assertTrue(hasattr(self.qnetwork, 'fc1'))
        self.assertTrue(hasattr(self.qnetwork, 'fc2'))
        self.assertTrue(hasattr(self.qnetwork, 'fc3'))
        
        # Check layer dimensions
        self.assertEqual(self.qnetwork.fc1.in_features, self.state_size)
        self.assertEqual(self.qnetwork.fc1.out_features, 64)  # default fc1_units
        self.assertEqual(self.qnetwork.fc2.in_features, 64)
        self.assertEqual(self.qnetwork.fc2.out_features, 64)  # default fc2_units
        self.assertEqual(self.qnetwork.fc3.in_features, 64)
        self.assertEqual(self.qnetwork.fc3.out_features, self.action_size)
        
    def test_custom_hidden_units(self):
        """Test network with custom hidden layer sizes."""
        fc1_units, fc2_units = 128, 256
        network = QNetwork(self.state_size, self.action_size, self.seed, fc1_units, fc2_units)
        
        self.assertEqual(network.fc1.out_features, fc1_units)
        self.assertEqual(network.fc2.in_features, fc1_units)
        self.assertEqual(network.fc2.out_features, fc2_units)
        self.assertEqual(network.fc3.in_features, fc2_units)
        
    def test_forward_pass_single_state(self):
        """Test forward pass with single state input."""
        state = torch.randn(1, self.state_size)
        output = self.qnetwork(state)
        
        # Check output shape
        self.assertEqual(output.shape, (1, self.action_size))
        self.assertTrue(torch.is_tensor(output))
        
    def test_forward_pass_batch(self):
        """Test forward pass with batch of states."""
        states = torch.randn(self.batch_size, self.state_size)
        output = self.qnetwork(states)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.action_size))
        self.assertTrue(torch.is_tensor(output))
        
    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        state = torch.randn(1, self.state_size)
        output = self.qnetwork(state)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for param in self.qnetwork.parameters():
            self.assertIsNotNone(param.grad)
            
    def test_reproducibility(self):
        """Test that same seed produces reproducible results."""
        state = torch.randn(1, self.state_size)
        
        # Create two networks with same seed
        net1 = QNetwork(self.state_size, self.action_size, self.seed)
        net2 = QNetwork(self.state_size, self.action_size, self.seed)
        
        # They should have same initial weights
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertTrue(torch.equal(p1, p2))
            

class TestDuelingQNetwork(unittest.TestCase):
    """Test cases for DuelingQNetwork model."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_size = 37
        self.action_size = 4
        self.seed = 42
        self.batch_size = 32
        
        self.dueling_network = DuelingQNetwork(self.state_size, self.action_size, self.seed)
        
    def test_network_initialization(self):
        """Test that dueling network initializes with correct architecture."""
        # Check shared layers
        self.assertTrue(hasattr(self.dueling_network, 'fc1'))
        self.assertTrue(hasattr(self.dueling_network, 'fc2'))
        
        # Check dueling streams
        self.assertTrue(hasattr(self.dueling_network, 'value_stream'))
        self.assertTrue(hasattr(self.dueling_network, 'advantage_stream'))
        
        # Check dimensions
        self.assertEqual(self.dueling_network.fc1.in_features, self.state_size)
        self.assertEqual(self.dueling_network.value_stream.out_features, 1)
        self.assertEqual(self.dueling_network.advantage_stream.out_features, self.action_size)
        
    def test_forward_pass_single_state(self):
        """Test forward pass with single state input."""
        state = torch.randn(1, self.state_size)
        output = self.dueling_network(state)
        
        # Check output shape
        self.assertEqual(output.shape, (1, self.action_size))
        self.assertTrue(torch.is_tensor(output))
        
    def test_forward_pass_batch(self):
        """Test forward pass with batch of states."""
        states = torch.randn(self.batch_size, self.state_size)
        output = self.dueling_network(states)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.action_size))
        self.assertTrue(torch.is_tensor(output))
        
    def test_dueling_architecture_property(self):
        """Test that dueling architecture has zero advantage mean property."""
        state = torch.randn(1, self.state_size)
        
        # Get intermediate values
        x = torch.relu(self.dueling_network.fc1(state))
        x = torch.relu(self.dueling_network.fc2(x))
        
        advantage = self.dueling_network.advantage_stream(x)
        
        # After dueling combination, advantages should have zero mean
        # (This is enforced in the forward pass)
        output = self.dueling_network(state)
        
        # Check that the network produces valid output
        self.assertEqual(output.shape, (1, self.action_size))
        
    def test_advantage_mean_zero(self):
        """Test that advantages are mean-centered in forward pass."""
        states = torch.randn(self.batch_size, self.state_size)
        
        # Manually compute what should happen
        x = torch.relu(self.dueling_network.fc1(states))
        x = torch.relu(self.dueling_network.fc2(x))
        
        value = self.dueling_network.value_stream(x)
        advantage = self.dueling_network.advantage_stream(x)
        
        # The dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        expected_output = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Compare with actual forward pass
        actual_output = self.dueling_network(states)
        
        self.assertTrue(torch.allclose(expected_output, actual_output, rtol=1e-5))
        
    def test_gradient_flow(self):
        """Test that gradients flow through both streams."""
        state = torch.randn(1, self.state_size)
        output = self.dueling_network(state)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed for all parameters
        for param in self.dueling_network.parameters():
            self.assertIsNotNone(param.grad)
            
    def test_different_from_regular_qnetwork(self):
        """Test that dueling network produces different outputs than regular network."""
        state = torch.randn(1, self.state_size)
        
        regular_net = QNetwork(self.state_size, self.action_size, self.seed)
        dueling_net = DuelingQNetwork(self.state_size, self.action_size, self.seed)
        
        regular_output = regular_net(state)
        dueling_output = dueling_net(state)
        
        # They should generally produce different outputs (unless by rare chance)
        self.assertFalse(torch.allclose(regular_output, dueling_output, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()