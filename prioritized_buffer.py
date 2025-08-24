import random
import numpy as np
from collections import namedtuple
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for efficient sampling from prioritized experience replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pointer = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.pointer + self.capacity - 1

        self.data[self.pointer] = data
        self.update(idx, p)

        self.pointer += 1
        if self.pointer >= self.capacity:
            self.pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Samples experiences based on their TD error magnitude.
    """
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta=0.4, beta_increment=0.001):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): determines how much prioritization is used (0 = uniform, 1 = full prioritization)
            beta (float): importance sampling weight (0 = no correction, 1 = full correction)
            beta_increment (float): increment for beta per sampling step
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # small constant to prevent zero priorities
        
        self.tree = SumTree(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        self.max_priority = 1.0  # initial priority for new experiences

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        priority = self.max_priority  # assign max priority to new experiences
        self.tree.add(priority, e)

    def sample(self):
        """Sample a batch of experiences based on priorities."""
        batch = []
        idxs = []
        priorities = []
        
        segment = self.tree.total() / self.batch_size
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # normalize weights

        states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(is_weights).float().to(device)

        return (states, actions, rewards, next_states, dones, weights, idxs)

    def update_priorities(self, idxs, errors):
        """Update priorities of sampled experiences."""
        for idx, error in zip(idxs, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.n_entries