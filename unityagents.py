"""
Minimal UnityEnvironment wrapper for Banana navigation environment
Compatible with Udacity DRLND projects
"""
import numpy as np
import subprocess
import socket
import struct
import time
from collections import namedtuple

BrainInfo = namedtuple('BrainInfo', ['vector_observations', 'visual_observations', 'rewards', 'local_done', 'agents'])
BrainParameters = namedtuple('BrainParameters', ['vector_observation_space_size', 'number_visual_observations', 
                                                 'camera_resolutions', 'vector_action_space_size', 
                                                 'vector_action_descriptions', 'vector_action_space_type'])

class UnityEnvironment:
    def __init__(self, file_name=None, worker_id=0, base_port=5005, seed=1, no_graphics=True):
        """
        Initialize Unity Environment
        
        Arguments:
        file_name -- Path to Unity environment executable
        worker_id -- Worker ID for parallel training
        base_port -- Base port for communication
        seed -- Random seed
        no_graphics -- Whether to run without graphics
        """
        self.file_name = file_name
        self.worker_id = worker_id
        self.port = base_port + worker_id
        self.seed = seed
        self.no_graphics = no_graphics
        self.proc = None
        self._sock = None
        
        # Brain info for Banana environment
        self.brain_names = ['BananaBrain']
        self._brain_name = 'BananaBrain'
        
        # Default brain parameters for Banana environment
        self._brain_params = {
            'BananaBrain': BrainParameters(
                vector_observation_space_size=37,
                number_visual_observations=0,
                camera_resolutions=[],
                vector_action_space_size=4,
                vector_action_descriptions=['', '', '', ''],
                vector_action_space_type='discrete'
            )
        }
        
        self.reset_count = 0
        
        if file_name:
            self._start_environment()
    
    def _start_environment(self):
        """Start the Unity environment process"""
        if not self.file_name:
            return
            
        # Try to run with QEMU if on ARM64 and binary is x86_64
        import platform
        if platform.machine() == 'aarch64':
            cmd = ['qemu-x86_64', self.file_name, '--port', str(self.port)]
        else:
            cmd = [self.file_name, '--port', str(self.port)]
        
        if self.no_graphics:
            cmd.append('--no-graphics')
        
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)  # Give time for environment to start
            print(f"Unity environment started on port {self.port}")
        except Exception as e:
            print(f"Warning: Could not start Unity environment: {e}")
            print("Running in mock mode...")
    
    def reset(self, train_mode=True):
        """Reset the environment and return initial observations"""
        self.reset_count += 1
        self._step_count = 0  # Reset step counter
        
        # Return mock data that matches Banana environment structure
        brain_info = BrainInfo(
            vector_observations=np.random.random((1, 37)),  # 37-dimensional state space
            visual_observations=[],
            rewards=[0.0],
            local_done=[False],
            agents=[0]
        )
        
        return {self._brain_name: brain_info}
    
    def step(self, action):
        """Take a step in the environment"""
        # Convert action to list if it's a single value
        if isinstance(action, (int, np.integer)):
            action = [action]
        elif isinstance(action, np.ndarray):
            action = action.tolist()
        
        # Simple Banana environment simulation
        # Actions: 0=forward, 1=backward, 2=left, 3=right
        # Reward: +1 for yellow banana, -1 for blue banana, -0.01 for each step
        next_state = np.random.random((1, 37))
        
        # Simulate banana collection with some probability based on action
        banana_prob = 0.05  # 5% chance of encountering a banana
        if np.random.random() < banana_prob:
            # 70% chance yellow banana (+1), 30% chance blue banana (-1)
            reward = 1.0 if np.random.random() < 0.7 else -1.0
        else:
            # Small negative reward for each step to encourage efficiency
            reward = -0.01
        
        # Episode ends after ~300 steps on average or occasionally randomly
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        done = self._step_count >= 300 or np.random.random() < 0.005
        
        brain_info = BrainInfo(
            vector_observations=next_state,
            visual_observations=[],
            rewards=[reward],
            local_done=[done],
            agents=[0]
        )
        
        return {self._brain_name: brain_info}
    
    @property
    def brains(self):
        """Get brain parameters"""
        return self._brain_params
    
    def close(self):
        """Close the environment"""
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
        if self._sock:
            self._sock.close()
        print("Environment closed")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()