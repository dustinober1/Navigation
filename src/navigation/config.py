"""Configuration file for DQN training hyperparameters."""

# Environment settings
UNITY_ENV_PATH = "/home/dobercode/Navigation/Banana_Linux_NoVis/Banana.x86_64"

# Network architecture
FC1_UNITS = 64
FC2_UNITS = 64

# Training hyperparameters
LEARNING_RATE = 5e-4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4

# Training parameters
N_EPISODES = 2000
MAX_STEPS = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

# Solved criteria
SOLVE_SCORE = 13.0
SOLVE_WINDOW = 100

# Saving
CHECKPOINT_PATH = "results/checkpoint.pth"
SCORES_PATH = "results/scores.npy"

# Random seed
SEED = 0

# DQN Variant Settings
DOUBLE_DQN = False
DUELING_DQN = False
PRIORITIZED_REPLAY = False

# Prioritized Experience Replay parameters
ALPHA = 0.6  # prioritization exponent
BETA = 0.4   # importance sampling exponent
BETA_INCREMENT = 0.001  # increment for beta per step