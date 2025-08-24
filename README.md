# Navigation Project - Advanced Deep Reinforcement Learning

This project implements multiple Deep Q-Network (DQN) variants to solve the Unity Banana Collector environment, featuring state-of-the-art improvements including Double DQN, Dueling DQN, and Prioritized Experience Replay.

## Project Details

### Environment Description
The agent navigates in a large, square world collecting bananas:
- **Goal**: Collect yellow bananas (+1 reward) while avoiding blue bananas (-1 reward)
- **State Space**: 37 dimensions including agent's velocity and ray-based perception of objects
- **Action Space**: 4 discrete actions
  - 0: Move forward
  - 1: Move backward
  - 2: Turn left
  - 3: Turn right
- **Solving Criteria**: Average score of +13 over 100 consecutive episodes

## Getting Started

### Prerequisites
- Python 3.10
- Unity ML-Agents
- PyTorch
- NumPy
- Matplotlib
- TensorBoard (for logging)
- pytest (for testing)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/navigation-dqn.git
cd Navigation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

   This will install all required packages including:
   - Core ML libraries (PyTorch, NumPy, Matplotlib)
   - Unity ML-Agents environment support  
   - TensorBoard for training visualization
   - pytest and testing utilities

3. Download the Unity Environment:
- The environment is pre-installed in the Udacity workspace at `/data/Banana_Linux_NoVis/Banana.x86_64`
- For local setup, download the appropriate version for your OS from the [Unity ML-Agents GitHub repository](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#banana-collector)

4. Update the environment path in `src/navigation/config.py`:
```python
UNITY_ENV_PATH = "/path/to/your/Banana.x86_64"  # Update this path
```

5. **Verify installation** by running the test suite:
```bash
# Quick installation verification
python scripts/run_tests.py

# If all tests pass, your environment is ready!
```

   **Note**: Some tests may be skipped if running without Unity environment, but core functionality tests should pass.

## DQN Variants Implemented

This project includes multiple state-of-the-art DQN improvements:

### 🚀 **Double DQN**
- Reduces overestimation bias by decoupling action selection from evaluation
- Uses local network to select actions, target network to evaluate them
- Typically improves sample efficiency and stability

### 🎯 **Dueling DQN** 
- Separates value function and advantage function estimation
- Better understands state values independent of action choices
- Particularly effective in environments where most actions don't affect the environment

### ⭐ **Prioritized Experience Replay**
- Samples important experiences more frequently based on TD error magnitude
- Uses importance sampling weights to correct for bias
- Significantly improves sample efficiency by focusing on informative experiences

### 🌈 **Rainbow DQN**
- Combines Double DQN + Dueling DQN + Prioritized Experience Replay
- State-of-the-art performance combining all improvements

## Instructions

### Command Line Training

Train different DQN variants using the command line:

```bash
# Standard DQN
python scripts/train.py --train

# Double DQN
python scripts/train.py --train --double-dqn

# Dueling DQN
python scripts/train.py --train --dueling-dqn

# Prioritized Experience Replay
python scripts/train.py --train --prioritized

# Rainbow (all features)
python scripts/train.py --train --double-dqn --dueling-dqn --prioritized

# Test trained agent
python scripts/train.py --test --checkpoint results/checkpoint.pth
```

### Compare All Variants

Run a comprehensive comparison of all DQN variants:

```bash
python scripts/compare_variants.py
```

This will:
- Train 6 different DQN variants
- Generate comparison plots (training curves, sample efficiency, performance)
- Create a summary table of results
- Save comparison visualization as `dqn_variants_comparison.png`

### Jupyter Notebook Training

1. **Original notebook** (monolithic code):
```bash
jupyter notebook notebooks/Navigation.ipynb
```

2. **Clean modular notebook**:
```bash
jupyter notebook notebooks/Navigation_Clean.ipynb
```

The clean notebook uses the new modular code structure for better organization.

## Testing

This project includes a comprehensive testing suite to ensure code reliability and catch regressions.

### Running Tests

**Option 1: Custom Test Runner**
```bash
# Run all tests
python scripts/run_tests.py

# Run tests with different verbosity levels
python scripts/run_tests.py --verbosity 0  # Quiet
python scripts/run_tests.py --verbosity 1  # Normal  
python scripts/run_tests.py --verbosity 2  # Verbose (default)

# Run specific test module
python scripts/run_tests.py --test tests.test_models

# Run specific test class
python scripts/run_tests.py --test tests.test_models.TestQNetwork

# Run specific test method
python scripts/run_tests.py --test tests.test_models.TestQNetwork.test_forward_pass

# List all available tests
python scripts/run_tests.py --list
```

**Option 2: pytest**
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=. --cov-report=html tests/

# Run specific test file
pytest tests/test_models.py -v

# Run tests matching pattern
pytest tests/ -k "test_network"
```

### Test Coverage

The testing suite covers all major components:

- **🧠 Neural Networks** (`test_models.py`)
  - QNetwork and DuelingQNetwork architecture validation
  - Forward pass with different input sizes
  - Gradient flow and parameter updates
  - Dueling architecture mathematical correctness

- **🗃️ Replay Buffers** (`test_buffer.py`, `test_prioritized_buffer.py`)
  - Standard and prioritized experience replay
  - Memory management and circular buffer behavior
  - Priority-based sampling and importance sampling weights
  - SumTree data structure functionality

- **🤖 Agents** (`test_agents.py`)
  - Standard and prioritized agent initialization
  - Double DQN and Dueling DQN variants
  - Action selection (epsilon-greedy vs greedy)
  - Learning process and network updates
  - Integration between agents, networks, and buffers

### Test Features

- **Comprehensive**: Tests all core classes and methods
- **Robust**: Handles edge cases and error conditions
- **Reproducible**: Seed-based testing for consistent results
- **Fast**: Unit tests run in seconds
- **Informative**: Clear output showing exactly what passed/failed

### Continuous Integration

The tests are designed to run in CI/CD pipelines:

```bash
# Example CI command
python scripts/run_tests.py --verbosity 1
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Tests failed!"
    exit 1
fi
```

## Project Structure

```
├── src/                       # Source code
│   └── navigation/           # Main package
│       ├── __init__.py      # Package init
│       ├── config.py        # Centralized configuration
│       ├── agents/          # DQN agent implementations
│       │   ├── __init__.py
│       │   ├── dqn_agent.py    # Standard DQN agent (Double + Dueling)
│       │   └── prioritized_agent.py # Prioritized Experience Replay agent
│       ├── models/          # Neural network architectures
│       │   ├── __init__.py
│       │   └── qnetwork.py     # QNetwork and DuelingQNetwork
│       └── buffers/         # Experience replay buffers
│           ├── __init__.py
│           ├── replay_buffer.py # Standard experience replay
│           └── prioritized_buffer.py # Prioritized experience replay
├── scripts/                   # Executable scripts
│   ├── train.py              # Command-line training script
│   ├── compare_variants.py   # DQN variants comparison tool
│   └── run_tests.py          # Test runner script
├── notebooks/                 # Jupyter notebooks
│   ├── Navigation.ipynb      # Original training notebook (monolithic)
│   └── Navigation_Clean.ipynb # Clean modular training notebook
├── tests/                     # Test suite directory
│   ├── __init__.py           # Test package init
│   ├── test_models.py        # Neural network tests
│   ├── test_buffer.py        # Replay buffer tests
│   ├── test_prioritized_buffer.py # Prioritized buffer tests
│   └── test_agents.py        # Agent tests
├── results/                   # Training results and outputs
│   ├── checkpoint.pth        # Saved model weights (created after training)
│   ├── scores.npy           # Training scores (created after training)
│   ├── training_progress.png # Training visualization (created after training)
│   └── dqn_variants_comparison.png # Comparison plots (created after comparison)
├── docs/                      # Documentation
│   └── Report.md             # Detailed project report
├── pytest.ini                # pytest configuration
├── requirements.txt           # Python dependencies (includes testing)
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

## Results

### Original Implementation
The standard DQN agent successfully solves the environment in **390 episodes**, achieving an average score of +13.01 over 100 consecutive episodes.

![Training Progress](training_progress.png)

### Advanced Variants Performance
The advanced DQN variants typically show improved performance:

- **Double DQN**: Often solves 50-100 episodes faster, more stable learning
- **Dueling DQN**: Better final performance, especially in complex state spaces  
- **Prioritized Experience Replay**: Significant sample efficiency improvements (30-50% faster)
- **Rainbow (all combined)**: Best overall performance, fastest convergence

Run `python scripts/compare_variants.py` to generate your own comparison results!

## Configuration

All hyperparameters are centralized in `src/navigation/config.py`:

```python
# Training hyperparameters
LEARNING_RATE = 5e-4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4

# DQN Variant Settings  
DOUBLE_DQN = False
DUELING_DQN = False
PRIORITIZED_REPLAY = False

# Prioritized Experience Replay parameters
ALPHA = 0.6  # prioritization exponent
BETA = 0.4   # importance sampling exponent
```

Easily modify these values to experiment with different configurations.

## Advanced Usage Examples

### Custom Hyperparameter Training
```python
from src.navigation.agents import Agent
from src.navigation.config import *

# Create agent with custom parameters
agent = Agent(
    state_size=37, action_size=4, seed=42,
    lr=1e-3, batch_size=128, 
    double_dqn=True, dueling_dqn=True
)
```

### Prioritized Experience Replay
```python  
from src.navigation.agents import PrioritizedAgent

# Create agent with prioritized replay
agent = PrioritizedAgent(
    state_size=37, action_size=4, seed=42,
    alpha=0.7,  # Higher prioritization
    beta=0.5,   # More importance sampling correction
    double_dqn=True, dueling_dqn=True
)
```

### Loading and Testing Models
```python
# Load any variant's trained model
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# Test performance
scores = []
for episode in range(10):
    score = run_episode(agent, env, train_mode=False)
    scores.append(score)
    
print(f"Average test score: {np.mean(scores):.2f}")
```

### Development and Testing Workflow

**1. Making Changes**
```bash
# Make your code changes
nano src/navigation/agents/dqn_agent.py

# Run tests to ensure nothing broke
python scripts/run_tests.py

# Run specific tests for the component you changed
python scripts/run_tests.py --test tests.test_agents
```

**2. Adding New Features**
```bash
# Add your new feature
nano src/navigation/new_feature.py

# Write tests for the new feature
nano tests/test_new_feature.py

# Run all tests to ensure integration
python scripts/run_tests.py
```

**3. Before Committing**
```bash
# Run full test suite
python scripts/run_tests.py

# Check test coverage
pytest --cov=src --cov-report=term-missing tests/

# Run performance comparison if needed
python scripts/compare_variants.py
```

### Debugging Failed Tests

If tests fail, you can debug them:

```bash
# Run failed test with maximum verbosity
python scripts/run_tests.py --test tests.test_models.TestQNetwork.test_forward_pass --verbosity 2

# Use pytest for debugging
pytest tests/test_models.py::TestQNetwork::test_forward_pass -v -s

# Run with pdb debugger
pytest tests/test_models.py --pdb
```

## Author

[Your Name]

## License

This project is licensed under the MIT License.