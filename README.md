# Navigation Project - Deep Reinforcement Learning

This project implements a Deep Q-Network (DQN) agent to solve the Unity Banana Collector environment.

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

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd navigation-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Unity Environment:
- The environment is pre-installed in the Udacity workspace at `/data/Banana_Linux_NoVis/Banana.x86_64`
- For local setup, download the appropriate version for your OS

## Instructions

### Training the Agent

1. Open the Jupyter notebook:
```bash
jupyter notebook Navigation.ipynb
```

2. Run all cells in sequence to:
   - Initialize the environment
   - Define the DQN agent architecture
   - Train the agent
   - Plot the results
   - Test the trained agent

3. The trained model weights will be saved as `checkpoint.pth`

### Using a Pre-trained Agent

To test a pre-trained agent:
```python
# Load the checkpoint
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# Run the test function
test_agent(num_episodes=5)
```

## Project Structure

```
├── Navigation.ipynb       # Main training notebook
├── checkpoint.pth         # Saved model weights
├── README.md             # This file
├── Report.md             # Detailed project report
└── requirements.txt      # Python dependencies
```

## Results

The agent successfully solves the environment in **390 episodes**, achieving an average score of +13.01 over 100 consecutive episodes.

## Author

[Your Name]

## License

This project is licensed under the MIT License.