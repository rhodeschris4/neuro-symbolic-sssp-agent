# Neuro-Symbolic Shortest Path Agent

This project is a Python implementation of a neuro-symbolic reinforcement learning agent. It combines a Deep Q-Network (DQN) with a symbolic shortest-path planner to solve grid-world environments like mazes. The core of the planner is a C++ implementation of the BMSSP algorithm, as described in the paper "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths."

## Features
- **Deep Q-Network (DQN):** The core model-free component for learning action values.
- **World Model:** A neural network that learns the environment's dynamics, enabling planning.
- **Symbolic Planner:** A high-performance C++ implementation of the BMSSP algorithm for finding optimal short-horizon plans.
- **Batched Graph Generation:** Efficiently constructs planning graphs using batched tensor operations.
- **Customizable Grid-World Environments:** Includes easily configurable Maze and Lava Pit environments.
- **Performance Tracking:** Generates learning curves and agent trajectory plots to visualize performance.

## Setup and Installation

**1. Clone the Repository**
```bash
[git https://github.com/rhodeschris4/neuro-symbolic-sssp-agent
cd neural-symbolic-bssmp-dqn
```

**2. Create and Activate a Virtual Environment**
```bash
# For macOS / Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Python Dependencies**
```bash
pip install -r requirements.txt
```

**4. Compile the C++ Module**
This command will compile the `bmss_p.cpp` file and install it into your environment.
```bash
pip install .
```

## How to Run

**To Train the Agent:**
The main script will run the training for the number of episodes specified in the `config` dictionary and save the final model weights and performance plots.
```bash
python3 main.py
```

**To Test a Trained Agent:**
After training is complete, this script will load the saved model weights (`agent_policy.pth`) and run the agent in a purely greedy evaluation mode.
```bash
python3 test_agent.py
```
