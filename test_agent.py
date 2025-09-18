# test_agent.py

import torch
import numpy as np
import time
from main import GridMazeEnv, plot_trajectory, config # Reuse our environment and plotting
from neural_symbolic_sssp.sssp_dqn import NeuroSymbolicSSSP_DQN # Reuse the agent and config

def test_agent():
    # --- 1. SETUP ---
    # Use the same device and environment setup as in training
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    goal_state = (8, 8)
    walls = [(i, 4) for i in range(2, 9)] + [(2, i) for i in range(5, 8)] + [(8, i) for i in range(5, 8)]
    env = GridMazeEnv(size=10, walls=walls, goal=goal_state)

    # --- 2. LOAD THE AGENT AND WEIGHTS ---
    agent = NeuroSymbolicSSSP_DQN(
        state_dim=env.state_dim, action_dim=env.action_dim,
        goal_state=np.array(goal_state, dtype=np.float32), config=config, device=device
    )

    # Load the dictionary of weights we saved
    agent.policy_net.load_state_dict(torch.load('agent_policy.pth', map_location=device))
    # Set the network to evaluation mode
    agent.policy_net.eval()

    print("Agent loaded and ready for testing.")

    # --- 3. RUN A TEST EPISODE ---
    state_np = env.reset()
    trajectory = [state_np]
    total_reward = 0

    for t in range(config['max_steps_per_episode']):
        state = torch.tensor(np.array([state_np]), device=device, dtype=torch.float32)

        # Get the greedy action (no exploration)
        action = agent.select_action(state, evaluate=True)

        next_state_np, reward, done = env.step(action.item())

        trajectory.append(next_state_np)
        state_np = next_state_np
        total_reward += reward

        if done:
            break

    print(f"\nTest episode finished in {t+1} steps with a total reward of {total_reward:.2f}")
    plot_trajectory(trajectory, goal_state, env.walls, env.size, "test")

if __name__ == '__main__':
    test_agent()
