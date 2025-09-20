# test_agent.py

import torch
import numpy as np
import time
from pathlib import Path
# We need to import the new, generalized environment from main
from main import GridMazeEnv, plot_trajectory
from neural_symbolic_sssp.sssp_dqn import NeuroSymbolicSSSP_DQN

def test_agent(model_path):
    """
    Loads a trained agent and runs it for a few test episodes
    in a purely greedy (evaluation) mode.
    """
    # --- 1. SETUP ---
    # Define the configuration that the agent was trained with.
    # This should match the experiment you want to test.
    # For example, this matches the "Standard_Guided_Run".
    config = {
        'num_episodes': 1000, # Not used in testing, but good for consistency
        'max_steps_per_episode': 150,
        'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05,
        'eps_decay': 20000, 'lr_dqn': 1e-4, 'lr_wm': 1e-3,
        'replay_capacity': 10000, 'batch_size': 128,
        'target_update_freq': 500, 'planning_N': 75,
        'planning_C': 20.0, 'planner_intervention_freq': 25,
        'forced_exploration_steps': 5, 'planning_H': 15, 'planning_k': 3,
    }

    # Use the same device and environment setup as in training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    walls = [(i, 4) for i in range(2, 9)] + [(2, i) for i in range(5, 8)] + [(8, i) for i in range(5, 8)]
    env = GridMazeEnv(size=10, walls=walls)

    # --- 2. LOAD THE AGENT AND WEIGHTS ---
    agent = NeuroSymbolicSSSP_DQN(
        state_dim=env.state_dim,  # Uses the correct 4D state_dim
        action_dim=env.action_dim,
        config=config,
        device=device
    )

    # Load the dictionary of weights from the specified path
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    agent.policy_net.load_state_dict(torch.load(model_file, map_location=device))
    # Set the network to evaluation mode (this disables dropout, etc.)
    agent.policy_net.eval()

    print(f"Agent loaded from {model_path} and ready for testing.")

    # --- 3. RUN SEVERAL TEST EPISODES ---
    num_test_episodes = 5
    for i in range(num_test_episodes):
        state_np = env.reset()
        # The goal is now dynamic, so we get it from the environment for this episode
        current_goal = env.goal
        trajectory = [state_np[:2]]
        total_reward = 0
        print(f"\n--- Starting Test Episode {i+1}/{num_test_episodes} ---")
        print(f"Goal for this episode: {current_goal}")

        for t in range(config['max_steps_per_episode']):
            # State needs to be a 2D tensor: [[agent_x, agent_y, goal_x, goal_y]]
            state = torch.tensor(np.array([state_np]), device=device, dtype=torch.float32)

            # Get the greedy action (no exploration)
            action = agent.select_action(state, evaluate=True)
            action_val = action.item()

            next_state_np, reward, done = env.step(action_val)

            trajectory.append(next_state_np[:2])
            state_np = next_state_np
            total_reward += reward

            if done:
                break

        print(f"  > Episode finished in {t+1} steps with a total reward of {total_reward:.2f}")
        # Save the trajectory plot for this test run
        plot_save_path = model_file.parent / f"test_trajectory_run_{i+1}.png"
        plot_trajectory(trajectory, current_goal, env.walls, env.size, f"test_{i+1}", plot_save_path)
        print(f"  > Saved test trajectory plot to {plot_save_path}")
        time.sleep(1) # Pause to ensure files are written

if __name__ == '__main__':
    # --- IMPORTANT ---
    # You must provide the path to the trained model you want to test.
    # Find the 'agent_policy.pth' file inside one of your successful
    # experiment folders.
    # Example: 'experiment_results/20231027-143000_Standard_Guided_Run/agent_policy.pth'

    pathToModel = "./agent_policy.pth"

    test_agent(pathToModel)
