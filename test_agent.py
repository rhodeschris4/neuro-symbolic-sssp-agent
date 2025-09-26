# test_agent.py
import torch
import numpy as np
import time
import json # Add json import
from pathlib import Path
from main import GridMazeEnv, plot_trajectory
from neural_symbolic_sssp.sssp_dqn import NeuroSymbolicSSSP_DQN

def test_agent(model_path):
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # --- BEST PRACTICE: Load config automatically from the report.json file ---
    report_path = model_file.parent / "report.json"
    if not report_path.exists():
        print(f"Error: report.json not found in the same folder as the model!")
        return
    with open(report_path, 'r') as f:
        report = json.load(f)
    config = report['configuration']
    print("Successfully loaded configuration from report.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    walls = [(i, 4) for i in range(2, 9)] + [(2, i) for i in range(5, 8)] + [(8, i) for i in range(5, 8)]
    env = GridMazeEnv(size=10, walls=walls)

    agent = NeuroSymbolicSSSP_DQN(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=config,
        device=device
    )
    agent.policy_net.load_state_dict(torch.load(model_file, map_location=device))
    agent.policy_net.eval()
    print(f"Agent loaded from {model_path} and ready for testing.")

    num_test_episodes = 5
    for i in range(num_test_episodes):
        state_np = env.reset()
        current_goal = env.goal
        trajectory = [state_np[:2]]
        total_reward = 0
        print(f"\n--- Starting Test Episode {i+1}/{num_test_episodes} ---")
        print(f"Goal for this episode: {current_goal}")

        for t in range(config['max_steps_per_episode']):
            state = torch.tensor(np.array([state_np]), device=device, dtype=torch.float32)
            action = agent.select_action(state, evaluate=True)
            action_val = action.item()
            next_state_np, reward, done = env.step(action_val)
            trajectory.append(next_state_np[:2])
            state_np = next_state_np
            total_reward += reward
            if done:
                break

        print(f"  > Episode finished in {t+1} steps with a total reward of {total_reward:.2f}")
        plot_save_path = model_file.parent / f"test_trajectory_run_{i+1}.png"
        plot_trajectory(trajectory, current_goal, env.walls, env.size, f"test_{i+1}", plot_save_path)
        print(f"  > Saved test trajectory plot to {plot_save_path}")
        time.sleep(1)

if __name__ == '__main__':
    # Now you just need to point to the model file.
    # The script will handle loading the correct config.
    pathToModel = "experiment_results/20250920-015513_Standard_Guided_Run/agent_policy.pth" # Example path
    test_agent(pathToModel)
