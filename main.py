# main.py

import torch
import numpy as np
import math
from itertools import count
import matplotlib.pyplot as plt
from pathlib import Path

from neural_symbolic_sssp.sssp_dqn import NeuroSymbolicSSSP_DQN

# --- Environment Class (Only GridMazeEnv is present) ---
class GridMazeEnv:
    def __init__(self, size=10, walls=None, goal=None):
        self.size = size
        self.walls = set(walls) if walls else set()
        self.goal = tuple(goal) if goal else (size - 1, size - 1)
        self.state = None
        self.state_dim = 2
        self.action_dim = 4 # 0:N, 1:S, 2:E, 3:W
        self.last_action = None

    def reset(self):
        self.last_action = None
        while True:
            self.state = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if self.state not in self.walls and self.state != self.goal:
                break
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        if action == opposites.get(self.last_action):
            reward = -1.0
            done = False
            return np.array(self.state, dtype=np.float32), reward, done

        self.last_action = action
        r, c = self.state
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c += 1
        elif action == 3: c -= 1

        if not (0 <= r < self.size and 0 <= c < self.size) or (r, c) in self.walls:
            pass # Hit a wall, state does not change
        else:
            self.state = (r, c) # Valid move, update state

        done = self.state == self.goal
        reward = 0.0 if done else -1.0

        return np.array(self.state, dtype=np.float32), reward, done

# --- Plotting Functions (remain the same) ---
def plot_rewards(episode_rewards):
    # --- NEW: Use pathlib ---
    output_dir = Path("outputs")
    # Good practice: create the directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    save_path = output_dir / "learning_curve.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved learning curve to {save_path}")

def plot_trajectory(trajectory, goal, walls, size, episode_num):
    # --- NEW: Use pathlib ---
    output_dir = Path("outputs")
    # Good practice: create the directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 8))
    if walls:
        wall_x = [w[1] for w in walls]
        wall_y = [w[0] for w in walls]
        plt.scatter(wall_x, wall_y, marker='s', s=150, color='black', label='Walls')

    traj_x = [p[1] for p in trajectory]
    traj_y = [p[0] for p in trajectory]
    plt.plot(traj_x, traj_y, '-o', label=f'Agent Path Ep {episode_num}', markersize=4, zorder=2)

    plt.plot(traj_x[0], traj_y[0], 'go', markersize=12, label='Start', zorder=3)
    plt.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal', zorder=3)

    plt.title(f'Agent Trajectory for Episode {episode_num}')
    plt.xlabel('X Coordinate (Column)')
    plt.ylabel('Y Coordinate (Row)')
    plt.xlim(-0.5, size - 0.5)
    plt.ylim(-0.5, size - 0.5)
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    save_path = output_dir / f'maze_trajectory_episode_{episode_num}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved maze trajectory plot to maze_trajectory_episode_{episode_num}.png")

# --- Configuration ---
config = {
    'num_episodes': 500, 'max_steps_per_episode': 150,
    'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05,
    'eps_decay': 20000, 'lr_dqn': 1e-5, 'lr_wm': 1e-3,
    'replay_capacity': 10000, 'batch_size': 128, 'target_update_freq': 10,
    'planning_k': 3, 'planning_H': 20, 'planning_N': 75,
    'planning_C': 20.0,
}

# In main.py

if __name__ == '__main__':
     # --- NEW: Check for CUDA, then MPS, then CPU ---
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
    agent = NeuroSymbolicSSSP_DQN(
        state_dim=env.state_dim, action_dim=env.action_dim,
        goal_state=np.array(goal_state, dtype=np.float32), config=config, device=device
    )

    episode_rewards = []
    plot_every_n_episodes = 25

    # --- NEW: Create a fixed set of states for evaluating Q-values ---
    fixed_states_for_q_eval = [env.reset() for _ in range(32)]
    fixed_states_for_q_eval = torch.tensor(np.array(fixed_states_for_q_eval), device=device, dtype=torch.float32)


    for i_episode in range(config['num_episodes']):
        state_np = env.reset()
        state = torch.tensor(np.array([state_np]), device=device, dtype=torch.float32)

        # --- NEW: Add trackers for new stats ---
        total_reward = 0
        trajectory = [state_np]
        dqn_losses = []
        wm_losses = []
        total_successful_plans = 0

        for t in range(config['max_steps_per_episode']):
            action = agent.select_action(state, evaluate=False)
            next_state_np, reward, done = env.step(action.item())

            total_reward += reward
            trajectory.append(next_state_np)

            action_val = action.item()
            agent.memory.push(state, torch.tensor([[action_val]], device=device, dtype=torch.long),
                              torch.tensor(np.array([next_state_np]), device=device, dtype=torch.float32),
                              torch.tensor([reward], device=device, dtype=torch.float32))
            state = torch.tensor(np.array([next_state_np]), device=device, dtype=torch.float32)

            # --- NEW: Capture the returned stats ---
            dqn_loss = agent.update_direct_rl()
            if dqn_loss is not None: dqn_losses.append(dqn_loss)

            wm_loss = agent.update_world_model()
            if wm_loss is not None: wm_losses.append(wm_loss)

            successful_plans = agent.planning_phase(t, config['max_steps_per_episode'])
            total_successful_plans += successful_plans

            if done:
                break

        # --- End of episode loop ---
        episode_rewards.append(total_reward)
        final_steps = t + 1

        # --- NEW: Calculate and format the new stats for printing ---
        avg_dqn_loss = np.mean(dqn_losses) if dqn_losses else 0
        avg_wm_loss = np.mean(wm_losses) if wm_losses else 0
        total_planning_ops = final_steps * config['planning_k']
        planner_success_rate = total_successful_plans / total_planning_ops if total_planning_ops > 0 else 0
        avg_q_value = agent.get_avg_q_value(fixed_states_for_q_eval)

        print(f"\n--- Episode {i_episode:4d} Summary ---")
        print(f"  Steps: {final_steps:3d} | Total Reward: {total_reward:6.2f} | Epsilon: {agent.get_epsilon():.2f}")
        print(f"  Avg DQN Loss: {avg_dqn_loss:7.4f} | Avg WM Loss: {avg_wm_loss:7.4f}")
        print(f"  Planner Success: {planner_success_rate:6.1%} | Avg Q-Value: {avg_q_value:8.2f}")

        if i_episode % plot_every_n_episodes == 0 or i_episode == config['num_episodes'] - 1:
            plot_trajectory(trajectory, goal_state, env.walls, env.size, i_episode)

        if i_episode % config['target_update_freq'] == 0:
            agent.update_target_net()

    print("\nTraining complete.")
    plot_rewards(episode_rewards)

    # --- NEW: Save the trained model's weights ---
    torch.save(agent.policy_net.state_dict(), 'agent_policy.pth')
    print("Saved trained policy network to agent_policy.pth")
