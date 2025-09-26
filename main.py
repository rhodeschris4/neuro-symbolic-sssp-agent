import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from neural_symbolic_sssp.sssp_dqn import NeuroSymbolicSSSP_DQN

# --- Environment Class (MODIFIED FOR GENERALIZATION) ---
class GridMazeEnv:
    def __init__(self, size=10, walls=None):
        self.size = size
        self.walls = set(walls) if walls else set()
        self.goal = None # Goal is now set in reset()
        self.state = None
        self.state_dim = 4 # agent_x, agent_y, goal_x, goal_y
        self.action_dim = 4 # 0:N, 1:S, 2:E, 3:W
        self.last_action = None

    def _get_random_valid_pos(self):
        """Helper to get a random position that is not a wall."""
        while True:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in self.walls:
                return pos

    def reset(self):
        """Resets the agent and places the goal in a new random location."""
        self.last_action = None
        self.goal = self._get_random_valid_pos()
        self.state = self._get_random_valid_pos()

        # Ensure agent and goal are not in the same spot
        while self.state == self.goal:
            self.state = self._get_random_valid_pos()

        # Return the new 4D state representation
        return np.array(self.state + self.goal, dtype=np.float32)

    def step(self, action):
        # ... (step logic remains exactly the same) ...
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        if action == opposites.get(self.last_action):
            reward = -1.0
            done = False
            return np.array(self.state + self.goal, dtype=np.float32), reward, done

        self.last_action = action
        r, c = self.state
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c += 1
        elif action == 3: c -= 1

        if not (0 <= r < self.size and 0 <= c < self.size) or (r, c) in self.walls:
            pass
        else:
            self.state = (r, c)

        done = self.state == self.goal
        reward = 0.0 if done else -1.0
        return np.array(self.state + self.goal, dtype=np.float32), reward, done

# --- Plotting Functions (no changes here) ---
# ... (plot_rewards and plot_trajectory functions remain the same) ...
def plot_rewards(episode_rewards, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_trajectory(trajectory, goal, walls, size, episode_num, save_path):
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
    plt.savefig(save_path)
    plt.close()


# --- Core Training Logic (MODIFIED FOR GENERALIZATION) ---
def run_training(config, output_dir):
    # --- Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    non_blocking = True if device.type == 'cuda' else False

    walls = [(i, 4) for i in range(2, 9)] + [(2, i) for i in range(5, 8)] + [(8, i) for i in range(5, 8)]
    env = GridMazeEnv(size=10, walls=walls) # Goal is no longer fixed

    agent = NeuroSymbolicSSSP_DQN(
        state_dim=env.state_dim, # Now correctly uses state_dim = 4
        action_dim=env.action_dim,
        config=config,
        device=device
    )

    episode_rewards = []
    # Increase episodes for this harder task
    config['num_episodes'] = config.get('num_episodes', 500) * 3
    plot_every_n_episodes = config['num_episodes'] // 10

    # No longer have a single fixed set of states for Q-eval
    total_steps = 0

    # --- Training Loop ---
    for i_episode in range(config['num_episodes']):
        state_np = env.reset()
        state = torch.tensor(np.array([state_np]), device=device, dtype=torch.float32)

        total_reward = 0
        trajectory = [state_np[:2]] # Only track the agent's position for plotting
        dqn_losses = []
        wm_losses = []
        action_plan = []
        planning_opportunities = 0
        total_successful_plans = 0

        for t in range(config['max_steps_per_episode']):
            # ... (inner loop logic is largely the same, but handles 4D state) ...
            if not action_plan and t > 0 and t % config.get('planner_intervention_freq', 99999) == 0:
                planning_opportunities += 1
                plan, success = agent.plan_action_sequence(state)
                if success:
                    total_successful_plans += success
                    action_plan = plan[:config.get('forced_exploration_steps', 0)]

            if action_plan:
                action_val = action_plan.pop(0)
                action = torch.tensor([[action_val]], device=device, dtype=torch.long)
            else:
                action = agent.select_action(state, evaluate=False)

            action_val = action.item()
            next_state_np, reward, done = env.step(action_val)
            total_reward += reward
            trajectory.append(next_state_np[:2])

            # Push 4D states to memory
            # Create the next_state_tensor once, the efficient way
            next_state_tensor = torch.tensor(np.array([next_state_np]), device=device, dtype=torch.float32)

            # Now use this efficient tensor for both operations
            agent.memory.push(
                state,
                action,
                next_state_tensor, # Use the pre-made tensor
                torch.tensor([reward], device=device, dtype=torch.float32),
                torch.tensor([float(done)], device=device, dtype=torch.float32)
            )
            state = next_state_tensor # And reuse it here

            # ... (learning updates and target network updates remain the same) ...
            dqn_loss = agent.update_direct_rl()
            if dqn_loss is not None: dqn_losses.append(dqn_loss)
            wm_loss = agent.update_world_model()
            if wm_loss is not None: wm_losses.append(wm_loss)

            if total_steps % config['target_update_freq'] == 0:
                agent.update_target_net()
            total_steps += 1
            if done:
                break

        episode_rewards.append(total_reward)

        if (i_episode + 1) % 50 == 0:
            avg_dqn_loss = np.mean(dqn_losses) if dqn_losses else 0
            avg_wm_loss = np.mean(wm_losses) if wm_losses else 0
            planner_success_rate = total_successful_plans / planning_opportunities if planning_opportunities > 0 else 0

            print(f"\n--- Episode {i_episode+1:4d} Summary ---")
            print(f"  Steps: {t+1:3d} | Total Reward: {total_reward:6.2f} | Epsilon: {agent.get_epsilon():.2f}")
            print(f"  Avg DQN Loss: {avg_dqn_loss:7.4f} | Avg WM Loss: {avg_wm_loss:7.4f}")
            print(f"  Planner Success: {planner_success_rate:.1%}")

        if (i_episode + 1) % plot_every_n_episodes == 0 or i_episode == config['num_episodes'] - 1:
            traj_save_path = output_dir / f'trajectory_ep_{i_episode+1}.png'
            plot_trajectory(trajectory, env.goal, env.walls, env.size, i_episode+1, traj_save_path)

    # ... (finalization and return logic remains the same) ...
    print("\nTraining complete.")
    plot_rewards(episode_rewards, output_dir / "learning_curve.png")
    torch.save(agent.policy_net.state_dict(), output_dir / 'agent_policy.pth')

    final_metrics = {
        "average_reward_last_50_eps": np.mean(episode_rewards[-50:]),
    }
    return final_metrics
