import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from neural_symbolic_sssp.sssp_dqn import NeuroSymbolicSSSP_DQN

class GridMazeEnv:
    def __init__(self, size=10, walls=None):
        self.size = size
        self.walls = set(walls) if walls else set()
        self.goal = None
        self.state = None
        self.state_dim = 4 # agent_x, agent_y, goal_x, goal_y
        self.action_dim = 4 # 0:N, 1:S, 2:E, 3:W
        self.last_action = None

    def _get_random_valid_pos(self):
        while True:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in self.walls:
                return pos

    def reset(self):
        self.last_action = None
        self.goal = self._get_random_valid_pos()
        self.state = self._get_random_valid_pos()
        while self.state == self.goal:
            self.state = self._get_random_valid_pos()
        return np.array(self.state + self.goal, dtype=np.float32)

    def step(self, action):
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        if action == opposites.get(self.last_action):
            reward = -1.0
            return np.array(self.state + self.goal, dtype=np.float32), reward, False

        self.last_action = action
        r, c = self.state
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c += 1
        elif action == 3: c -= 1

        if not (0 <= r < self.size and 0 <= c < self.size) or (r, c) in self.walls:
            pass # Hit a wall, state does not change
        else:
            self.state = (r, c)

        done = self.state == self.goal
        reward = 0.0 if done else -1.0
        return np.array(self.state + self.goal, dtype=np.float32), reward, done

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
    plt.xlim(-0.5, size - 0.5)
    plt.ylim(-0.5, size - 0.5)
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def run_training(config, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    walls = [(i, 4) for i in range(2, 9)] + [(2, i) for i in range(5, 8)] + [(8, i) for i in range(5, 8)]
    env = GridMazeEnv(size=10, walls=walls)

    agent = NeuroSymbolicSSSP_DQN(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=config,
        device=device
    )

    num_episodes = config.get('num_episodes', 500)
    plot_every_n_episodes = num_episodes // 10
    episode_rewards = []
    total_steps = 0

    for i_episode in range(num_episodes):
        state_np = env.reset()
        state = torch.tensor(np.array([state_np]), device=device, dtype=torch.float32)

        total_reward = 0
        trajectory = [state_np[:2]]
        dqn_losses = []
        wm_losses = []
        total_successful_plans = 0
        planning_opportunities = 0

        for t in range(config['max_steps_per_episode']):
            action = agent.select_action(state, evaluate=False)
            next_state_np, reward, done = env.step(action.item())

            total_reward += reward
            trajectory.append(next_state_np[:2])

            next_state_tensor = torch.tensor(np.array([next_state_np]), device=device, dtype=torch.float32)
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)

            agent.memory.push(state, action, next_state_tensor, reward_tensor)
            state = next_state_tensor

            # --- UPDATED TRAINING STEP ---
            planning_loss = None
            plan_success_count = 0
            if total_steps > 0 and total_steps % config.get('planner_intervention_freq', 99999) == 0:
                planning_opportunities += config.get('planning_k', 1)
                for _ in range(config.get('planning_k', 1)):
                    p_loss, p_success = agent.planning_update()
                    if p_loss is not None:
                        planning_loss = p_loss
                    plan_success_count += p_success

            total_successful_plans += plan_success_count

            # The main optimization step now includes the optional planning loss
            dqn_loss = agent.optimize_model(planning_loss)
            if dqn_loss is not None:
                dqn_losses.append(dqn_loss)

            wm_loss = agent.update_world_model()
            if wm_loss is not None:
                wm_losses.append(wm_loss)

            if total_steps > 0 and total_steps % config['target_update_freq'] == 0:
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
            print(f"  Steps: {t+1:3d} | Reward: {total_reward:6.2f} | Epsilon: {agent.get_epsilon():.3f}")
            print(f"  Avg Losses (DQN/WM): {avg_dqn_loss:7.4f} / {avg_wm_loss:7.4f}")
            print(f"  Planner Success Rate: {planner_success_rate:.1%}")

        if (i_episode + 1) % plot_every_n_episodes == 0 or i_episode == num_episodes - 1:
            traj_save_path = output_dir / f'trajectory_ep_{i_episode+1}.png'
            plot_trajectory(trajectory, env.goal, env.walls, env.size, i_episode+1, traj_save_path)

    print("\nTraining complete.")
    plot_rewards(episode_rewards, output_dir / "learning_curve.png")
    torch.save(agent.policy_net.state_dict(), output_dir / 'agent_policy.pth')

    final_metrics = {
        "average_reward_last_50_eps": np.mean(episode_rewards[-50:]),
    }
    return final_metrics

if __name__ == '__main__':
    config = {
        'num_episodes': 1500,
        'max_steps_per_episode': 150,
        'gamma': 0.99,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'eps_decay': 80000, # Slower decay
        'replay_capacity': 10000,
        'batch_size': 128,
        'target_update_freq': 500,
        'planning_C': 20.0,
        'lr_dqn': 1e-4,
        'lr_wm': 1e-3,
        'planning_N': 75,
        'planning_H': 15,
        'planner_intervention_freq': 25,
        'planning_k': 3,
        'planning_loss_weight': 0.5 # NEW: Weight for the planner's loss
    }

    output_dir = Path("single_run_output")
    output_dir.mkdir(exist_ok=True)

    run_training(config, output_dir)
