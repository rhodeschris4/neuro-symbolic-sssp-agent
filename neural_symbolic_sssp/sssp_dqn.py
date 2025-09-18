# sssp_dqn.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
import math
import time

from neural_symbolic_sssp.models import DQN, WorldModel
from neural_symbolic_sssp.utils import ReplayBuffer, Transition
#from bmss_p import bmss_p
import bmss_p_cpp

class NeuroSymbolicSSSP_DQN:
    def __init__(self, state_dim, action_dim, goal_state, config, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # --- Initialize Networks ---
        # self.policy_net = DQN(state_dim, action_dim).to(self.device)
        # self.target_net = DQN(state_dim, action_dim).to(self.device)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()

        # self.world_model = WorldModel(state_dim, action_dim).to(self.device)

        # --- Initialize Networks ---
        # Create the model instances first
        policy_net_instance = DQN(state_dim, action_dim).to(self.device)
        target_net_instance = DQN(state_dim, action_dim).to(self.device)
        world_model_instance = WorldModel(state_dim, action_dim).to(self.device)

        # --- NEW: Apply TorchScript ---
        # This compiles the models for a potential speedup
        self.policy_net = torch.jit.script(policy_net_instance)
        self.target_net = torch.jit.script(target_net_instance)
        self.world_model = torch.jit.script(world_model_instance)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # -----------------------------


        # --- Optimizers ---
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config['lr_dqn'])
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=config['lr_wm'])

        self.memory = ReplayBuffer(config['replay_capacity'])

        # --- Neuro-Symbolic Parameters ---
        self.goal_state = torch.tensor(goal_state, dtype=torch.float32, device=self.device)
        self.steps_done = 0

    def select_action(self, state, evaluate=False):
        # sample = random.random()
        # eps_threshold = self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
        #     math.exp(-1. * self.steps_done / self.config['eps_decay'])
        # self.steps_done += 1

        # if sample > eps_threshold:
        #     with torch.no_grad():
        #         return self.policy_net(state).max(1)[1].view(1, 1)
        # else:
        #     return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
        # If we are in evaluation mode, always be greedy
        if evaluate:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

        # Otherwise, use epsilon-greedy for training
        sample = random.random()
        eps_threshold = self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
            math.exp(-1. * self.steps_done / self.config['eps_decay'])
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def potential_function(self, state):
        # Example potential: negative Euclidean distance to goal
        #return -torch.linalg.norm(state - self.goal_state, dim=-1) Euclidean Distance

        # Use Manhattan distance for a grid world: |x1 - x2| + |y1 - y2|
        return -torch.sum(torch.abs(state - self.goal_state), dim=-1)

    def update_direct_rl(self):
        # if len(self.memory) < self.config['batch_size']: return None
        #
        # transitions = self.memory.sample(self.config['batch_size'])
        # batch = Transition(*zip(*transitions))
        #
        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward)
        # next_state_batch = torch.cat(batch.next_state)
        #
        # q_values = self.policy_net(state_batch).gather(1, action_batch)
        # next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        #
        # expected_q_values = (next_q_values * self.config['gamma']) + reward_batch
        #
        # loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        #
        # self.policy_optimizer.zero_grad()
        # loss.backward()
        # self.policy_optimizer.step()
        if len(self.memory) < self.config['batch_size']: return None

        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # --- DOUBLE DQN LOGIC ---
        # 1. Get the best action for the next state from the POLICY network
        best_next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)

        # 2. Get the Q-value for that action from the TARGET network
        q_values_next_target = self.target_net(next_state_batch)
        next_q_values = q_values_next_target.gather(1, best_next_actions).squeeze(1).detach()
        # ------------------------

        expected_q_values = (next_q_values * self.config['gamma']) + reward_batch

        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()

    def update_world_model(self):
        if len(self.memory) < self.config['batch_size']: return

        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        pred_next_state, pred_reward = self.world_model(state_batch, action_batch.squeeze(1))

        state_loss = F.mse_loss(pred_next_state, next_state_batch)
        reward_loss = F.mse_loss(pred_reward.squeeze(1), reward_batch)
        loss = state_loss + reward_loss

        self.world_model_optimizer.zero_grad()
        loss.backward()
        self.world_model_optimizer.step()

        return loss.item() # --- CHANGE: Return the loss ---

    # In NeuroSymbolicSSSP_DQN class
    # NEW
    def _reconstruct_path_and_actions(self, start_node, end_node, preds, V_nodes):
        """Helper to reconstruct the path and corresponding actions from planner output."""
        if end_node not in preds and end_node != start_node:
            return [] # No path found

        path_nodes = []
        curr = end_node
        while curr != start_node:
            path_nodes.append(curr)
            if curr not in preds: return [] # Path is broken
            curr = preds[curr]
        path_nodes.append(start_node)
        path_nodes.reverse()

        actions = []
        for i in range(len(path_nodes) - 1):
            # Get coordinates from V_nodes
            pos_curr = V_nodes[path_nodes[i]].cpu().numpy().flatten()
            pos_next = V_nodes[path_nodes[i+1]].cpu().numpy().flatten()

            # Determine action based on change in coordinates
            delta_r, delta_c = pos_next[0] - pos_curr[0], pos_next[1] - pos_curr[1]

            if delta_r == -1 and delta_c == 0: actions.append(0) # North
            elif delta_r == 1 and delta_c == 0: actions.append(1) # South
            elif delta_r == 0 and delta_c == 1: actions.append(2) # East
            elif delta_r == 0 and delta_c == -1: actions.append(3) # West
            # If nodes are not adjacent, it's a model error. Stop the plan.
            else: return actions

        return actions

    #NEW
    def plan_action_sequence(self, current_state_tensor):
        """
        Generates a sequence of optimal actions using the world model and BMSSP planner.
        This implements the core of the "Guided Exploration" strategy.
        """
        if len(self.memory) < self.config['planning_N']:
            return [], 0

        # 1. Graph Extraction (same as before)
        sampled_transitions = self.memory.sample(self.config['planning_N'])
        V_states_list = [t.state for t in sampled_transitions] + [current_state_tensor]
        V_nodes = {i: state for i, state in enumerate(V_states_list)}
        all_V_states = torch.cat(V_states_list, dim=0)

        batch_states = all_V_states.repeat_interleave(self.action_dim, dim=0)
        batch_actions = torch.arange(self.action_dim, device=self.device).repeat(len(V_nodes))

        with torch.no_grad():
            pred_next_states, pred_rewards = self.world_model(batch_states, batch_actions)

        dists = torch.cdist(pred_next_states, all_V_states)
        closest_v_indices = torch.argmin(dists, dim=1)

        graph = defaultdict(list)
        phi_V = self.potential_function(all_V_states)
        for j in range(batch_states.shape[0]):
            u_idx = j // self.action_dim
            v_idx = closest_v_indices[j].item()
            r_hat = pred_rewards[j]
            phi_su = phi_V[u_idx]
            phi_sv = phi_V[v_idx]
            r_shaped = r_hat + self.config['gamma'] * phi_sv - phi_su
            cost = self.config['planning_C'] - r_shaped.item()
            graph[u_idx].append((v_idx, cost))

        # 2. Plan Generation
        start_node = len(V_nodes) - 1 # The current state is the last one we added

        # Find the node in the graph closest to the actual goal
        goal_dists = torch.linalg.norm(all_V_states - self.goal_state, dim=1)
        end_node = torch.argmin(goal_dists).item()

        # Run BMSSP to get path costs and predecessors
        costs, preds = bmss_p_cpp.bmss_p_with_preds(graph, start_node, max_depth=self.config['planning_H'])

        # 3. Reconstruct the plan
        action_plan = self._reconstruct_path_and_actions(start_node, end_node, preds, V_nodes)

        plan_found = 1 if action_plan else 0
        return action_plan, plan_found

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Add this new method to the NeuroSymbolicSSSP_DQN class
    def get_epsilon(self):
        return self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
            math.exp(-1. * self.steps_done / self.config['eps_decay'])

    # --- NEW: Add this new method to the class ---
    def get_avg_q_value(self, states):
        with torch.no_grad():
            avg_q = self.policy_net(states).max(1)[0].mean().item()
        return avg_q
