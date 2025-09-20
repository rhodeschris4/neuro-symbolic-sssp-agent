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
import bmss_p_cpp

class NeuroSymbolicSSSP_DQN:
    def __init__(self, state_dim, action_dim, config, device): # goal_state removed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device

        # ... (network and optimizer initialization remains the same) ...
        policy_net_instance = DQN(state_dim, action_dim).to(self.device)
        target_net_instance = DQN(state_dim, action_dim).to(self.device)
        world_model_instance = WorldModel(state_dim, action_dim).to(self.device)
        self.policy_net = torch.jit.script(policy_net_instance)
        self.target_net = torch.jit.script(target_net_instance)
        self.world_model = torch.jit.script(world_model_instance)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config['lr_dqn'])
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=config['lr_wm'])
        self.memory = ReplayBuffer(config['replay_capacity'])
        self.steps_done = 0

    # ... (select_action, update_direct_rl, update_world_model remain the same) ...
    def select_action(self, state, evaluate=False):
        if evaluate:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        sample = random.random()
        eps_threshold = self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
            math.exp(-1. * self.steps_done / self.config['eps_decay'])
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def update_direct_rl(self):
        if len(self.memory) < self.config['batch_size']: return None
        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        best_next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
        q_values_next_target = self.target_net(next_state_batch)
        next_q_values = q_values_next_target.gather(1, best_next_actions).squeeze(1).detach()
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
        return loss.item()

    # --- MODIFIED FOR GENERALIZATION ---
    def potential_function(self, state):
        """Calculates potential based on Manhattan distance to the goal in the state."""
        agent_pos = state[:, :2]
        goal_pos = state[:, 2:]
        return -torch.sum(torch.abs(agent_pos - goal_pos), dim=-1)

    # --- plan_action_sequence is also MODIFIED ---
    def plan_action_sequence(self, current_state_tensor):
        if len(self.memory) < self.config['planning_N']:
            return [], 0

        # Extract the dynamic goal from the current state
        dynamic_goal_state = current_state_tensor[:, 2:].squeeze(0)

        # 1. Graph Extraction
        sampled_transitions = self.memory.sample(self.config['planning_N'])
        # We only care about the agent's position for planning nodes
        V_agent_pos_list = [t.state[:, :2] for t in sampled_transitions] + [current_state_tensor[:, :2]]
        V_nodes = {i: pos for i, pos in enumerate(V_agent_pos_list)}
        all_V_agent_pos = torch.cat(V_agent_pos_list, dim=0)

        # To use the world model, we must reconstruct a full 4D state for prediction
        # We'll assume the goal for all sampled states was the current dynamic_goal_state
        reconstructed_states = torch.cat([all_V_agent_pos, dynamic_goal_state.repeat(len(V_nodes), 1)], dim=1)

        batch_states = reconstructed_states.repeat_interleave(self.action_dim, dim=0)
        batch_actions = torch.arange(self.action_dim, device=self.device).repeat(len(V_nodes))

        with torch.no_grad():
            pred_next_states, pred_rewards = self.world_model(batch_states, batch_actions)

        # We only need the agent position part of the predicted next states for graph building
        pred_next_agent_pos = pred_next_states[:, :2]

        dists = torch.cdist(pred_next_agent_pos, all_V_agent_pos)
        closest_v_indices = torch.argmin(dists, dim=1)

        graph = defaultdict(list)
        # The potential function here needs a full 4D state
        phi_V = self.potential_function(reconstructed_states)
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
        start_node = len(V_nodes) - 1
        goal_dists = torch.linalg.norm(all_V_agent_pos - dynamic_goal_state, dim=1)
        end_node = torch.argmin(goal_dists).item()

        costs, preds = bmss_p_cpp.bmss_p_with_preds(graph, start_node, max_depth=self.config['planning_H'])

        action_plan = self._reconstruct_path_and_actions(start_node, end_node, preds, V_nodes)

        return action_plan, 1 if action_plan else 0

    # ... (The rest of the class, including _reconstruct_path_and_actions, update_target_net, get_epsilon, remains unchanged) ...
    def _reconstruct_path_and_actions(self, start_node, end_node, preds, V_nodes):
        if end_node not in preds and end_node != start_node:
            return []
        path_nodes = []
        curr = end_node
        while curr != start_node:
            path_nodes.append(curr)
            if curr not in preds: return []
            curr = preds[curr]
        path_nodes.append(start_node)
        path_nodes.reverse()
        actions = []
        for i in range(len(path_nodes) - 1):
            pos_curr = V_nodes[path_nodes[i]].cpu().numpy().flatten()
            pos_next = V_nodes[path_nodes[i+1]].cpu().numpy().flatten()
            delta_r, delta_c = pos_next[0] - pos_curr[0], pos_next[1] - pos_curr[1]
            if delta_r == -1 and delta_c == 0: actions.append(0)
            elif delta_r == 1 and delta_c == 0: actions.append(1)
            elif delta_r == 0 and delta_c == 1: actions.append(2)
            elif delta_r == 0 and delta_c == -1: actions.append(3)
            else: return actions
        return actions

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_epsilon(self):
        return self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
            math.exp(-1. * self.steps_done / self.config['eps_decay'])
