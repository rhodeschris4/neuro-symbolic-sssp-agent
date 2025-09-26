import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict, deque
import math
import time
from neural_symbolic_sssp.models import DQN, WorldModel
from neural_symbolic_sssp.utils import ReplayBuffer, Transition
import bmss_p_cpp

class NeuroSymbolicSSSP_DQN:
    def __init__(self, state_dim, action_dim, config, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device

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

    def optimize_model(self, planning_loss=None):
        """
        UPDATED: This is now the central optimization function for the DQN.
        It combines the standard TD loss with an optional planning loss.
        """
        if len(self.memory) < self.config['batch_size']:
            return None

        # --- 1. Calculate Standard TD Loss from experience replay ---
        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        next_state_batch = torch.cat(batch.next_state)

        q_values = self.policy_net(state_batch).gather(1, action_batch)


        with torch.no_grad():
            best_next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
            q_values_next_target = self.target_net(next_state_batch)
            next_q_values = q_values_next_target.gather(1, best_next_actions).squeeze(1)
            expected_q_values = (next_q_values * self.config['gamma']) + reward_batch

        td_loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

        # --- 2. Combine with Planning Loss if available ---
        total_loss = td_loss
        if planning_loss is not None:
            planning_weight = self.config.get('planning_loss_weight', 1.0)
            total_loss = total_loss + planning_weight * planning_loss

        # --- 3. Perform a single optimization step ---

        best_next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
        q_values_next_target = self.target_net(next_state_batch)
        next_q_values = q_values_next_target.gather(1, best_next_actions).squeeze(1).detach()
        non_terminal_mask = 1.0 - done_batch
        expected_q_values = (next_q_values * self.config['gamma'] * non_terminal_mask) + reward_batch
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()

        return total_loss.item()

    def update_world_model(self):
        if len(self.memory) < self.config['batch_size']: return None
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

    def potential_function(self, state):
        agent_pos = state[:, :2]
        goal_pos = state[:, 2:]
        return -torch.sum(torch.abs(agent_pos - goal_pos), dim=-1)

    def planning_update(self):
        """
        UPDATED: This function no longer updates the network directly.
        Instead, it calculates and returns the planning loss tensor.
        """
        if len(self.memory) < self.config['planning_N']:
            return None, 0

        (s_p, a_p, _, _) = self.memory.sample(1)[0]

        with torch.no_grad():
            s_p_prime_hat, r_p_hat = self.world_model(s_p, a_p.squeeze(0))

        graph = defaultdict(list)
        state_map = {tuple(s_p_prime_hat.cpu().numpy().flatten()): (0, s_p_prime_hat)}
        node_id_counter = 1
        current_layer_states = [s_p_prime_hat]

        for _ in range(self.config['planning_H']):
            if not current_layer_states or len(state_map) >= self.config['planning_N']:
                break

            layer_state_tensors = torch.cat(current_layer_states, dim=0)
            num_states_in_layer = layer_state_tensors.size(0)
            batch_states = layer_state_tensors.repeat_interleave(self.action_dim, dim=0)
            batch_actions = torch.arange(self.action_dim, device=self.device).repeat(num_states_in_layer)

            with torch.no_grad():
                pred_next_states, pred_rewards = self.world_model(batch_states, batch_actions)

            next_layer_states = []
            for i in range(batch_states.size(0)):
                u_state_tuple = tuple(batch_states[i].cpu().numpy().flatten())
                u_idx, u_state_tensor = state_map[u_state_tuple]
                v_state_tensor = pred_next_states[i].unsqueeze(0)
                v_state_tuple = tuple(v_state_tensor.cpu().numpy().flatten())

                if v_state_tuple not in state_map:
                    if len(state_map) >= self.config['planning_N']: continue
                    v_idx = node_id_counter
                    state_map[v_state_tuple] = (v_idx, v_state_tensor)
                    node_id_counter += 1
                    next_layer_states.append(v_state_tensor)
                else:
                    v_idx, _ = state_map[v_state_tuple]

                r_hat = pred_rewards[i]
                phi_su = self.potential_function(u_state_tensor)
                phi_sv = self.potential_function(v_state_tensor)
                r_shaped = r_hat + self.config['gamma'] * phi_sv - phi_su
                cost = self.config['planning_C'] - r_shaped.item()
                graph[u_idx].append((v_idx, cost))
            current_layer_states = next_layer_states

        start_node = 0
        costs = bmss_p_cpp.bmss_p(graph, start_node, max_depth=self.config['planning_H'])

        if not costs: return None, 0

        optimal_cost_to_go = min(costs.values())
        V_star_shaped = -optimal_cost_to_go
        phi_s_p_prime = self.potential_function(s_p_prime_hat).item()
        q_target = r_p_hat.item() + self.config['gamma'] * (V_star_shaped + phi_s_p_prime)

        current_q_val = self.policy_net(s_p).gather(1, a_p)
        target_tensor = torch.tensor([[q_target]], device=self.device, dtype=torch.float32)

        planning_loss = F.mse_loss(current_q_val, target_tensor)

        return planning_loss, 1

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_epsilon(self):
        return self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
            math.exp(-1. * self.steps_done / self.config['eps_decay'])
