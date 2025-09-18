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

    '''def planning_phase(self):
        # Pseudocode: PLANNING_PHASE
        if len(self.memory) < self.config['planning_N']: return

        print("--- Starting Planning Phase ---") # Add this
        for x in range(self.config['planning_k']):
            print(f"  Planning iteration {x+1}/{self.config['planning_k']}...") # Add this
            # 1. Graph Extraction
            sampled_transitions = self.memory.sample(self.config['planning_N'])
            V_states = [t.state for t in sampled_transitions]
            V_nodes = {i: state for i, state in enumerate(V_states)}

            graph = defaultdict(list)

            for u_idx, s_u_tensor in V_nodes.items():
                for a_u in range(self.action_dim):
                    a_u_tensor = torch.tensor([a_u], device=self.device)

                    with torch.no_grad():
                        s_prime_hat_tensor, r_hat_tensor = self.world_model(s_u_tensor, a_u_tensor)

                    # Find closest state in graph to form edge
                    all_states_tensor = torch.cat(list(V_nodes.values()))
                    dists = torch.linalg.norm(all_states_tensor - s_prime_hat_tensor, dim=1)
                    v_idx = torch.argmin(dists).item()
                    s_v_tensor = V_nodes[v_idx]

                    # 2. Reward Shaping and Cost Transformation
                    phi_su = self.potential_function(s_u_tensor)
                    phi_sv = self.potential_function(s_v_tensor)

                    r_shaped = r_hat_tensor + self.config['gamma'] * phi_sv - phi_su

                    # Ensure cost is non-negative
                    cost = self.config['planning_C'] - r_shaped.item()
                    graph[u_idx].append((v_idx, cost))

            # 3. Short-Horizon SSSP and Q-Network Update
            (s_p, a_p, _, _) = self.memory.sample(1)[0]

            with torch.no_grad():
                r_p_hat, s_p_prime_hat = self.world_model(s_p, a_p.squeeze(0))

            # Find start and end nodes in the graph
            all_states_tensor = torch.cat(list(V_nodes.values()))
            start_dists = torch.linalg.norm(all_states_tensor - s_p, dim=1)
            start_node = torch.argmin(start_dists).item()

            end_dists = torch.linalg.norm(all_states_tensor - s_p_prime_hat, dim=1)
            end_node = torch.argmin(end_dists).item()

            # Run BMSSP
            costs = bmss_p_cpp.bmss_p(graph, start_node, max_depth=self.config['planning_H'])

            # --- ADD THIS DEBUGGING LINE ---
            print(f"DEBUG: Type of costs is {type(costs)}, value is {costs}")
            # --------------------------------

            # --- ADD TIMING AROUND THE C++ CALL ---
            print(f"    Graph built. Calling C++ BMSSP...")
            start_time = time.time()
            costs = bmss_p_cpp.bmss_p(graph, start_node, max_depth=self.config['planning_H'])
            end_time = time.time()
            print(f"    C++ BMSSP finished in {end_time - start_time:.4f} seconds.")
                        # ----------------------------------------

            optimal_cost_to_go = costs.get(end_node, float('inf'))

            if optimal_cost_to_go != float('inf'):
                # Convert cost back to value
                V_star = -optimal_cost_to_go

                # Update Q-network with the planner's target
                planning_target = r_p_hat + self.config['gamma'] * V_star

                current_q = self.policy_net(s_p).gather(1, a_p)
                loss = F.mse_loss(current_q, planning_target.unsqueeze(0))

                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()'''
                # In sssp_dqn.py

    # In sssp_dqn.py

    def planning_phase(self, current_step, max_steps):
        if len(self.memory) < self.config['planning_N']: return 0

        #print(f"\n--- [Step {current_step+1}/{max_steps}] Starting Planning Phase ---")
        successful_plans = 0

        for i in range(self.config['planning_k']):
            t_start_iter = time.time()
            #print(f"  Planning iteration {i+1}/{self.config['planning_k']}...")

            # 1. Graph Extraction (Batched Version)
            sampled_transitions = self.memory.sample(self.config['planning_N'])
            V_states_list = [t.state for t in sampled_transitions]
            V_nodes = {i: state for i, state in enumerate(V_states_list)}
            all_V_states = torch.cat(V_states_list, dim=0)

            # Batch Preparation
            batch_states = all_V_states.repeat_interleave(self.action_dim, dim=0)
            batch_actions = torch.arange(self.action_dim, device=self.device).repeat(self.config['planning_N'])

            # Batch Prediction
            with torch.no_grad():
                pred_next_states, pred_rewards = self.world_model(batch_states, batch_actions)

            # Batch Nearest-Neighbor Search
            dists = torch.cdist(pred_next_states, all_V_states)
            closest_v_indices = torch.argmin(dists, dim=1)

            # Graph Construction
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

            # 2. SSSP and Q-Network Update
            (s_p, a_p, _, _) = self.memory.sample(1)[0]

            with torch.no_grad():
                s_p_prime_hat, r_p_hat = self.world_model(s_p, a_p.squeeze(0))

            start_dists = torch.linalg.norm(all_V_states - s_p.squeeze(0), dim=1)
            start_node = torch.argmin(start_dists).item()

            end_dists = torch.linalg.norm(all_V_states - s_p_prime_hat.squeeze(0), dim=1)
            end_node = torch.argmin(end_dists).item()

            costs = bmss_p_cpp.bmss_p(graph, start_node, max_depth=self.config['planning_H'])
            optimal_cost_to_go = costs.get(end_node, float('inf'))

            plan_found = optimal_cost_to_go != float('inf')
            #print(f"    [Iter {i+1}] Plan from node {start_node} to {end_node}. Path found: {plan_found}")

            if plan_found:
                successful_plans += 1

                # --- THIS IS THE Q-UPDATE LOGIC ---
                # Convert the planner's path cost back into a value estimate
                V_star = -optimal_cost_to_go

                # Create the high-quality learning target
                planning_target = r_p_hat + self.config['gamma'] * V_star

                # Get the DQN's current prediction for comparison
                current_q = self.policy_net(s_p).gather(1, a_p)

                # Calculate the loss and update the DQN's weights
                loss = F.mse_loss(current_q, planning_target)

                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                # ------------------------------------

            t_end_iter = time.time()
            #print(f"  Iteration finished in {t_end_iter - t_start_iter:.4f}s.")

        return successful_plans

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
