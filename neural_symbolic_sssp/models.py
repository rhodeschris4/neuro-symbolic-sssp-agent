# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    # --- NO CHANGES NEEDED FOR DQN ---
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class WorldModel(nn.Module):
    # --- MODIFIED FOR GENERALIZATION ---
    def __init__(self, state_dim, action_dim):
        super(WorldModel, self).__init__()
        # state_dim is now 4. The input is (agent_pos, goal_pos)
        self.state_dim = state_dim
        self.layer1 = nn.Linear(state_dim + action_dim, 128)
        self.layer2 = nn.Linear(128, 128)

        # The state head now only predicts the agent's next position (2 dims)
        self.state_head = nn.Linear(128, 2)
        self.reward_head = nn.Linear(128, 1)

    def forward(self, state, action):
        action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        # Ensure state is 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.cat([state, action_one_hot], dim=1)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        # Predict only the agent's next position
        next_agent_pos = self.state_head(x)

        # The goal position doesn't change, so we concatenate it
        goal_pos = state[:, 2:]
        next_state = torch.cat([next_agent_pos, goal_pos], dim=1)

        reward = self.reward_head(x)
        return next_state, reward

    @property
    def action_dim(self):
        return self.layer1.in_features - self.state_dim
