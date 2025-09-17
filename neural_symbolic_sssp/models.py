# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """A simple MLP Q-Network."""
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
    """
    A simple MLP world model that predicts next state and reward.
    """
    def __init__(self, state_dim, action_dim):
        super(WorldModel, self).__init__()
        # Input is state + one-hot encoded action
        self.layer1 = nn.Linear(state_dim + action_dim, 128)
        self.layer2 = nn.Linear(128, 128)

        # Output heads for next state and reward
        self.state_head = nn.Linear(128, state_dim)
        self.reward_head = nn.Linear(128, 1)

    def forward(self, state, action):
        # One-hot encode action
        action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat([state, action_one_hot], dim=1)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        next_state = self.state_head(x)
        reward = self.reward_head(x)
        return next_state, reward

    @property
    def action_dim(self):
        # A bit of a hack to get action_dim from the input layer size
        return self.layer1.in_features - self.state_head.out_features
