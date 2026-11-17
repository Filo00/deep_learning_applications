import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """MLP che mappa uno stato nei logits delle azioni (la softmax e' dentro Categorical)."""
    def __init__(self, env, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, env.action_space.n),
        )

    def forward(self, state):
        return self.net(state)


def select_action(policy, state):
    """Campiona un'azione dalla policy. Restituisce (azione, log-probabilita' dell'azione)."""
    dist = Categorical(logits=policy(torch.as_tensor(state, dtype=torch.float32)))
    action = dist.sample()
    return action.item(), dist.log_prob(action)


class ValueNet(nn.Module):
    def __init__(self, env, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)