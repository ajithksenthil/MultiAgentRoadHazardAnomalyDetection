import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    def __init__(self, state_dim, num_agents):
        super(HyperNetwork, self).__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_agents)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_agents)
        )

    def forward(self, global_state):
        w1 = torch.abs(self.hyper_w1(global_state))
        w2 = torch.abs(self.hyper_w2(global_state))
        return w1, w2

class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_dim):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hyper_net = HyperNetwork(self.state_dim, self.num_agents)

    def forward(self, states, local_qs):
        w1, w2 = self.hyper_net(states)
        
        # Ensure that local_qs is 2D (batch, num_agents) if necessary
        if local_qs.dim() == 3:
            local_qs = local_qs.view(local_qs.size(0), -1)

        # Ensure w1 is correctly broadcastable with local_qs
        if w1.dim() != local_qs.dim():
            w1 = w1.unsqueeze(-1)

        mixed_qs = torch.einsum('bq,bq->b', w1, local_qs) + w2.sum(dim=1, keepdim=True)
        return mixed_qs

# Mock data for testing
state_dim = 20  # Example state dimension
num_agents = 5  # Number of agents
batch_size = 128

# Generate mock global state and local Q values
mock_global_state = torch.randn(batch_size, state_dim)
mock_local_qs = torch.randn(batch_size, num_agents, 1)  # Shape: [batch_size, num_agents, 1]

# Initialize HyperNetwork and MixingNetwork
hyper_net = HyperNetwork(state_dim, num_agents)
mixing_net = MixingNetwork(num_agents, state_dim)

# Test HyperNetwork
w1, w2 = hyper_net(mock_global_state)
print(f"Output shapes from HyperNetwork: w1 shape - {w1.shape}, w2 shape - {w2.shape}")

# Test MixingNetwork
mixed_qs = mixing_net(mock_global_state, mock_local_qs)
print(f"Output shape from MixingNetwork: mixed_qs shape - {mixed_qs.shape}")

