import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from mSAC_models import Actor, Critic, MixingNetwork
from torch.autograd import gradcheck
import torchviz
"""
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)  # GRU layer for temporal dependencies
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, hx):
        print("actor forward start")
        print("hx version", hx._version)
        # Reshape state if it is 3D (batch, sequence, features)
        if state.dim() == 3:
            print("reshaping state")
            state = state.view(-1, state.size(-1))

        x = F.relu(self.fc1(state))
        hx_clone = hx.clone()  # Clone to avoid inplace modification
        print("hx_clone actor version", hx_clone._version)
        hx_updated = self.gru(x, hx_clone)
        print("hx_updated actor version", hx_updated._version)
        x = F.relu(self.fc2(hx_updated))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevents numerical instability
        print("actor forward end")
        return mu, log_std, hx_updated
    
    def sample(self, state, hx):
        # Debugging: Print the shape of state and hx
        # print(f"State shape: {state.shape}, HX shape: {hx.shape}")
        batch_size = state.size(0)  # Get the batch size from the state tensor
        print("hx version actor sample ", hx._version)
        if hx.size(0) != batch_size:
            hx = hx.expand(batch_size, -1).contiguous().clone()  # Adjust hx to match the batch size
        print("hx version actor sample ", hx._version)
        # Forward pass to get mean, log std, and updated hidden state
        mu, log_std, hx = self.forward(state, hx)
        print("hx version actor sample ", hx._version)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)  # Ensures action bounds
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), hx


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)  # GRU layer for temporal dependencies
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)


    def forward(self, state, action, hx):
        print("critic forward start")
        batch_size = state.size(0)  # Get the batch size from the state tensor
        hx = hx.expand(batch_size, -1).contiguous().clone()

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x)) 
        hx_clone = hx.clone()  # Clone to avoid inplace modification
        print("hx_clone version ", hx_clone._version) # version 0
        hx_updated = self.gru(x, hx_clone) # when I take away the GRU we still get the same error as before, with or without it
        print("hx_updated ", hx_updated._version) # version 2
        # x = x.detach() # how come adding this does not change the outcome, the error comes regardless
        x = F.relu(self.fc2(hx_updated))
        # x = x.detach() # how come when I add this it works but when I take it away same error
        print("Before forward in Critic - q_value version:", x._version)
        q_value = self.q_head(x)
        print("After forward in Critic - q_value version:", q_value._version)
        return q_value, hx_updated
    
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
        
        # Debugging: Print the shapes of tensors
        print(f"w1 Shape: {w1.shape}, w2 Shape: {w2.shape}, local_qs Shape: {local_qs.shape}")

        # Ensure that local_qs is 2D (batch, num_agents) if necessary
        if local_qs.dim() == 3:
            local_qs = local_qs.view(local_qs.size(0), -1)

        # Ensure w1 is correctly broadcastable with local_qs
        if w1.dim() != local_qs.dim():
            w1 = w1.unsqueeze(-1)

        mixed_qs = torch.einsum('bq,bq->b', w1, local_qs) + w2.sum(dim=1, keepdim=True)
        return mixed_qs
"""
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mock data
state_dim = 20  # Example state dimension
action_dim = 5  # Example action dimension
hidden_dim = 256
batch_size = 128
num_agents = 5  # Number of agents

states = torch.randn(batch_size, state_dim).to(device)
actions = torch.randn(batch_size, action_dim).to(device)
rewards = torch.randn(batch_size, 1).to(device)
next_states = torch.randn(batch_size, state_dim).to(device)
dones = torch.randint(0, 2, (batch_size, 1)).float().to(device)

# Initialize models
actors = [Actor(state_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
critics = [Critic(state_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
mixing_network = MixingNetwork(num_agents, state_dim).to(device)

# Initialize optimizers
actor_optimizers = [optim.Adam(actor.parameters(), lr=1e-4) for actor in actors]
critic_optimizers = [optim.Adam(critic.parameters(), lr=1e-3) for critic in critics]

# Update parameters function
def update_parameters(agent_idx):
    critic_hx = torch.zeros(batch_size, hidden_dim).to(device)  # Adjust the initial hidden state size
    current_Q_values, current_hx = critics[agent_idx](states, actions, critic_hx)
    print("current_Q_values, current_hx version", current_Q_values._version, current_hx._version)
    # current_Q_values = current_Q_values.detach() # how come detaching the current_Q_values makes it so it runs smoothly?

    actor_hx = torch.zeros(batch_size, hidden_dim).to(device)  # Adjust the initial hidden state size
    print("actor_hx version ", actor_hx._version)
    new_actions, log_probs, new_hx = actors[agent_idx](states, actor_hx)
    print("new_actions, log_probs, new_hx version ", new_actions._version, log_probs._version, new_hx._version)
    
    new_Q_values, _ = critics[agent_idx](states, new_actions, new_hx)
    print("new_Q_values version ", new_Q_values._version)
    
    # Calculate total Q-value using mixing network
    # current_Q_values = current_Q_values.detach() # so this when added makes the code run smoothly before it can reach the mixing network
    print("Before mixing network backward - current_Q_values version:", current_Q_values._version)
    total_Q = mixing_network(states, torch.stack([current_Q_values for _ in range(num_agents)], dim=1))
    print("after mixing network backward - current_Q_values version:", current_Q_values._version)
    current_Q_values = current_Q_values.detach() # so when this is added, the error still shows up regardless
    print("total_Q version", total_Q._version)
    # Calculate target Q-values using dummy data
    target_total_Q = rewards + (1 - dones) * total_Q.detach()
    print("target_total_Q version", target_total_Q._version)
    # Critic loss
    critic_loss = F.mse_loss(total_Q, target_total_Q)
    
    # Visualize the computation graph
    graph = torchviz.make_dot(critic_loss, params=dict(critics[agent_idx].named_parameters()))
    graph.render("critic_computation_graph", format="png")  # Saves the graph as a PNG file

    critic_optimizers[agent_idx].zero_grad()
    print("Before critic loss backward - current_Q_values version:", current_Q_values._version)
    critic_loss.backward()
    print("After critic loss backward - current_Q_values version:", current_Q_values._version)
    critic_optimizers[agent_idx].step()
    print("After critic loss backward: current_Q_values, total_Q, target_total_Q versions:", current_Q_values._version, total_Q._version, target_total_Q._version)

    # Actor loss
    # new_Q_values = new_Q_values.clone().detach()
    print("Before actor loss computation - new_Q_values version:", new_Q_values._version)
    actor_loss = -(log_probs.mean() + new_Q_values.mean())
    print("After actor loss computation - new_Q_values version:", new_Q_values._version)

    actor_optimizers[agent_idx].zero_grad()
    print("Before actor loss backward - new_Q_values version:", new_Q_values._version)
    actor_loss.backward()
    print("After actor loss backward - new_Q_values version:", new_Q_values._version)
    actor_optimizers[agent_idx].step()
    print("After actor loss backward: new_actions, log_probs, new_Q_values versions:", new_actions._version, log_probs._version, new_Q_values._version)

# Testing update parameters for one agent
torch.autograd.set_detect_anomaly(True)
update_parameters(0)
print("test complete")
