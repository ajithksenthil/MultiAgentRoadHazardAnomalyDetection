
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from mSAC_models import Actor, Critic, MixingNetwork
import numpy as np

# Set default CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(1)  # Set default device in case of multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_critic_network():
    # Creating mock data
    state_dim = 20  # Assuming state dimension
    action_dim = 5  # Assuming action dimension
    hidden_dim = 256
    batch_size = 128

    mock_state = torch.randn(batch_size, state_dim)
    mock_action = torch.randn(batch_size, action_dim)
    hx = torch.zeros(hidden_dim)

    # Initialize critic network
    critic = Critic(state_dim, action_dim, hidden_dim)
    critic = critic.to(device)

    # Forward pass
    q_value, _ = critic(mock_state.to(device), mock_action.to(device), hx.to(device))

    # Compute loss and backward pass
    mock_target = torch.randn_like(q_value)
    loss = torch.nn.functional.mse_loss(q_value, mock_target)
    loss.backward()

    print("Critic network test completed without error.")



def test_actor_network():
    # Creating mock data
    state_dim = 20  # Assuming state dimension
    action_dim = 5  # Assuming action dimension
    hidden_dim = 256
    batch_size = 128

    mock_state = torch.randn(batch_size, state_dim)
    hx = torch.zeros(hidden_dim)

    # Initialize actor network
    actor = Actor(state_dim, action_dim, hidden_dim)
    actor = actor.to(device)

    # Sample action and compute log probabilities
    action, log_prob, _ = actor.sample(mock_state.to(device), hx.to(device))

    # Compute a dummy loss and perform backward pass
    dummy_loss = -log_prob.mean()
    dummy_loss.backward()

    print("Actor network test completed without error.")


def test_mixing_network():
    # Mock data
    num_agents = 5  # Number of agents
    state_dim = 20  # Dimension of the global state
    batch_size = 128  # Number of samples in a batch

    mock_states = torch.randn(batch_size, state_dim).to(device)
    mock_individual_qs = torch.randn(batch_size, num_agents, 1).to(device)

    # Initialize mixing network
    mixing_network = MixingNetwork(num_agents, state_dim)
    mixing_network.to(device)

    # Forward pass
    mixed_qs = mixing_network(mock_states, mock_individual_qs)

    # Dummy target for loss calculation
    mock_target = torch.randn_like(mixed_qs)
    
    # Loss and backward pass
    loss = F.mse_loss(mixed_qs, mock_target)
    loss.backward()

    print("Mixing network test completed without error.")



if __name__ == '__main__':
    test_critic_network()
    test_actor_network()
    test_mixing_network()
