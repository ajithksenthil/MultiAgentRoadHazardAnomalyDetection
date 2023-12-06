# mSAC_agent.py
import torch
import torch.optim as optim
import torch.nn as nn
from mSAC_models import Actor, Critic, MixingNetwork
import numpy as np

# Set default CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(1)  # Set default device in case of multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_dim, action_dim, num_agents, batch_size, actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99, alpha=0.2, hidden_dim=256):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.actors = [nn.DataParallel(Actor(state_dim, action_dim, hidden_dim), device_ids=[1]).to(device) for _ in range(num_agents)]
        self.critics = [nn.DataParallel(Critic(state_dim, action_dim, hidden_dim), device_ids=[1]).to(device) for _ in range(num_agents)]
        self.target_critics = [nn.DataParallel(Critic(state_dim, action_dim, hidden_dim), device_ids=[1]).to(device) for _ in range(num_agents)]
        
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        self.mixing_network = MixingNetwork(num_agents, state_dim).to(device)
        self.target_mixing_network = MixingNetwork(num_agents, state_dim).to(device)
        self.mixing_network_optimizer = optim.Adam(self.mixing_network.parameters(), lr=critic_lr)

        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.num_agents = num_agents
        self.actor_hxs = [torch.zeros(batch_size, hidden_dim).to(device) for _ in range(num_agents)] # changing to batch size
        self.critic_hxs = [torch.zeros(batch_size, hidden_dim).to(device) for _ in range(num_agents)]
        # Initialize target critic hidden states
        self.target_critic_hxs = [torch.zeros(batch_size, hidden_dim).to(device) for _ in range(num_agents)]

        # Clear the cache after initializing all components
        torch.cuda.empty_cache()

    
    def select_action(self, state, agent_idx):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        batch_size = state_tensor.size(0)

        hx = self.actor_hxs[agent_idx]
        if hx.size(0) != batch_size:
            hx = torch.zeros(batch_size, self.hidden_dim).to(device)

        raw_action, log_prob, new_hx = self.actors[agent_idx].module.sample(state_tensor, hx)
        self.actor_hxs[agent_idx] = new_hx.detach()  # Update the hidden state

        # Remove the batch dimension if it's present
        if raw_action.dim() > 1:
            raw_action = raw_action.squeeze(0)

        # Check the shape of the raw_action tensor
        expected_shape = (5,)  # Adjust based on your expected action dimensions
        if raw_action.shape != expected_shape:
            raise ValueError(f"Unexpected shape of raw_action: {raw_action.shape}, expected: {expected_shape}")

        # Process the action
        action = {
            'vehicle_control': {
                'throttle': raw_action[0].item(),
                'steer': raw_action[1].item(),
                'brake': raw_action[2].item()
            },
            'traffic_manager_control': {
                'lane_change': raw_action[3].item(),
                'speed_adjustment': raw_action[4].item()
            }
        }

        return action, log_prob

    def update_parameters(self, batch, agent_idx):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)

        critic_hx = self.critic_hxs[agent_idx].clone().detach()
        current_Q_values, current_hx = self.critics[agent_idx](states, actions, critic_hx)

        actor_hx = self.actor_hxs[agent_idx].clone().detach()
        new_actions, log_probs, new_hx = self.actors[agent_idx].module.sample(states, actor_hx)

        new_Q_values, _ = self.critics[agent_idx](states, new_actions, new_hx)

        # Calculate total Q-value using mixing network
        total_Q = self.mixing_network(states, torch.stack([critic(states, actions, hx)[0] for critic, hx in zip(self.critics, self.critic_hxs)], dim=1))

        # Calculate target Q-values using target critic networks
        with torch.no_grad():
            target_qs = []
            for idx in range(self.num_agents):
                target_critic_hx = self.target_critic_hxs[idx].clone().detach()
                target_Q, _ = self.target_critics[idx](next_states, new_actions, target_critic_hx)
                target_qs.append(target_Q)

            target_qs_stacked = torch.stack(target_qs, dim=1)
            target_total_Q = self.target_mixing_network(next_states, target_qs_stacked)
            target_total_Q = rewards + self.gamma * (1 - dones) * target_total_Q

        # Critic loss
        critic_loss = torch.nn.functional.mse_loss(total_Q, target_total_Q)
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[agent_idx].step()

        # Actor loss
        actor_loss = -(self.alpha * log_probs + new_Q_values).mean()
        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_idx].step()

        # Update hidden states and target networks
        self.critic_hxs[agent_idx] = current_hx.clone().detach()
        self.actor_hxs[agent_idx] = new_hx.clone().detach()
        self.soft_update_target_networks()



    def soft_update_target_networks(self):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update target mixing network
        for target_param, param in zip(self.target_mixing_network.parameters(), self.mixing_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    # Reset hidden states at the beginning of each episode
    def reset_hxs(self):
        self.actor_hxs = [torch.zeros(self.batch_size, self.hidden_dim).to(device) for _ in range(self.num_agents)]
        self.critic_hxs = [torch.zeros(self.batch_size, self.hidden_dim).to(device) for _ in range(self.num_agents)]


