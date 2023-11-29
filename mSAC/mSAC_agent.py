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
    def __init__(self, state_dim, action_dim, num_agents, actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99, alpha=0.2, hidden_dim=256):
        self.hidden_dim = hidden_dim
        self.actors = [nn.DataParallel(Actor(state_dim, action_dim, hidden_dim)).to(device) for _ in range(num_agents)]
        self.critics = [nn.DataParallel(Critic(state_dim, action_dim, hidden_dim)).to(device) for _ in range(num_agents)]
        self.target_critics = [nn.DataParallel(Critic(state_dim, action_dim, hidden_dim)).to(device) for _ in range(num_agents)]
        
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        self.mixing_network = MixingNetwork(num_agents, state_dim)
        self.target_mixing_network = MixingNetwork(num_agents, state_dim)
        self.mixing_network_optimizer = optim.Adam(self.mixing_network.parameters(), lr=critic_lr)

        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.num_agents = num_agents
        self.actor_hxs = [torch.zeros(1, hidden_dim).to(device) for _ in range(num_agents)]
        self.critic_hxs = [torch.zeros(1, hidden_dim).to(device) for _ in range(num_agents)]

        # Clear the cache after initializing all components
        torch.cuda.empty_cache()

    def select_action(self, state, agent_idx):
        # Convert state to tensor and sample action from the actor network
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)

        # Access the 'module' attribute to call 'sample'
        raw_action, log_prob, hx = self.actors[agent_idx].module.sample(state_tensor, self.actor_hxs[agent_idx])
        self.actor_hxs[agent_idx] = hx.detach()

        # Assuming raw_action contains values for vehicle control (throttle, steer, brake)
        # and traffic manager control (lane_change, speed_adjustment)
        # The exact structure and indexing depend on how your network outputs the action
        action = {
            'vehicle_control': {
                'throttle': raw_action[0].item(),  # Assuming first element is throttle
                'steer': raw_action[1].item(),     # Assuming second element is steering
                'brake': raw_action[2].item()      # Assuming third element is braking
            },
            'traffic_manager_control': {
                'lane_change': raw_action[3].item(),       # Assuming fourth element is lane change decision
                'speed_adjustment': raw_action[4].item()   # Assuming fifth element is speed adjustment
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

        # Get current Q-values from the critic network for current state-action pairs
        current_Q_values, current_hx = self.critics[agent_idx](states, actions, self.critic_hxs[agent_idx])
        
        # Sample new actions and their log probabilities from the actor network
        new_actions, log_probs, new_hx = self.actors[agent_idx].sample(states, self.actor_hxs[agent_idx])
        
        # Get new Q-values from the critic network for the new state-action pairs
        new_Q_values, _ = self.critics[agent_idx](states, new_actions, new_hx)

        # Calculate total Q-value using the mixing network
        total_Q = self.mixing_network(states, torch.stack([critic(states, actions, hx)[0] for critic, hx in zip(self.critics, self.critic_hxs)], dim=1))
        
        # Calculate the target Q-values using target critic networks
        target_total_Q = torch.zeros_like(total_Q)
        with torch.no_grad():
            for idx in range(self.num_agents):
                target_actions, _, target_hx = self.target_critics[idx].sample(next_states, self.target_critic_hxs[idx])
                target_Q, _ = self.target_critics[idx](next_states, target_actions, target_hx)
                target_total_Q += target_Q
            target_total_Q = rewards + self.gamma * (1 - dones) * target_total_Q

        # Critic loss is the mean squared TD error
        critic_loss = torch.nn.functional.mse_loss(current_Q_values, target_total_Q.detach())
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[agent_idx].step()

        # Update the actor using the policy gradient
        actor_loss = -(self.alpha * log_probs + new_Q_values).mean()
        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_idx].step()

        # Update hidden states
        self.critic_hxs[agent_idx] = current_hx.detach()
        self.actor_hxs[agent_idx] = new_hx.detach()

        # Soft update of target networks
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update the target mixing network
        self.target_mixing_network.soft_update()


    # Reset hidden states at the beginning of each episode
    def reset_hxs(self):
        self.actor_hxs = [torch.zeros(1, self.hidden_dim).to(device) for _ in range(self.num_agents)]
        self.critic_hxs = [torch.zeros(1, self.hidden_dim).to(device) for _ in range(self.num_agents)]

    # Additional methods as necessary

