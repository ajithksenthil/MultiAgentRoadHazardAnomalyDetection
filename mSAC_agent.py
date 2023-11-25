# mSAC_agent.py
import torch
import torch.optim as optim
from mSAC_models import Actor, Critic, MixingNetwork
import numpy as np

class Agent:
    def __init__(self, state_dim, action_dim, num_agents, actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99, alpha=0.2, hidden_dim=256):
        self.hidden_dim = hidden_dim  # Define the hidden_dim for the entire class
        self.actors = [Actor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        
        # Ensure target critics are initialized with the same weights
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        self.mixing_network = MixingNetwork(num_agents, state_dim, hidden_dim)
        self.target_mixing_network = MixingNetwork(num_agents, state_dim, hidden_dim)
        self.mixing_network_optimizer = optim.Adam(self.mixing_network.parameters(), lr=critic_lr)

        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.num_agents = num_agents
        
        # Hidden states for GRUs in actors and critics
        self.actor_hxs = [torch.zeros(1, hidden_dim) for _ in range(num_agents)]
        self.critic_hxs = [torch.zeros(1, hidden_dim) for _ in range(num_agents)]

    def select_actions(self, states):
        actions = []
        log_probs = []
        for idx, (actor, state) in enumerate(zip(self.actors, states)):
            state = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, hx = actor.sample(state, self.actor_hxs[idx])
            actions.append(action.detach().cpu().numpy().flatten())
            log_probs.append(log_prob.detach())
            self.actor_hxs[idx] = hx.detach()  # Detach hx to prevent backpropagating through it
        return actions, log_probs

    def update_parameters(self, batch, agent_idx):
        states, actions, rewards, next_states, dones = batch

        # Convert batch to tensors, ensuring correct shapes and types
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Add batch dimension
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

         # Update Critic and Mixing Network with hidden states
        current_Q_values, current_hx = self.critics[agent_idx](states, actions, self.critic_hxs[agent_idx])
        new_actions, log_probs, new_hx = self.actors[agent_idx].sample(states, self.actor_hxs[agent_idx])
        new_Q_values, _ = self.critics[agent_idx](states, new_actions, new_hx)
        
        # Update the hidden states for target critics
        with torch.no_grad():
            target_Q_values = torch.zeros_like(current_Q_values)
            target_hxs = [torch.zeros(1, self.hidden_dim) for _ in range(self.num_agents)]
            for idx in range(self.num_agents):
                target_actions, _, target_hx = self.target_actors[idx].sample(next_states, target_hxs[idx])
                target_Q_values += self.target_critics[idx](next_states, target_actions, target_hx)
                target_hxs[idx] = target_hx

            target_Q_values = rewards + self.gamma * (1 - dones) * target_Q_values


        # Compute critic loss and update critic
        critic_loss = torch.nn.functional.mse_loss(current_Q_values, target_Q_values.detach())
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[agent_idx].step()

        # Update actor network with hidden states
        actor_loss = -new_Q_values.mean()
        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_idx].step()

        # Update hidden states for the current critic
        self.critic_hxs[agent_idx] = current_hx.detach()

        
        # Update mixing network
        total_Q = self.mixing_network(states, torch.stack([critic.q_value for critic in self.critics], dim=1))
        mixing_loss = torch.nn.functional.mse_loss(total_Q, target_Q_values.detach())
        self.mixing_network_optimizer.zero_grad()
        mixing_loss.backward()
        self.mixing_network_optimizer.step()

        # Soft update the target networks
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.target_mixing_network.soft_update(self.mixing_network, self.tau)

    # Reset hidden states at the beginning of each episode
    def reset_hxs(self):
        self.actor_hxs = [torch.zeros(1, self.hidden_dim) for _ in range(self.num_agents)]
        self.critic_hxs = [torch.zeros(1, self.hidden_dim) for _ in range(self.num_agents)]

    # Additional methods as necessary

'''
Enhancements:
- Mixing Network: A new class `MixingNetwork` is used to combine the Q-values from each critic. It needs to be defined in `mSAC_models.py`.
- Multiple Actors and Critics: The `Agent` class now manages a list of actors and critics for each agent in the environment.
- Update Mechanism: The update method now includes logic for the mixing network alongside the individual actor and critic updates.
- CARLA Integration: Adjustments are made to align the agents with the CARLA environment, such as action selection and state representation.
'''

# Add the MixingNetwork class implementation in mSAC_models.py

'''
Enhancements:
- Error Handling: Add try-except blocks where appropriate to handle unexpected errors.
- Tensor Compatibility: Ensure that tensor shapes are compatible with network inputs and outputs.
- Performance Monitoring: Add logging for rewards, losses, etc., to track agent performance over time.
- CARLA Integration: Adjust the state and action dimensions to match those provided by CARLA, and ensure that actions selected by the agent can be executed by CARLA vehicles.
'''


'''
Key Components:
Initialization: Set up the actor and critic networks, their target networks, and optimizers.
Action Selection: Use the actor network to select actions. Actions are sampled from the policy distribution to ensure exploration.
Parameter Update: Update the parameters of the actor and critic networks using sampled experiences from the replay buffer. Implement soft updates for the target network.
Hyperparameters: Learning rates, discount factor (
γ
γ), soft update coefficient (
τ
τ), and entropy coefficient (
α
α) are crucial for tuning the performance of SAC.
Integration with CARLA:
The state_dim and action_dim should correspond to the dimensions of the state and action spaces as defined by the CARLA environment and the Traffic Manager's requirements.
The action selection and update mechanisms must be compatible with the real-time demands of driving scenarios in CARLA.
Additional Considerations:
Error Handling: Include error handling and validation checks where necessary.
Compatibility: Ensure that the tensors' shapes are compatible with the inputs and outputs of the actor and critic networks.
Performance Monitoring: Consider adding mechanisms for logging and monitoring the performance of the agent.

'''