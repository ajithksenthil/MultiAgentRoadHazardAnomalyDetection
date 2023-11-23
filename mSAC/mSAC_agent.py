# mSAC_agent.py
import torch
import torch.optim as optim
from mSAC_models import Actor, Critic
import numpy as np

class Agent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99, alpha=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        
        # Ensure the target critic is initialized with the same weights
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha

    def select_action(self, state):
        # Convert numpy array to torch tensor
        state = torch.FloatTensor(state.reshape(1, -1))
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def update_parameters(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Convert batch to tensors, ensuring correct shapes and types
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Add batch dimension
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update Critic
        with torch.no_grad():
            next_actions, log_probs = self.actor.sample(next_states)
            q_target_next = self.target_critic(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * (q_target_next - self.alpha * log_probs)
        q_current = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(q_current, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        new_actions, log_probs = self.actor.sample(states)
        q_new_actions = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_probs - q_new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks with soft update
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Add any other necessary methods

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