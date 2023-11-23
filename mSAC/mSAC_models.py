# mSAC_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevents numerical instability
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)  # Ensures action bounds
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_head(x)
        return q_value

# Add any other necessary classes or functions

'''
Modifications for CARLA Integration:
- Action Space: The action space for driving tasks can be complex (e.g., steering, acceleration, braking). Ensure that the action_dim is set appropriately.
- State Space: The state input should incorporate relevant features like vehicle dynamics, sensor readings, and environmental cues.
- Network Complexity: Depending on the complexity of the driving scenario, you may need to adjust the hidden_dim or add more layers to the networks.
- Non-linearity: ReLU is used for simplicity. You may experiment with other activation functions based on the specific requirements of your tasks.
- Stability: The clamping of log_std is crucial for preventing numerical issues during training.
- Bounded Actions: The tanh function ensures that the actions are bounded, which is necessary for controlling vehicle movements.

Considerations:
- Ensure the Actor model generates actions suitable for the driving tasks in CARLA (e.g., steering angles, throttle values).
- The Critic model should effectively estimate the Q-values given the states and actions in the driving context.
- You might need to incorporate additional sensory inputs or state information relevant to your specific CARLA simulation setup.
'''


'''
Actor Network
Architecture: Two fully connected layers followed by separate heads for mean (mu_head) and log standard deviation (log_std_head) of action distributions.
Action Sampling: The sample method uses the reparameterization trick for stochastic policy. It samples actions from the normal distribution and applies the tanh function for bounded actions.
Stability: log_std is clamped to prevent numerical instability.
Critic Network
Architecture: Consists of fully connected layers that combine state and action inputs to estimate the Q-value.
Forward Pass: The state and action are concatenated and passed through the network to output the Q-value.
Considerations
Hyperparameters: hidden_dim can be adjusted according to the complexity of the problem.
Activation Functions: ReLU is used here, but you can experiment with others like LeakyReLU or ELU.
Normalization: Consider using batch normalization or layer normalization if you face issues with training stability.
This setup establishes a basic framework for the actor and critic models in mSAC. You might need to adapt the architecture and parameters based on the specific requirements and challenges of your environment and tasks.
'''