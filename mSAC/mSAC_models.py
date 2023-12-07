# mSAC_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # x = x.clone() # this does not help either
        x = F.relu(self.fc2(hx_updated))
        # x = x.clone() # this does not help
        # x = x.detach() # how come when I add this it works but when I take it away same error
        print("Before forward in Critic - q_value version:", x._version)
        q_value = self.q_head(x)
        print("After forward in Critic - q_value version:", q_value._version)
        return q_value, hx_updated
    
    """
    /Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/autograd/__init__.py:200: UserWarning: Error detected in AddmmBackward0. Traceback of forward call that caused the error:
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/test_critic_loss_new.py", line 70, in <module>
    update_parameters(0)
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/test_critic_loss_new.py", line 42, in update_parameters
    new_Q_values, _ = critics[agent_idx](states, new_actions, new_hx)
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/mSAC_models.py", line 81, in forward
    q_value = self.q_head(x)
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
 (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/python_anomaly_mode.cpp:119.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Traceback (most recent call last):
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/test_critic_loss_new.py", line 70, in <module>
    update_parameters(0)
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/test_critic_loss_new.py", line 64, in update_parameters
    actor_loss.backward()
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/autograd/__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [256, 1]], which is output 0 of AsStridedBackward0, is at version 2; expected version 1 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
    """



    """ 
    def forward(self, state, action, hx):
        batch_size = state.size(0)  # Get the batch size from the state tensor
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))

        # Clone hx to avoid inplace modification
        hx_clone = hx.clone()
        hx_updated = self.gru(x, hx_clone)  # Update hidden state with the clone
        x = F.relu(self.fc2(hx_updated))
        q_value = self.q_head(x)
        return q_value, hx_updated
    """
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

        print("local qs size: ", local_qs.size()) # this is [128, 5, 1] this could be the issue with the inplace operation
        # Ensure w1 is correctly broadcastable with local_qs
        if w1.dim() != local_qs.dim():
            w1 = w1.unsqueeze(-1)

        mixed_qs = torch.einsum('bq,bq->b', w1, local_qs) + w2.sum(dim=1, keepdim=True)
        return mixed_qs

# Define any additional necessary classes or functions

'''
Modifications:
- HyperNetwork: This class generates the weights for the mixing network using the global state.
- MixingNetwork: This class takes the outputs of individual agent's Q-networks and combines them into a total Q-value (Qtot).
- CARLA Integration: The mixing network must be able to handle the state information provided by the CARLA environment, which will be more complex than simple numerical states.

Considerations:
- Ensure the MixingNetwork is compatible with the global state and individual Q-values provided by the CARLA simulation.
- The HyperNetwork's output must be of the appropriate shape and have the correct constraints (e.g., non-negativity for weights).
- You might need to tune the size and structure of the HyperNetwork and MixingNetwork to fit the scale and complexity of your multi-agent system in CARLA.
- Test and validate the network outputs to ensure they are being combined correctly and that the resultant Qtot is reasonable and useful for training.
'''

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

