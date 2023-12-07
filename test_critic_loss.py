import torch
import torch.optim as optim
import torch.nn.functional as F
from mSAC_models import Actor, Critic, MixingNetwork

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mock data
state_dim = 20
action_dim = 5
hidden_dim = 256
batch_size = 128
num_agents = 5

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
    critic_hx = torch.zeros(batch_size, hidden_dim).to(device).clone().detach()  # Ensure hx is detached
    current_Q_values, current_hx = critics[agent_idx](states.clone(), actions.clone(), critic_hx.detach())

    actor_hx = torch.zeros(batch_size, hidden_dim).to(device).clone().detach()  # Ensure hx is detached
    new_actions, log_probs, new_hx = actors[agent_idx](states.clone(), actor_hx.detach())

    # new_Q_values, _ = critics[agent_idx](states.clone(), new_actions.clone(), new_hx.detach())
    new_Q_values, _ = critics[agent_idx](states, new_actions, new_hx)
    

    # Calculate total Q-value using mixing network
    total_Q = mixing_network(states.clone(), torch.stack([current_Q_values.clone() for _ in range(num_agents)], dim=1))

    # Calculate target Q-values using dummy data
    target_total_Q = rewards.clone() + (1 - dones.clone()) * total_Q.detach()

    # Critic loss
    critic_loss = F.mse_loss(total_Q, target_total_Q)
    critic_optimizers[agent_idx].zero_grad()
    critic_loss.backward()
    critic_optimizers[agent_idx].step()

    # Detach tensors before actor backward pass
    """  
    new_actions = new_actions.detach()
    log_probs = log_probs.detach()
    new_Q_values = new_Q_values.detach()
    """
    # new_Q_values = new_Q_values.clone().detach()
    # Check if tensors require gradients
    print("new_actions requires_grad:", new_actions.requires_grad)
    print("log_probs requires_grad:", log_probs.requires_grad)
    print("new_Q_values requires_grad:", new_Q_values.requires_grad)

    # Actor loss
    actor_loss = -(log_probs.mean() + new_Q_values.mean())

    # Ensure actor_loss is valid for backward
    actor_optimizers[agent_idx].zero_grad()
    actor_loss.backward()
    actor_optimizers[agent_idx].step()

    # Print versions after backward passes
    print("Versions after critic backward: current_Q_values {}, current_hx {}".format(current_Q_values._version, current_hx._version))
    print("Versions after actor backward: new_actions {}, log_probs {}, new_Q_values {}".format(new_actions._version, log_probs._version, new_Q_values._version))

# Testing update parameters for one agent
torch.autograd.set_detect_anomaly(True)
update_parameters(0)
print("test complete")


""" 
This is my error running this script: 

(base) ajithsenthil@Ajiths-MBP-2 mSAC % python test_critic_loss.py
w1 Shape: torch.Size([128, 5]), w2 Shape: torch.Size([128, 5]), local_qs Shape: torch.Size([128, 5, 1])
new_actions requires_grad: True
log_probs requires_grad: True
new_Q_values requires_grad: True
/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/autograd/__init__.py:200: UserWarning: Error detected in AddmmBackward0. Traceback of forward call that caused the error:
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/test_critic_loss.py", line 150, in <module>
    update_parameters(0)
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/test_critic_loss.py", line 110, in update_parameters
    new_Q_values, _ = critics[agent_idx](states.clone(), new_actions.clone(), new_hx.detach())
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/mSAC_models.py", line 61, in forward
    q_value = self.q_head(x)
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
 (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/python_anomaly_mode.cpp:119.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Traceback (most recent call last):
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/test_critic_loss.py", line 150, in <module>
    update_parameters(0)
  File "/Users/ajithsenthil/Desktop/RobotLearning/FinalProject/mSAC/test_critic_loss.py", line 141, in update_parameters
    actor_loss.backward()
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "/Users/ajithsenthil/opt/anaconda3/lib/python3.9/site-packages/torch/autograd/__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [256, 1]], which is output 0 of AsStridedBackward0, is at version 2; expected version 1 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
(base) ajithsenthil@Ajiths-MBP-2 mSAC % 

"""