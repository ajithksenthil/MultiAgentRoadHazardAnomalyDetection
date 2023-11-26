# mSAC_agent.py
import torch
import torch.optim as optim
from mSAC_models import Actor, Critic, MixingNetwork
import numpy as np

class Agent:
    def __init__(self, state_dim, action_dim, num_agents, traffic_manager_api, hazard_detection_model, actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99, alpha=0.2, hidden_dim=256):
        self.hidden_dim = hidden_dim
        self.actors = [Actor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        
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
        self.actor_hxs = [torch.zeros(1, hidden_dim) for _ in range(num_agents)]
        self.critic_hxs = [torch.zeros(1, hidden_dim) for _ in range(num_agents)]
        self.traffic_manager_api = traffic_manager_api
        self.hazard_detection_model = hazard_detection_model

    def select_action(self, state, agent_idx, vehicle):
        # Convert state to tensor and sample action from the actor network
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, hx = self.actors[agent_idx].sample(state_tensor, self.actor_hxs[agent_idx])
        self.actor_hxs[agent_idx] = hx.detach()

        # Check for hazard detection
        hazard_detected = self.detect_hazard(state_tensor)
        if hazard_detected:
            # Respond to hazard using the traffic manager API
            safe_action = self.respond_to_hazard(vehicle)
            # Override the original action with the safe action
            action = safe_action

        return action.detach().cpu().numpy().flatten(), log_prob


    def detect_hazard(self, state_tensor):
        """
        Use the hazard detection model to check if there's a hazard in the current state.
        """
        with torch.no_grad():
            hazard_score = self.hazard_detection_model(state_tensor)
        return torch.sigmoid(hazard_score).item() > 0.5

    def respond_to_hazard(self, vehicle):
        """
        Determines an appropriate response to a detected hazard.
        This function uses the Traffic Manager API to modify the behavior of the vehicle.
        """
        # logic: If a hazard is detected, change lanes or adjust speed

        # Check if a lane change is possible
        can_change_lane_left = self.traffic_manager_api.can_change_lane(vehicle, False)
        can_change_lane_right = self.traffic_manager_api.can_change_lane(vehicle, True)

        # Decide on the direction of lane change or speed adjustment
        if can_change_lane_left:
            self.traffic_manager_api.force_lane_change(vehicle, False)  # Change lane to the left
            # Define a 'safe' action corresponding to a lane change to the left
            safe_action = {'steer': -0.5, 'throttle': 0.7, 'brake': 0.0}
        elif can_change_lane_right:
            self.traffic_manager_api.force_lane_change(vehicle, True)  # Change lane to the right
            # Define a 'safe' action corresponding to a lane change to the right
            safe_action = {'steer': 0.5, 'throttle': 0.7, 'brake': 0.0}
        else:
            # If lane change isn't possible, reduce speed to avoid hazard
            current_speed = self.traffic_manager_api.get_speed_limit(vehicle)
            self.traffic_manager_api.vehicle_percentage_speed_difference(vehicle, -30)  # Reduce speed by 30%
            # Define a 'safe' action corresponding to slowing down
            safe_action = {'steer': 0.0, 'throttle': 0.3, 'brake': 0.1}

        return safe_action


    def update_parameters(self, batch, agent_idx):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_Q_values, current_hx = self.critics[agent_idx](states, actions, self.critic_hxs[agent_idx])
        new_actions, log_probs, new_hx = self.actors[agent_idx].sample(states, self.actor_hxs[agent_idx])
        new_Q_values, _ = self.critics[agent_idx](states, new_actions, new_hx)

        total_Q = self.mixing_network(states, torch.stack([critic(states, actions, hx)[0] for critic, hx in zip(self.critics, self.critic_hxs)], dim=1))
        target_total_Q = torch.zeros_like(total_Q)

        with torch.no_grad():
            for idx in range(self.num_agents):
                target_actions, _, target_hx = self.target_critics[idx].sample(next_states, self.target_critic_hxs[idx])
                target_Q, _ = self.target_critics[idx](next_states, target_actions, target_hx)
                target_total_Q += target_Q

            target_total_Q = rewards + self.gamma * (1 - dones) * target_total_Q

        critic_loss = torch.nn.functional.mse_loss(total_Q, target_total_Q.detach())
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[agent_idx].step()

        actor_loss = -new_Q_values.mean()
        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_idx].step()

        self.critic_hxs[agent_idx] = current_hx.detach()
        self.actor_hxs[agent_idx] = new_hx.detach()

        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.target_mixing_network.soft_update

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

# each script and their key roles in the context of the mSAC implementation for hazard avoidance in CARLA:

# environment.py
# Role: This script is crucial for creating a realistic and interactive simulation environment within CARLA. It handles the initialization of the CARLA world, including setting up vehicles, sensors (like cameras and LIDAR), and the hazard detection model.
# Key Focus: The primary goal is to simulate a dynamic environment where agents can perceive and interact with various elements, including hazardous conditions. The script should accurately capture environmental states and provide the necessary data to agents for decision-making. This involves processing sensor data and translating vehicle actions into the CARLA environment.

# mSAC_models.py
# Role: Houses the neural network architectures for the mSAC algorithm, specifically the Actor, Critic, and Mixing Network models. These models are responsible for learning the optimal policy and value functions.
# Key Focus: The Actor model determines the best actions in given states, while the Critic assesses the quality of those actions. The Mixing Network is crucial for multi-agent scenarios, as it combines individual value functions into a global perspective, aiding in coordinated decision-making for hazard avoidance.

# replay_buffer.py
# Role: Implements the ReplayBuffer, a data structure that stores and retrieves experiences of agents (state, action, reward, next state, done). This is a key component for experience replay in reinforcement learning.
# Key Focus: Efficiently manage past experiences to provide a diverse and informative set of data for training the agents. This helps in stabilizing and improving the learning process, especially in complex environments where hazards need to be detected and avoided.

# traffic_manager_api.py
# Role: Provides an interface to CARLA's Traffic Manager, which controls the behavior of non-player characters (NPCs) and traffic in the simulation.
# Key Focus: Utilize the API to manipulate traffic scenarios and create challenging situations for testing and improving agents' hazard avoidance strategies. This script can help simulate realistic traffic conditions and unexpected events that require quick and effective responses from the agents.

# experiments.py
# Role: Orchestrates the training, testing, and evaluation of the mSAC agents within the CARLA environment. It sets up the environment, initializes agents, and runs the training and evaluation loops.
# Key Focus: Conduct comprehensive experiments to test the effectiveness of the trained agents in hazard avoidance. This includes varying environmental conditions, introducing different types of hazards, and assessing agents' performance under different scenarios.

# mSAC_train.py
# Role: Contains the training loop where the agents interact with the environment, collect experiences, and update their policies and value functions based on the mSAC algorithm.
# Key Focus: The script is central to optimizing the agents' learning process, ensuring they can accurately learn from their environment and improve their hazard avoidance strategies. It manages the balance between exploration and exploitation and updates the agents' neural networks.

# mSAC_agent.py
# Role: Defines the Agent class, which includes mechanisms for decision-making and learning. Each agent uses this class to select actions, update its policy, and learn from experiences.
# Key Focus: Ensure that each agent can independently make informed decisions based on its perception of the environment and collaboratively work towards effective hazard avoidance. This involves managing the actor and critic updates and ensuring proper coordination among multiple agents.
# By focusing on these specific roles and objectives, each script contributes to the overall goal of developing sophisticated agents capable of effectively navigating and avoiding hazards in a dynamic and realistic simulation environment.