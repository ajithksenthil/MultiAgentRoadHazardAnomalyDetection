# experiments.py
import carla
import os
import sys
from environment import CarlaEnv
from mSAC_train import train_mSAC
import torch


def run_experiment(client, traffic_manager, num_agents, num_episodes):
    env = CarlaEnv(client, traffic_manager, num_agents)

    # Train mSAC agents
    agents = train_mSAC(env, num_agents, num_episodes, max_timesteps=1000, batch_size=128)
    torch.cuda.empty_cache()
    print("finished training mSAC")

    # Simulation loop
    for episode in range(num_episodes):
        states = env.reset()  # states shape: [num_agents, state_size]
        total_rewards = [0 for _ in range(num_agents)]
        done = False

        while not done:
            actions = []
            for idx, agent in enumerate(agents):
                agent_state = states[idx]  # Get the state for this specific agent
                action, _ = agent.select_action(agent_state, idx)
                actions.append(action)

            next_states, rewards, done, info = env.step(actions)
            states = next_states

            # Accumulate rewards for each agent
            for i, reward in enumerate(rewards):
                total_rewards[i] += reward

        print(f"Episode: {episode+1}, Total Rewards: {total_rewards}")
    
    print("finished simulation")

    env.cleanup()

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    run_experiment(client, traffic_manager, num_agents=5, num_episodes=100)




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