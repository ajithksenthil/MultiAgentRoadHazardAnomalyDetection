# experiments.py
import carla
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from environment import CarlaEnv
from mSAC_train import train_mSAC
import torch


# Set default CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(1)  # Set default device in case of multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_experiment(client, traffic_manager, num_agents, num_episodes):
    env = CarlaEnv(client, traffic_manager, num_agents)

    # Train mSAC agents
    agents = train_mSAC(env, num_agents, num_episodes, max_timesteps=1000, batch_size=128)
    torch.cuda.empty_cache()
    print("finished training mSAC")

    # Data storage for visualization
    episode_data = []
    all_actions = []  # Store all actions here

    # Simulation loop
    max_timesteps_per_episode = 1000  # Set this based on your simulation requirements

    for episode in range(num_episodes):
        hazard_type = env.create_hazardous_scenario()  # Create a hazardous scenario
        states = env.reset()  # states shape: [num_agents, state_size]
        total_rewards = [0 for _ in range(num_agents)]
        episode_collisions = 0
        hazard_encounters = 0
        done = False
        episode_actions = []  # Track actions for this episod
        current_timestep = 0  # Initialize timestep counter for this episode

        while not done:
            actions = []
            for idx, agent in enumerate(agents):
                agent_state = states[idx]  # Get the state for this specific agent
                action, _ = agent.select_action(agent_state, idx)
                actions.append(action)
                episode_actions.append(action.tolist())  # Store actions

            next_states, rewards, done, info = env.step(actions)
            states = next_states

            # Update metrics from info
            episode_collisions += info['collisions']
            hazard_encounters += info['hazard_encounters']

            # Accumulate rewards
            for i, reward in enumerate(rewards):
                total_rewards[i] += reward

            # Increment timestep counter
            current_timestep += 1

            # Check if max timesteps reached or other termination conditions met
            done = done or env.check_done(max_timesteps=max_timesteps_per_episode, current_timestep=current_timestep)

        all_actions.append(episode_actions)  # Store episode actions
        episode_data.append({
            "episode": episode + 1,
            "hazard_type": hazard_type,
            "rewards": sum(total_rewards),
            "collisions": episode_collisions,
            "hazard_encounters": hazard_encounters
        })
        print(f"Episode: {episode+1}, Total Rewards: {total_rewards}, Collisions: {episode_collisions}, Hazard Encounters: {hazard_encounters}")

    
    print("finished simulation")

    # Save the episode data to a CSV file
    df = pd.DataFrame(episode_data)
    df.to_csv('episode_data.csv', index=False)

    # Data Visualization
    print("data visualization")
    visualize_data(df)
    visualize_actions(all_actions)  # Visualize actions
    print("finished data visualization, cleanup")
    env.cleanup()


def visualize_actions(all_actions, save_path='visualization_output'):

    # Create the directory for saving the outputs if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert all_actions to a DataFrame for easier manipulation
    action_columns = ['throttle', 'steer', 'brake', 'lane_change', 'speed_adjustment']
    flattened_actions = []

    for episode_actions in all_actions:
        for action in episode_actions:
            flattened_actions.append([
                action['vehicle_control']['throttle'],
                action['vehicle_control']['steer'],
                action['vehicle_control']['brake'],
                action['traffic_manager_control']['lane_change'],
                action['traffic_manager_control']['speed_adjustment']
            ])

    actions_df = pd.DataFrame(flattened_actions, columns=action_columns)

    # Save actions data to CSV
    actions_csv_path = os.path.join(save_path, 'actions_data.csv')
    actions_df.to_csv(actions_csv_path, index=False)

    # Plotting Throttle, Steer, Brake Distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.histplot(actions_df['throttle'], bins=30, ax=axes[0])
    axes[0].set_title("Throttle Distribution")
    sns.histplot(actions_df['steer'], bins=30, ax=axes[1])
    axes[1].set_title("Steer Distribution")
    sns.histplot(actions_df['brake'], bins=30, ax=axes[2])
    axes[2].set_title("Brake Distribution")
    fig.savefig(os.path.join(save_path, 'action_distributions.png'))
    plt.show()

    # Plotting Lane Change Decisions
    plt.figure(figsize=(8, 6))
    sns.countplot(x='lane_change', data=actions_df)
    plt.title("Lane Change Decisions")
    plt.xlabel("Lane Change")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_path, 'lane_change_decisions.png'))
    plt.show()

    # Plotting Speed Adjustments
    plt.figure(figsize=(8, 6))
    sns.histplot(actions_df['speed_adjustment'], bins=30)
    plt.title("Speed Adjustment Distribution")
    plt.xlabel("Speed Adjustment")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_path, 'speed_adjustment_distribution.png'))
    plt.show()

# Example usage
# visualize_actions(all_actions, save_path='my_visualizations')



def visualize_data(df):
    # Plotting Total Rewards
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='episode', y='rewards', hue='hazard_type')
    plt.title("Total Rewards per Episode by Hazard Type")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(title='Hazard Type')
    plt.savefig('total_rewards_per_episode.png')
    plt.show()

    # Plotting Collision Counts by Hazard Type
    plt.figure(figsize=(12, 6))
    sns.barplot(x='episode', y='collisions', hue='hazard_type', data=df)
    plt.title("Collisions per Episode by Hazard Type")
    plt.xlabel("Episode")
    plt.ylabel("Collision Count")
    plt.legend(title='Hazard Type')
    plt.savefig('collisions_per_episode.png')
    plt.show()

    # Plotting Hazard Encounters by Type
    plt.figure(figsize=(12, 6))
    sns.countplot(x='hazard_type', data=df)
    plt.title("Hazard Encounters by Type")
    plt.xlabel("Hazard Type")
    plt.ylabel("Count")
    plt.savefig('hazard_encounters_by_type.png')
    plt.show()


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