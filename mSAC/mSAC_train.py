# mSAC_train.py
import os
import torch
from mSAC_agent import Agent
from replay_buffer import ReplayBuffer

def train_mSAC(env, num_agents, max_episodes, max_timesteps, batch_size):

    # reduce batch size TODO see if this works
    batch_size = batch_size // 4

    # Initialize agents and their optimizers
    agents = [Agent(state_dim=env.state_size, action_dim=env.action_size, num_agents=num_agents,
                    hidden_dim=256) for _ in range(num_agents)]

    # Initialize replay buffer
    print("initialized agents")
    buffer_size = 1000000
    replay_buffer = ReplayBuffer(buffer_size, num_agents)

    # Directory for saving models
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    # Function to save models
    def save_model(agent, episode, agent_idx):
        torch.save(agent.actor.state_dict(), os.path.join(model_dir, f"actor_agent_{agent_idx}_episode_{episode}.pth"))
        torch.save(agent.critic.state_dict(), os.path.join(model_dir, f"critic_agent_{agent_idx}_episode_{episode}.pth"))

    # Function to evaluate agents
    def evaluate_agent(agent, env, agent_idx,num_runs=10):
        total_reward = 0.0
        for _ in range(num_runs):
            state = env.reset()
            done = False
            while not done:
                action, _ = agent.select_action(state, agent_idx) # changed to not include ego vehicle anymore
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
        avg_reward = total_reward / num_runs
        return avg_reward

    # Training Loop
    for episode in range(max_episodes):
        print("episode", episode)
        states = env.reset()
        total_rewards = [0 for _ in range(num_agents)]

        for t in range(max_timesteps):
            actions = []
            for i, agent in enumerate(agents):
                action, _ = agent.select_action(states[i], i)  # Corrected call to select_action, no more ego vehicle
                actions.append(action)

            next_states, rewards, done, _ = env.step(actions)
            replay_buffer.push(states, actions, rewards, next_states, done)

            if len(replay_buffer) > batch_size:
                for i in range(num_agents):
                    samples = replay_buffer.sample(batch_size, i)
                    agents[i].update_parameters(samples, i)

            states = next_states
            for i in range(num_agents):
                total_rewards[i] += rewards[i]

            if done:
                break

        if episode % 100 == 0:
            for i, agent in enumerate(agents):
                avg_reward = evaluate_agent(agent, env, i)
                print(f"Agent {i}, Episode {episode}, Evaluation Average Reward: {avg_reward}")
                save_model(agent, episode, i)
        torch.cuda.empty_cache()

    # Save final models and return agents
    for i, agent in enumerate(agents):
        save_model(agent, max_episodes, i)

    return agents


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