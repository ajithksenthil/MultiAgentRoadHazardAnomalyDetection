# mSAC_train.py
import os
import torch
from mSAC_models import Actor, Critic, MixingNetwork
from mSAC_agent import Agent
from replay_buffer import ReplayBuffer
from environment import CarlaEnv

# Initialize Carla environment
env = CarlaEnv()  # Ensure CarlaEnv is configured for multiple agents

# Define the number of agents and the mixing network
num_agents = 2  # Adjust based on your scenario
mixing_network = MixingNetwork(num_agents=num_agents, state_dim=env.state_size)

# Initialize agents and their optimizers
agents = [Agent(state_size=env.state_size, action_size=env.action_size) for _ in range(num_agents)]

# Initialize replay buffer
buffer_size = 1000000  # Adjust as needed
replay_buffer = ReplayBuffer(buffer_size, num_agents)  # Ensure ReplayBuffer handles experiences for multiple agents

# Training hyperparameters
max_episodes = 10000
max_timesteps = 1000
batch_size = 128

# Directory for saving models
model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)

def save_model(agent, episode, agent_idx):
    torch.save(agent.actor.state_dict(), os.path.join(model_dir, f"actor_agent_{agent_idx}_episode_{episode}.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(model_dir, f"critic_agent_{agent_idx}_episode_{episode}.pth"))

def evaluate_agent(agent, env, num_runs=10):
    total_reward = 0.0
    for _ in range(num_runs):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
    avg_reward = total_reward / num_runs
    return avg_reward

for episode in range(max_episodes):
    states = env.reset()  # Reset environment and get initial states for all agents
    total_rewards = [0 for _ in range(num_agents)]

    for t in range(max_timesteps):
        actions = [agent.select_action(states[i]) for i, agent in enumerate(agents)]
        next_states, rewards, done, _ = env.step(actions)  # Apply actions and get results

        # Store transitions in replay buffer
        replay_buffer.push(states, actions, rewards, next_states, done)

        # Update parameters of each agent if replay buffer is sufficiently large
        if len(replay_buffer) > batch_size:
            for i, agent in enumerate(agents):
                samples = replay_buffer.sample(batch_size, i)  # Ensure replay_buffer.sample handles multi-agent setup
                agent.update_parameters(samples, mixing_network)

        states = next_states
        for i in range(num_agents):
            total_rewards[i] += rewards[i]

        if done:
            break

    avg_reward = sum(total_rewards) / num_agents
    print(f"Episode {episode} Average Reward: {avg_reward}")

    # Performance evaluation and model saving
    if episode % 100 == 0:
        for i, agent in enumerate(agents):
            avg_reward = evaluate_agent(agent, env)
            print(f"Agent {i}, Episode {episode}, Evaluation Average Reward: {avg_reward}")
            save_model(agent, episode, i)

# Save final models
for i, agent in enumerate(agents):
    save_model(agent, max_episodes, i)


# '''
# Modifications:
# - Multi-agent support in the environment and replay buffer.
# - Each agent updates using its own set of experiences.
# - Performance evaluation and model saving logic to be implemented.

# Additional Notes:
# - Ensure the `CarlaEnv` class is capable of simulating multiple agents and returning their states and rewards.
# - Implement model saving and performance evaluation to track agent learning progress.
# - Consider any specific configurations or constraints of the CARLA simulator and Traffic Manager.
# '''

# '''
# Key Components:
# - Initialization: Set up the environment, agents, and replay buffer.
# - Looping Over Episodes: Each episode corresponds to a run of the simulation.
# - Action Selection: Each agent selects an action based on the current state.
# - Environment Step: Apply the actions to the environment and observe the next state and reward.
# - Replay Buffer: Store experiences (state, action, reward, next state, done) in the buffer.
# - Agent Update: Sample a batch from the replay buffer and update each agent's parameters.
# - Logging and Saving: Print out rewards for monitoring and save models periodically.

# Additional Considerations:
# - Hyperparameters: Define or tune hyperparameters such as learning rates, buffer size, discount factor, etc.
# - Environment Interface: Ensure the environment provides the necessary methods (reset, step) and properties (state_size, action_size).
# - Agent Methods: Ensure the Agent class has methods for selecting actions (select_action) and updating parameters (update_parameters).
# - Error Handling: Include error handling and validation checks as needed.

# This script forms the core training loop for the mSAC algorithm. Fill in specifics related to your environment and agents, and consider adding functionality for evaluation, logging, and model saving as per your requirements.
# '''