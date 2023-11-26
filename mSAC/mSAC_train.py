# mSAC_train.py
import os
import torch
from mSAC_models import Actor, Critic, MixingNetwork
from mSAC_agent import Agent
from replay_buffer import ReplayBuffer

def train_mSAC(env, num_agents, max_episodes, max_timesteps, batch_size):
    # Define the mixing network
    mixing_network = MixingNetwork(num_agents=num_agents, state_dim=env.state_size)

    # Initialize agents and their optimizers
    agents = [Agent(state_dim=env.state_size, action_dim=env.action_size, num_agents=num_agents, hidden_dim=256) for _ in range(num_agents)]

    # Initialize replay buffer
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
    def evaluate_agent(agent, env, num_runs=10):
        total_reward = 0.0
        for _ in range(num_runs):
            state = env.reset()
            done = False
            while not done:
                action, _ = agent.select_action(state, agent_idx)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
        avg_reward = total_reward / num_runs
        return avg_reward

    # Training Loop
    for episode in range(max_episodes):
        states = env.reset()
        total_rewards = [0 for _ in range(num_agents)]

        for t in range(max_timesteps):
            actions = []
            for i, agent in enumerate(agents):
                action, _ = agent.select_action(states[i], i)  # Corrected call to select_action
                actions.append(action)

            next_states, rewards, done, _ = env.step(actions)
            replay_buffer.push(states, actions, rewards, next_states, done)

            if len(replay_buffer) > batch_size:
                for i in range(num_agents):
                    samples = replay_buffer.sample(batch_size, i)
                    agents[i].update_parameters(samples, mixing_network, i)

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

    # Save final models and return agents
    for i, agent in enumerate(agents):
        save_model(agent, max_episodes, i)

    return agents
