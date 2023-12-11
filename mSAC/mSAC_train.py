# mSAC_train.py
import os
import torch
from mSAC_agent import Agent
from replay_buffer import ReplayBuffer
import cProfile
# import pstats


# Set default CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(1)  # Set default device in case of multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import json

class TrainingLogger:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []

    def log(self, episode, timestep, rewards, avg_reward=None, hazard_list=None):
        self.data.append({
            "episode": episode,
            "timestep": timestep,
            "rewards": rewards,
            "average_reward": avg_reward,
            "hazard_list": hazard_list
        })

    def save(self):
        with open(self.file_name, 'w') as f:
            json.dump(self.data, f, indent=4)


def train_mSAC(env, num_agents, max_episodes, max_timesteps, batch_size):
    # Initialize the logger
    logger = TrainingLogger("training_log.json")

    # reduce batch size 
    batch_size = batch_size 

    # Initialize agents and their optimizers
    agents = [Agent(state_dim=env.state_size, action_dim=env.action_size, num_agents=num_agents, batch_size = batch_size,
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
        actor = agent.actors[agent_idx].module  # Accessing the specific actor for the given agent index
        critic = agent.critics[agent_idx].module  # Accessing the specific critic for the given agent index
        torch.save(actor.state_dict(), os.path.join(model_dir, f"actor_agent_{agent_idx}_episode_{episode}.pth"))
        torch.save(critic.state_dict(), os.path.join(model_dir, f"critic_agent_{agent_idx}_episode_{episode}.pth"))


    # Function to evaluate agents
    def evaluate_agent(agent, env, agent_idx, num_runs=10):
        total_reward = 0.0
        for _ in range(num_runs):
            states = env.reset()
            initial_hazard_type, initial_hazard_type_list = env.create_hazardous_scenario(num_hazards=5) 
            done = False
            eval_time_step = 0 
            while not done:
                # print("eval_time_step", eval_time_step)
                action, _ = agent.select_action(states[agent_idx], agent_idx)  # Select action for the specific agent

                # Create a list of actions for all agents, with dummy actions for other agents
                actions = [env.default_action() for _ in range(env.num_agents)]
                actions[agent_idx] = action  # Replace the action for the evaluated agent

                next_states, rewards, done, _ = env.step(actions)
                states = next_states
                total_reward += rewards[agent_idx]  # Accumulate reward for the specific agent
                done = env.check_done(max_timesteps=max_timesteps, current_timestep=eval_time_step)
                eval_time_step = eval_time_step + 1

        avg_reward = total_reward / num_runs
        return avg_reward



    profiler = cProfile.Profile()
    num_hazards = 5
    # Training Loop
    torch.autograd.set_detect_anomaly(True)
    for episode in range(max_episodes):
        print("true episode", episode)
        states = env.reset() # gets rid of hazards 
        initial_hazard_type, initial_hazard_type_list = env.create_hazardous_scenario(num_hazards=num_hazards) 
        total_rewards = [0 for _ in range(num_agents)]

        for t in range(max_timesteps):
            print("t", t)
            actions = []
            for i, agent in enumerate(agents):
                action, _ = agent.select_action(states[i], i)  # Corrected call to select_action, no more ego vehicle
                actions.append(action)

            # print(f"mSAC_train.py:Training loop: First action element: {actions[0]}, Type: {type(actions[0])}")
            # profiler.enable()  # Start profiling
            next_states, rewards, done, _ = env.step(actions)
            # profiler.disable()  # Stop profiling
            # stats = pstats.Stats(profiler).sort_stats('tottime')
            # stats.print_stats()
            # print("pushing to replay buffer")
            replay_buffer.push(states, actions, rewards, next_states, [done] * num_agents)
             #print("pushed to rb")
            if len(replay_buffer) > batch_size:
                # print("replay buffer greater than batch size")
                for i in range(num_agents):
                    samples = replay_buffer.sample(batch_size, i)
                    agents[i].update_parameters(samples, i)
            # print("finished updating parameters")
            states = next_states
            for i in range(num_agents):
                total_rewards[i] += rewards[i]
            # print("updated states and total rewards")

            if done:
                break

        if episode % 100 == 0:
            for i, agent in enumerate(agents):
                # print("evaluating agent")
                avg_reward = evaluate_agent(agent, env, i)
                print(f"Agent {i}, Episode {episode}, Evaluation Average Reward: {avg_reward}")
                save_model(agent, episode, i)

        logger.log(episode, t, total_rewards, avg_reward, initial_hazard_type_list)

        torch.cuda.empty_cache()

    print("finished epochs now saving final max model")
    # Save final models and return agents
    for i, agent in enumerate(agents):
        save_model(agent, max_episodes, i)

    print("returning agents")
    # Save the logged data
    logger.save()

    return agents


