# experiments.py

import carla
from environment import CarlaEnv
from mSAC_train import train_mSAC

def run_experiment(client, traffic_manager, num_agents, num_episodes):
    env = CarlaEnv(client, traffic_manager)

    # Train mSAC agents
    agents = train_mSAC(env, num_agents, num_episodes, max_timesteps=1000, batch_size=128)

    # Simulation loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            actions = [agent.select_action(state) for agent in agents]
            next_state, rewards, done, info = env.step(actions)
            state = next_state
            total_reward += sum(rewards)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    env.cleanup()

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    
    run_experiment(client, traffic_manager, num_agents=5, num_episodes=100)
