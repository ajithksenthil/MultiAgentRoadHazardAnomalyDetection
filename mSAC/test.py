import carla
import random
import time

def main():
    # client = carla.Client('localhost', 2000)
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)

    vehicle = None
    try:
        # Spawn a vehicle and put it on autopilot
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)

        print(f"Spawned vehicle at {spawn_point.location}. Vehicle ID: {vehicle.id}")

        # Configure Traffic Manager
        traffic_manager.ignore_lights_percentage(vehicle, 100)  # Ignore traffic lights
        traffic_manager.distance_to_leading_vehicle(vehicle, 5)  # Maintain a 5 meter distance to leading vehicles

        # Simulate driving for 10 seconds before anomaly
        print("Vehicle is driving...")
        for _ in range(10):
            location = vehicle.get_location()
            print(f"Vehicle location: x={location.x}, y={location.y}, z={location.z}")
            time.sleep(1)

        # Simulate anomaly detection and change behavior
        print("Simulating anomaly detection...")
        anomaly_detected = True
        if anomaly_detected:
            print("Anomaly detected! Changing vehicle behavior.")
            traffic_manager.auto_lane_change(vehicle, False)  # Disable lane changes
            traffic_manager.vehicle_percentage_speed_difference(vehicle, -20)  # Reduce speed by 20%
            print("Vehicle behavior changed: lane change disabled, speed reduced.")
        
        # Observe the vehicle behavior for 30 seconds
        print("Observing changed vehicle behavior...")
        for _ in range(30):
            location = vehicle.get_location()
            speed = vehicle.get_velocity()
            speed = 3.6 * (speed.x**2 + speed.y**2 + speed.z**2)**0.5  # Convert m/s to km/h
            print(f"Vehicle location: x={location.x}, y={location.y}, z={location.z}, speed: {speed:.2f} km/h")
            time.sleep(1)

    finally:
        # Clean up
        if vehicle is not None:
            vehicle.destroy()
            print("Vehicle destroyed.")
        print("End of simulation")

if __name__ == '__main__':
    main()




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