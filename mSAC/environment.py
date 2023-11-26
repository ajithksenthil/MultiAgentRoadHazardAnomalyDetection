import carla
import numpy as np
import random
from traffic_manager_api import TrafficManagerAPI
import sys
import os

# Append the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from gcp_baseline_train import ResNetBinaryClassifier
import torch
import torchvision.transforms as transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CarlaEnv:
    def __init__(self, client, traffic_manager):
        # Initialize CARLA world using provided client and traffic manager
        self.world = client.get_world()

        # Configure Traffic Manager using provided traffic manager
        self.traffic_manager_api = TrafficManagerAPI(traffic_manager)

        # Vehicle and sensor setup
        self.setup_vehicle_and_sensors()

        # State and action dimensions
        self.state_size = 10  # Define the size of the state
        self.action_size = 5  # Define the size of the action
        
        # Initialize the hazard detection model
        self.hazard_detection_model = ResNetBinaryClassifier().to(device)
        model_path = '../trained_model.pth'  # Path to the model in the parent directory
        self.hazard_detection_model.load_state_dict(torch.load(model_path))
        self.hazard_detection_model.eval()

        # Define the transform for preprocessing the input image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



    def setup_vehicle_and_sensors(self):
        # Vehicle setup
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Sensor setup
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # Configure camera parameters if necessary
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Example position
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        # Camera sensor setup with listener
        self.camera_sensor.listen(lambda image_data: self.camera_callback(image_data))

        # LIDAR sensor setup with listener
        self.lidar_sensor.listen(lambda lidar_data: self.lidar_callback(lidar_data))
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        # Configure LIDAR parameters if necessary
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))  # Example position
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)

    def camera_callback(self, image_data):
        # Process the image data
        array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_data.height, image_data.width, 4))
        self.camera_data = array[:, :, :3]  # Store RGB data

    
    def retrieve_camera_data(self):
        # Return the latest processed camera data
        return self.camera_data if self.camera_data is not None else np.zeros((800, 600, 3))

    def retrieve_lidar_data(self):
        # Return the latest processed LIDAR data
        return self.lidar_data if self.lidar_data is not None else np.zeros((...))  # Default size/format


    def create_hazardous_scenario(self):
        """
        Create a hazardous scenario in the environment.
        This function will be called from the experiments script.
        """
        # Randomly select a type of hazard to create
        hazard_type = random.choice(['static_obstacle', 'dynamic_event', 'weather_change'])

        if hazard_type == 'static_obstacle':
            self.create_static_obstacle()
        elif hazard_type == 'dynamic_event':
            self.create_dynamic_event()
        elif hazard_type == 'weather_change':
            self.change_weather_conditions()

    def create_static_obstacle(self):
        """
        Spawn a static obstacle on the road.
        """
        obstacle_bp = self.world.get_blueprint_library().find('static.prop.streetbarrier')
        obstacle_location = self.select_hazard_location()
        self.world.spawn_actor(obstacle_bp, carla.Transform(obstacle_location))

    def create_dynamic_event(self):
        """
        Create a dynamic event like sudden pedestrian crossing or vehicle breakdown.
        """
        # Example: Sudden pedestrian crossing
        pedestrian_bp = self.world.get_blueprint_library().filter("walker.pedestrian.*")[0]
        pedestrian_location = self.select_hazard_location(offset_y=3)  # Slightly offset from the road
        self.world.spawn_actor(pedestrian_bp, carla.Transform(pedestrian_location))

    def change_weather_conditions(self):
        """
        Change weather to adverse conditions.
        """
        # Example: Heavy rain
        weather = carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=100.0,
            precipitation_deposits=50.0,
            wind_intensity=50.0
        )
        self.world.set_weather(weather)

    def select_hazard_location(self, offset_y=0):
        """
        Select a random location on the road to place the hazard.
        """
        map = self.world.get_map()
        spawn_points = map.get_spawn_points()
        hazard_location = random.choice(spawn_points).location
        hazard_location.y += offset_y  # Adjusting position if needed
        return hazard_location

    def reset(self):
        # Reset logic...
        return self.get_state()

    def step(self, action):
        # Convert action to CARLA vehicle control
        control = carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2])
        self.ego_vehicle.apply_control(control)

        # Process sensor data and detect anomalies
        sensor_data = self.get_sensor_data()  # Retrieve data from sensors
        anomaly_detected = self.hazard_detector.detect_hazards(sensor_data)
        reward = self.calculate_reward(anomaly_detected)

        next_state = self.get_state()
        done = self.check_done()
        info = {"anomaly": anomaly_detected}
        
        return next_state, reward, done, info

    
    def process_sensor_data(self):
        # Processing camera data
        camera_data = self.retrieve_camera_data()
        hazards_detected = self.detect_hazards_in_image(camera_data)
        self.hazard_flag = hazards_detected

        # Processing LIDAR data
        lidar_data = self.retrieve_lidar_data()
        processed_lidar_data = self.process_lidar_data(lidar_data)
        self.lidar_info = processed_lidar_data

        # Combining data from all sensors for a complete state representation
        self.current_sensor_state = self.combine_sensor_data(camera_data, processed_lidar_data)


    def get_sensor_data(self):
        # Initialize an empty dictionary to hold sensor data
        sensor_data = {}

        # Retrieve data from each sensor and process it
        sensor_data["camera"] = self.process_camera_data(self.camera_sensor.retrieve_data())
        sensor_data["lidar"] = self.process_lidar_data(self.lidar_sensor.retrieve_data())
        # Include other sensors as necessary

        # Return the processed sensor data
        return sensor_data

    def retrieve_camera_data(self):
        # Retrieve the latest image from the camera sensor
        image = self.camera_sensor.get_image()
        # Convert the CARLA image to a format suitable for processing
        # For example, converting to a numpy array
        image_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_array = np.reshape(image_array, (image.height, image.width, 4))  # Assuming RGBA format
        return image_array[:, :, :3]  # Return RGB part


    def detect_hazards_in_image(self, image):
        # Convert the raw image data to a PIL image
        pil_image = Image.fromarray(image).convert('RGB')

        # Apply the transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(device)

        # Predict with the model
        with torch.no_grad():
            output = self.hazard_detection_model(input_tensor).squeeze()
            prediction = torch.sigmoid(output).item()

        # Determine if a hazard is detected based on the prediction threshold
        hazard_detected = prediction > 0.5  # adjust the threshold as needed
        return hazard_detected

    
    # Helper methods for processing each type of sensor data
    def process_camera_data(self, camera_data):
        # Process camera data (e.g., image normalization)
        processed_camera_data = camera_data
        return processed_camera_data

    def process_lidar_data(self, lidar_data):
        # Process LIDAR data into a simplified 2D array representation

        # # Define the dimensions of the 2D array (example values)
        # width, height = 100, 100  # Adjust based on your needs

        # # Initialize the 2D array with high values (indicating 'no object detected')
        # processed_data = np.full((height, width), np.inf)

        # # Example processing: populate the array with distances
        # for point in lidar_data:
        #     # Convert the point's location to array indices (simplified example)
        #     x_index = int((point.location.x / MAX_X) * width)
        #     y_index = int((point.location.y / MAX_Y) * height)

        #     # Ensure indices are within bounds
        #     x_index = min(max(x_index, 0), width - 1)
        #     y_index = min(max(y_index, 0), height - 1)

        #     # Update the distance in the array if this point is closer
        #     distance = np.sqrt(point.location.x**2 + point.location.y**2)
        #     processed_data[y_index, x_index] = min(processed_data[y_index, x_index], distance)
        processed_data = lidar_data
        return processed_data


    def retrieve_lidar_data(self):
        # Retrieve LIDAR data
        lidar_data = self.lidar_sensor.get_data()
        # Process the LIDAR data as needed for your application
        # For example, converting to a point cloud or a 2D representation
        processed_lidar_data = self.process_lidar_data(lidar_data)  # You need to define this function
        return processed_lidar_data


    def calculate_reward(self, anomaly_detected):
        # Calculate reward based on the presence of anomalies and other factors
        reward = 0.0
        if anomaly_detected:
            reward -= 50  # Penalize for hazard proximity
        # Additional reward calculations
        return reward

    def combine_sensor_data(self, camera_data, lidar_data):
        # Combine data from camera and LIDAR
        # This could mean fusing the data into a single representation
        combined_sensor_data = camera_data + lidar_data
        return combined_sensor_data


    def retrieve_sensor_data(self):
        # Placeholder for sensor data retrieval and processing
        # Example: Get camera and lidar data and convert to numerical features
        camera_data = self.process_camera_data(self.camera_sensor.retrieve_data())
        lidar_data = self.process_lidar_data(self.lidar_sensor.retrieve_data())

        # Combine and flatten data into a single numpy array
        sensor_data = np.concatenate([camera_data, lidar_data])
        return sensor_data

    def get_vehicle_telemetry(self):
        # Get vehicle telemetry data
        velocity = self.ego_vehicle.get_velocity()
        acceleration = self.ego_vehicle.get_acceleration()
        wheel_angle = self.ego_vehicle.get_control().steer

        # Convert to numerical values
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        acc = np.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)

        # Combine into a single array
        telemetry_data = np.array([speed, acc, wheel_angle])
        return telemetry_data

    def get_environmental_conditions(self):
        # Placeholder for environmental conditions
        # Example: Weather conditions, time of day, etc.
        weather = self.world.get_weather()
        time_of_day = self.world.get_snapshot().timestamp.elapsed_seconds

        # Convert to numerical values (this is a simplified example)
        weather_condition = np.array([weather.cloudiness, weather.precipitation, weather.wind_intensity])
        time_data = np.array([time_of_day])

        # Combine into a single array
        environmental_data = np.concatenate([weather_condition, time_data])
        return environmental_data
    
    def get_state(self):
        # Retrieve various components of the state
        sensor_data = self.retrieve_sensor_data()
        vehicle_telemetry = self.get_vehicle_telemetry()
        environmental_conditions = self.get_environmental_conditions()
        hazard_detection = self.anomaly_flag

        # Convert these components into a numerical sequence
        # This is a placeholder implementation; you should replace it with your actual state representation
        state = np.concatenate([sensor_data, vehicle_telemetry, environmental_conditions, [hazard_detection]])
        return state

    def check_done(self):
        if self.reached_destination() or self.encountered_major_hazard() or self.exceeded_time_limit():
            return True
        return False
    
    def reached_destination(self):
        # Placeholder logic for checking if the destination is reached
        # For example, checking the vehicle's position against a target location
        return False  # Replace with actual logic

    def exceeded_time_limit(self):
        # Check if the time limit for the episode is exceeded
        # You need to define and update a variable that tracks the elapsed time
        return False  # Replace with actual logic


    def encountered_major_hazard(self):
        # Logic to determine if a major hazard is encountered
        # For example, checking for collisions or proximity to hazards
        return False  # Replace with actual logic

    def cleanup(self):
        # Destroy the vehicle and sensors
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        if self.camera_sensor:
            self.camera_sensor.stop()  # Stop listening
            self.camera_sensor.destroy()
        if self.lidar_sensor:
            self.lidar_sensor.stop()  # Stop listening
            self.lidar_sensor.destroy()

# Add additional methods as needed for retrieving data and conditions



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