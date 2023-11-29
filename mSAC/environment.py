# environment.py
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
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# Set default CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(1)  # Set default device in case of multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CarlaEnv:
    def __init__(self, client, traffic_manager, num_agents):
        # Initialize CARLA world using provided client and traffic manager
        self.world = client.get_world()

        # Configure Traffic Manager using provided traffic manager
        self.traffic_manager_api = TrafficManagerAPI(traffic_manager)

        # Initialize the hazard detection model
        self.hazard_detection_model = ResNetBinaryClassifier().to('cuda:1')
        model_path = '../trained_model.pth'  # Path to the model in the parent directory
        self.hazard_detection_model.load_state_dict(torch.load(model_path, map_location='cuda:1'))
        self.hazard_detection_model = nn.DataParallel(self.hazard_detection_model, device_ids=[1, 0])
        self.hazard_detection_model = self.hazard_detection_model.to('cuda:1')
        self.hazard_detection_model.eval()

        # Vehicle and sensor setup
        self.camera_sensor_data = {}  # Dictionary to store camera data for each vehicle
        self.setup_vehicle_and_sensors(num_agents=num_agents)

        # Initialize data structures for storing individual vehicle states
        self.anomaly_flags = [False] * num_agents
        self.vehicle_states = [None] * num_agents

        # State and action dimensions
        # Vehicle telemetry + environmental conditions + hazard detection
        additional_data_size = 3 + 3 + 1  # telemetry (3) + environmental conditions (3) + hazard detection (1)

        # Vehicle info
        vehicle_info_size = 12  # location (3) + rotation (3) + velocity (3) + acceleration (3)

        # Total state size
        total_state_size = additional_data_size + vehicle_info_size

        self.state_size = int(total_state_size)
        self.action_size = 5  # Define the size of the action (3 for vehicle control and 2 for traffic manager control)

        # Define the transform for preprocessing the input image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def setup_vehicle_and_sensors(self, num_agents):
        self.vehicles = []
        self.camera_sensors = []

        # Vehicle setup
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()

        for _ in range(num_agents):
            vehicle = None
            for spawn_point in spawn_points:
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle is not None:
                    break

            if not vehicle:
                raise RuntimeError("Failed to spawn vehicle: No free spawn points available.")

            self.vehicles.append(vehicle)

            # Sensor setup for each vehicle
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('sensor_tick', '0.1')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Adjust as needed for each vehicle
            camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            camera_sensor.listen(lambda image_data: self.camera_callback(image_data, vehicle.id))
            self.camera_sensors.append(camera_sensor)



    def get_state(self):

        states = [] # Initialize an empty list to hold the states of all vehicles

        for idx, vehicle in enumerate(self.vehicles):
            # Retrieve various components of the state
            vehicle_telemetry = self.get_vehicle_telemetry(vehicle)
            environmental_conditions = self.get_environmental_conditions()

            # Assuming anomaly_flag is a boolean, convert it to an integer for concatenation
            hazard_detection = 1 if self.anomaly_flags[idx] else 0

            # Extract vehicle information and flatten it
            vehicle_info_dict = self.get_vehicle_info(vehicle)
            vehicle_info = np.array([vehicle_info_dict[key] for key in ['location', 'rotation', 'velocity', 'acceleration']]).flatten()

            # Concatenate all the components into a single array
            state = np.concatenate([vehicle_telemetry, environmental_conditions, [hazard_detection], vehicle_info])
            states.append(state)  # Append the state of this vehicle to the states list

        return states
    
    def step(self, actions):
        """
        Perform a step in the environment based on the given actions for each vehicle.

        Args:
            actions (list): A list of actions, one for each vehicle.

        Returns:
            next_states (list): The next states of all vehicles.
            rewards (list): Rewards obtained by each vehicle.
            done (bool): Whether the episode is finished.
            info (dict): Additional information about the step.
        """
        rewards = []
        anomaly_detected_flags = []

        for idx, vehicle in enumerate(self.vehicles):
            action = actions[idx]  # Get the action for this vehicle

            # Decompose the action into vehicle control and traffic manager control parts
            vehicle_control_action = action['vehicle_control']
            traffic_manager_action = action['traffic_manager_control']

            # Apply vehicle control
            self.apply_control_to_vehicle(vehicle, vehicle_control_action)

            # Perform traffic manager related actions if applicable
            self.perform_traffic_manager_control(vehicle, traffic_manager_action)

            # Update sensor data and hazard detection for this vehicle
            self.process_sensor_data(vehicle)
            anomaly_detected = self.detect_hazards_for_vehicle(vehicle)
            self.anomaly_flags[idx] = anomaly_detected
            anomaly_detected_flags.append(anomaly_detected)

            # Calculate reward for this vehicle
            reward = self.calculate_reward(vehicle, anomaly_detected, action)
            rewards.append(reward)

        # Check if the episode is done
        done = self.check_done()

        # Get the next state for all vehicles
        next_states = self.get_state()

        info = {"anomaly": anomaly_detected_flags}

        return next_states, rewards, done, info

    def calculate_reward(self, vehicle, anomaly_detected, action):
        """
        Calculate the reward for a vehicle based on anomaly detection and other factors.

        Args:
            vehicle: The vehicle for which to calculate the reward.
            anomaly_detected (bool): Whether an anomaly was detected.
            action: The action taken by the vehicle.

        Returns:
            float: The calculated reward for the vehicle.
        """
        reward = 0.0

        # Penalize for proximity to hazards (example calculation, adjust as needed)
        hazard_proximity = self.calculate_hazard_proximity(vehicle)
        reward -= self.hazard_proximity_penalty(hazard_proximity)

        # Reward for detecting an anomaly
        if anomaly_detected:
            reward += 50  # Positive reward for detecting an anomaly

        # Additional considerations based on the action taken
        # Example: Penalize for risky maneuvers or excessive speed

        return reward

    def hazard_proximity_penalty(self, proximity):
        """
        Calculate a penalty based on the proximity to a hazard.

        Args:
            proximity (float): The proximity to the nearest hazard.

        Returns:
            float: The penalty for being close to a hazard.
        """
        # Example penalty calculation, modify based on your hazard proximity measure
        penalty = max(0, 50 - proximity * 10)  # Linear penalty as an example
        return penalty

    def calculate_hazard_proximity(self, vehicle):
        """
        Calculate the proximity of a vehicle to the nearest hazard.

        Args:
            vehicle: The vehicle for which to calculate the hazard proximity.

        Returns:
            float: The proximity to the nearest hazard.
        """
        # Implement logic to calculate the distance to the nearest hazard
        # This is an example and needs to be adjusted based on your environment
        nearest_hazard_distance = self.get_nearest_hazard_distance(vehicle)
        return nearest_hazard_distance

    def apply_control_to_vehicle(self, vehicle, control_actions):
        """
        Apply control actions to a specific vehicle.

        Args:
            vehicle (carla.Vehicle): The vehicle to control.
            control_actions (dict): Control actions including throttle, steer, and brake.
        """
        throttle, steer, brake = control_actions['throttle'], control_actions['steer'], control_actions['brake']
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        vehicle.apply_control(control)
    
    def perform_traffic_manager_control(self, vehicle, action):
        """
        Perform traffic manager control actions for a specific vehicle.

        Args:
            vehicle (carla.Vehicle): The vehicle to apply traffic manager actions to.
            action (dict): Traffic manager actions including lane change and speed adjustment.
        """
        lane_change_decision = action['lane_change']
        speed_adjustment = action['speed_adjustment']

        # Perform lane change if necessary
        if lane_change_decision == "left":
            self.traffic_manager_api.force_lane_change(vehicle, to_left=True)
        elif lane_change_decision == "right":
            self.traffic_manager_api.force_lane_change(vehicle, to_left=False)

        # Adjust speed if necessary
        if speed_adjustment != 0:
            current_speed_limit = self.traffic_manager_api.get_speed_limit(vehicle)
            new_speed_limit = current_speed_limit + speed_adjustment
            self.traffic_manager_api.set_speed_limit(vehicle, new_speed_limit)



    def camera_callback(self, image_data, vehicle_id):
        # Process the image data
        array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_data.height, image_data.width, 4))
        camera_data = array[:, :, :3]  # Store RGB data
        self.camera_sensor_data[vehicle_id] = camera_data
        # Perform hazard detection
        hazard_detected = self.detect_hazards_in_image(camera_data)

        # Update the anomaly flag based on hazard detection for the specific vehicle
        # You might want to maintain a dictionary of anomaly flags for each vehicle
        self.anomaly_flags[vehicle_id] = hazard_detected

    


    def retrieve_camera_data(self, vehicle_id):
        # Retrieve the latest processed camera data for a specific vehicle
        return self.camera_sensor_data.get(vehicle_id, np.zeros((800, 600, 3)))


    def get_vehicle_info(self, vehicle):
        """
        Retrieve information about a specific vehicle.

        Args:
            vehicle (carla.Vehicle): The vehicle to retrieve information for.
        """
        location = vehicle.get_location()
        rotation = vehicle.get_transform().rotation
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()

        vehicle_info = {
            'location': (location.x, location.y, location.z),
            'rotation': (rotation.pitch, rotation.yaw, rotation.roll),
            'velocity': (velocity.x, velocity.y, velocity.z),
            'acceleration': (acceleration.x, acceleration.y, acceleration.z)
        }
        return vehicle_info

    def get_vehicle_telemetry(self, vehicle):
        """
        Get telemetry data for a specific vehicle.

        Args:
            vehicle (carla.Vehicle): The vehicle to get telemetry data for.
        """
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        wheel_angle = vehicle.get_control().steer

        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        acc = np.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)

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
    

    # environment interaction methods

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
        # Reset logic...needs to get states for multi agent set up, 2d array where we can use it in the train script and iterate over each state for each vehicle
        return self.get_state()
    
    
    def process_sensor_data(self):
        # Processing camera data # TODO modify for multi agent
        camera_data = self.retrieve_camera_data()
        hazards_detected = self.detect_hazards_in_image(camera_data)

        # Update the anomaly flag based on hazard detection
        self.anomaly_flag = hazards_detected
        self.current_sensor_state = {'camera': camera_data}
   

    def detect_hazards_for_vehicle(self, vehicle):
        """
        Detect hazards for a specific vehicle and get its location.

        Args:
            vehicle: The vehicle for which to detect hazards.

        Returns:
            tuple: A tuple containing a boolean indicating whether a hazard is detected 
                and the location of the vehicle.
        """
        # Retrieve the image data from the vehicle's sensor
        image_data = self.get_image_data_from_vehicle(vehicle)

        # Convert the raw image data to a PIL image
        pil_image = Image.fromarray(image_data).convert('RGB')

        # Apply the transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to('cuda:1')

        # Predict with the model
        with torch.no_grad():
            output = self.hazard_detection_model(input_tensor).squeeze()
            prediction = torch.sigmoid(output).item()

        # Determine if a hazard is detected based on the prediction threshold
        hazard_detected = prediction > 0.5  # adjust the threshold as needed

        # Get the vehicle's location
        vehicle_info_dict = self.get_vehicle_info(vehicle)
        vehicle_location = vehicle_info_dict["location"]

        return hazard_detected, vehicle_location




    def retrieve_sensor_data(self):
        """
        Retrieve and combine processed data from all sensors.
        """
        # TODO modify for multi agent
        camera_data = self.retrieve_camera_data()
        sensor_data = {
            'camera': camera_data
        }

        # Additional processing might be required here
        return sensor_data

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

# Add additional methods as needed for retrieving data and conditions
