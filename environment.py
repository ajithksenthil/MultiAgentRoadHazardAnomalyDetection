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
        self.hazard_detection_model.load_state_dict(torch.load('path_to_saved_model.pth'))
        self.hazard_detection_model.eval()

        # Define the transform for preprocessing the input image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



    def setup_vehicle_and_sensors(self):
        # Code to spawn ego vehicle and attach sensors
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Set vehicle to autopilot and apply Traffic Manager settings
        self.ego_vehicle.set_autopilot(True, self.traffic_manager_api.traffic_manager.get_port())
        self.traffic_manager_api.set_auto_lane_change(self.ego_vehicle, True)

        # Sensor setup code here
        # Example: attaching a camera or lidar senso

    def reset(self):
        # Reset the environment to start a new episode
        # This should reset the position of the vehicle and clear any anomalies
        # Placeholder implementation
        return np.random.rand(self.state_size)

    def step(self, action):
        # Convert action to CARLA vehicle control
        control = carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2])
        self.ego_vehicle.apply_control(control)

        # Process sensor data and detect anomalies
        sensor_data = self.get_sensor_data()  # Retrieve data from sensors
        anomaly_detected = self.hazard_detector.detect_hazards(sensor_data)
        reward = self.calculate_reward(anomaly_detected)

        next_state = self.get_state(sensor_data)
        done = self.check_done()
        info = {"anomaly": anomaly_detected}

        return next_state, reward, done, info



    def detect_anomaly(self):
        # Implement anomaly detection logic
        # Example: Random chance of anomaly or based on vehicle's location
        return random.choice([True, False])

    
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
        # Assuming you have a camera sensor object set up
        image = self.camera_sensor.get_image()
        # Perform necessary conversions if required
        return image

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
        # Process LIDAR data
        # This could involve converting to a point cloud and analyzing it
        processed_lidar_data = lidar_data
        return processed_lidar_data

    def retrieve_lidar_data(self):
        # Assuming you have a LIDAR sensor object set up
        lidar_data = self.lidar_sensor.get_data()
        return lidar_data

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

    def get_state(self):
        state = {
            "sensor_data": self.retrieve_sensor_data(),
            "vehicle_telemetry": self.get_vehicle_telemetry(),
            "environmental_conditions": self.get_environmental_conditions(),
            "hazard_detection": self.anomaly_flag
        }
        return self.format_state_for_network(state)

    def check_done(self):
        if self.reached_destination() or self.encountered_major_hazard() or self.exceeded_time_limit():
            return True
        return False
    
    def cleanup(self):
        # Clean up and destroy actors, sensors, vehicles
        if self.ego_vehicle:
            self.ego_vehicle.destroy()

# Add additional methods as needed for retrieving data and conditions

# # Test case for CarlaEnv
# if __name__ == "__main__":
#     # Test code for environment interaction
#     env = CarlaEnv('localhost', 2000)  # Adjust host and port as necessary
#     state = env.reset()
#     for _ in range(10):
#         action = np.random.rand(env.action_size)
#         next_state, reward, done, info = env.step(action)
#         print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
#         if done:
#             break
#         state = next_state