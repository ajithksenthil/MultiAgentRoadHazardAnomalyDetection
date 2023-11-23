import carla
import random
import time

def main():
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
