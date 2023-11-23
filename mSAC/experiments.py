# experiments.py
import carla
import random
import time
from mSAC_agent import Agent

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    vehicles = []
    try:
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        # Spawn multiple vehicles and set different behaviors
        for _ in range(5):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
            spawn_point = random.choice(spawn_points)
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_autopilot(True)
            traffic_manager.ignore_lights_percentage(vehicle, random.randint(0, 100))
            traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(1.0, 5.0))
            traffic_manager.vehicle_percentage_speed_difference(vehicle, random.randint(-20, 20))
            vehicles.append(vehicle)
            print(f"Spawned vehicle at {spawn_point.location}. Vehicle ID: {vehicle.id}")

        # Simulate for 60 seconds
        for _ in range(60):
            world.tick()
            print("Vehicle states:")
            for vehicle in vehicles:
                location = vehicle.get_location()
                speed = vehicle.get_velocity()
                speed = 3.6 * (speed.x**2 + speed.y**2 + speed.z**2)**0.5  # Convert m/s to km/h
                print(f"Vehicle ID: {vehicle.id}, Location: x={location.x}, y={location.y}, z={location.z}, Speed: {speed:.2f} km/h")
            time.sleep(1)

    finally:
        # Clean up
        for vehicle in vehicles:
            vehicle.destroy()
        print("Vehicles destroyed.")
        print("End of simulation")

if __name__ == '__main__':
    main()
