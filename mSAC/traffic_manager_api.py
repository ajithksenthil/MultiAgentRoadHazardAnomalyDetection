import carla

class TrafficManagerAPI:
    def __init__(self, traffic_manager):
        self.traffic_manager = traffic_manager
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        # Additional initial configurations can be added here

    def set_auto_lane_change(self, actor, enable=True):
        """ Enables or disables automatic lane changing. """
        self.traffic_manager.auto_lane_change(actor, enable)

    def set_vehicle_speed_limit(self, actor, speed_limit):
        """ Sets a speed limit for the vehicle. """
        percentage_diff = ((speed_limit - actor.get_speed_limit()) / actor.get_speed_limit()) * 100
        self.traffic_manager.vehicle_percentage_speed_difference(actor, percentage_diff)

    def force_lane_change(self, actor, direction):
        """ Forces the vehicle to change lanes. """
        # True for left, False for right
        self.traffic_manager.force_lane_change(actor, direction)

    def enable_collision_detection(self, reference_actor, other_actor, enable=True):
        """ Enables or disables collision detection between specified actors. """
        self.traffic_manager.collision_detection(reference_actor, other_actor, enable)

    def set_distance_to_leading_vehicle(self, actor, distance):
        """ Sets the minimum distance to the leading vehicle. """
        self.traffic_manager.distance_to_leading_vehicle(actor, distance)

    # Additional methods for other Traffic Manager functionalities

# Example usage
# client = carla.Client('localhost', 2000)
# traffic_manager_api = TrafficManagerAPI(client)
# vehicle = ... # Assume this is a carla.Actor instance
# traffic_manager_api.set_auto_lane_change(vehicle, True)
