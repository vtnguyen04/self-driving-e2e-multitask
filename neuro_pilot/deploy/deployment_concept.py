import time

class CommandSupervisor:
    """
    Pseudo-code for NeuroPilot Command Logic 'Supervisor'.
    This logic sits ABOVE the E2E model.
    """
    def __init__(self, route_plan):
        self.state = "LANE_FOLLOW"
        self.route_plan = route_plan
        self.current_step = 0

    def step(self, car_pose, detected_objects):
        """
        car_pose: (x, y, yaw) from Localization
        detected_objects: List of signs (STOP, PRIORITY, etc.)
        """
        dist_to_intersection = self.get_dist_to_next_node(car_pose)

        if self.state == "LANE_FOLLOW":
            cmd = 0

            if dist_to_intersection < 1.0:
                self.state = "INTERSECTION_APPROACH"

        elif self.state == "INTERSECTION_APPROACH":
            cmd = 0

            if dist_to_intersection < 0.2:
                 next_action = self.route_plan[self.current_step]
                 self.state = "INTERSECTION_TRAVERSE"
                 self.target_command = next_action
                 self.exit_time = time.time() + 2.0

        elif self.state == "INTERSECTION_TRAVERSE":
            cmd = self.target_command

            if time.time() > self.exit_time:
                self.state = "LANE_FOLLOW"
                self.current_step += 1

        return cmd

    def get_dist_to_next_node(self, pose):
        return 999.0
