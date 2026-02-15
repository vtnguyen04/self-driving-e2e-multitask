
class CommandSupervisor:
    """
    Pseudo-code for BFMC Command Logic 'Supervisor'.
    This logic sits ABOVE the E2E model.
    """
    def __init__(self, route_plan):
        self.state = "LANE_FOLLOW"
        self.route_plan = route_plan # e.g., ['STRAIGHT', 'LEFT', 'STRAIGHT']
        self.current_step = 0

    def step(self, car_pose, detected_objects):
        """
        car_pose: (x, y, yaw) from Localization
        detected_objects: List of signs (STOP, PRIORITY, etc.)
        """
        # 1. Check for Intersection
        dist_to_intersection = self.get_dist_to_next_node(car_pose)

        # 2. State Machine
        if self.state == "LANE_FOLLOW":
            cmd = 0 # FOLLOW_LANE

            # Transition: Approaching intersection
            if dist_to_intersection < 1.0: # meters
                self.state = "INTERSECTION_APPROACH"

        elif self.state == "INTERSECTION_APPROACH":
            cmd = 0 # Still follow lane to the stop line

            # Transition: Arrived
            if dist_to_intersection < 0.2:
                 # Get next move from Global Plan
                 next_action = self.route_plan[self.current_step]
                 self.state = "INTERSECTION_TRAVERSE"
                 self.target_command = next_action # 1=Left, 2=Right, 3=Straight
                 self.exit_time = time.time() + 2.0 # Heuristic duration or until line crossed

        elif self.state == "INTERSECTION_TRAVERSE":
            cmd = self.target_command

            # Transition: Completed
            if time.time() > self.exit_time: # Or "Crossed Intersection Exit Line"
                self.state = "LANE_FOLLOW"
                self.current_step += 1

        return cmd

    def get_dist_to_next_node(self, pose):
        # ... logic using Graph/Map ...
        return 999.0
