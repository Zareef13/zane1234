#!/usr/bin/env python3
import rclpy # ROS2 client library
import numpy as np
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan 
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from scipy.interpolate import splev
from nav_msgs.msg import OccupancyGrid, Path
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
from scipy.interpolate import splrep
import typing as T





class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        xVal, yVal = x
        if (self.statespace_lo[0] <= xVal < self.statespace_hi[0]
            and self.statespace_lo[1] <= yVal < self.statespace_hi[1] and 
            self.occupancy.is_free(np.array(x))):
            return True
        else:
            return False

            
        
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        xval1, yval1 = x1
        xval2, yval2 = x2
        return np.sqrt((xval1 - xval2) ** 2 + (yval1 - yval2) ** 2)
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        directions = [(-self.resolution, 0),
                     (self.resolution, 0),
                     (0, - self.resolution),
                     (0, self.resolution),
                     (-self.resolution, self.resolution),
                     (self.resolution, self.resolution),
                     (- self.resolution, -self.resolution),
                     (self.resolution, - self.resolution)
        ]
        xVal, yVal = x
        for dx, dy in directions:
            nx = xVal + dx
            ny = yVal + dy
            point = self.snap_to_grid((nx, ny))
            if self.is_free(point):
                neighbors.append(point)
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))


    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while self.open_set:
            x_curr = self.find_best_est_cost_through()
            if x_curr == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            else:
                self.open_set.remove(x_curr)
                self.closed_set.add(x_curr)
                neighbors = self.get_neighbors(x_curr)
                for neighbor in neighbors:
                    if neighbor in self.closed_set:
                        continue 
                    tentative_cost_to_arrive = self.cost_to_arrive[x_curr] + self.distance(x_curr, neighbor)

                    if neighbor not in self.open_set:
                        self.open_set.add(neighbor)
                    elif tentative_cost_to_arrive >= self.cost_to_arrive[neighbor]:
                        continue
                    self.came_from[neighbor] = x_curr
                    self.cost_to_arrive[neighbor] = tentative_cost_to_arrive
                    self.est_cost_through[neighbor] = tentative_cost_to_arrive + self.distance(neighbor, self.x_goal)

        return False


        ########## Code ends here ##########



class Navigator(BaseNavigator):
    def __init__(self) -> None:
        super().__init__("Navigator")
        # heading controller 
        self.kp = 2.0 
        self.V_PREV_THRES = 0.0001

        # Trajectory Tracking controllers
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0

        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0
        


    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) ->  TurtleBotControl:
        wrappedError = wrap_angle( goal.theta - state.theta)
        omega = self.kp * wrappedError
        message = TurtleBotControl()
        message.omega = omega
        return message

    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:
        """ Compute control target using a trajectory tracking controller

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """
        x ,y, th = state.x, state.y, state.theta

        dt = t - self.t_prev
        x_d, xd_d, xdd_d = splev(t, plan.path_x_spline, der=0),  splev(t, plan.path_x_spline, der=1),  splev(t, plan.path_x_spline, der=2)
        y_d, yd_d, ydd_d = splev(t, plan.path_y_spline, der=0),  splev(t, plan.path_y_spline, der=1),  splev(t, plan.path_y_spline, der=2)


        ########## Code starts here ##########
        vold = self.V_prev
        if vold < self.V_PREV_THRES:
            vold = self.V_PREV_THRES
        vx = vold * np.cos(th)
        vy = vold * np.sin(th)
        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - vx)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - vy)
        om =  (u2 * np.cos(th) - u1 * np.sin(th)) / vold

        a = u1 * np.cos(th) + u2 * np.sin(th)
        V= a * dt + vold
        if V <  self.V_PREV_THRES:
            V = self.V_PREV_THRES

        ########## Code ends here ##########

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        control = TurtleBotControl()
        control.v = V
        control.omega = om
        return control


    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting

        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """
        x_init = (state.x, state.y)
        x_goal = (goal.x, goal.y)
        statespace_lo = occupancy.origin_xy
        statespace_hi = occupancy.origin_xy + occupancy.size_xy * occupancy.resolution

        astar = AStar(
            statespace_lo,
            statespace_hi,
            x_init,
            x_goal,
            occupancy,
            resolution)

        if not astar.solve() or len(astar.path) < 4:
            return None
        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0


        path = np.asarray(astar.path)
        v_desired = 0.15
        spline_alpha = 0.08

        ts = [0.0]

        for i in range(len(path) - 1):
            distance = np.linalg.norm(path[i + 1] - path[i])
            time = distance / v_desired
            ts.append(ts[-1] + time)
        ts = np.array(ts)
        path_x_spline = splrep(ts, path[:, 0], s=spline_alpha, k=3)

        path_y_spline = splrep(ts, path[:, 1], s=spline_alpha, k=3)

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )

        

        


if __name__ == "__main__":
    rclpy.init()                # initialize ROS2 Client library
    node = Navigator()          # create the node instance 
    rclpy.spin(node)            # call ROS2 default scheduler
    rclpy.shutdown()            # clean up after node exits

