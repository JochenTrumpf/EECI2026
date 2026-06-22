import numpy as np
from util import quat_utils, pose_utils

from sensors.sensor import sensor

from .robot_state import INS_robot_state
from .robot import robot

from filters.filter import filter

class simple_quad(robot):
    """
    Quadrotor robot controlled by waypoints, measuring GPS, IMU and relative bearings to stationary landmarks

    robot_state (IMU_robot_state):
     p - 3D position w.r.t. global reference frame expressed in global reference frame
     v - linear velocity w.r.t. global reference frame expressed in global reference frame
     q - orientation (quaternion) w.r.t global reference frame expressed in global reference frame; np.array([[1.],[0.],[0.],[0.]]) is identity
     ab - 3-axis accelerometer bias
     wb - 3-axis gyro bias

     a - linear acceleration w.r.t. global reference frame expressed in body-fixed reference frame
     w - angular velocity (3D vector) w.r.t. global reference frame expressed in body-fixed reference frame
    
    parameters:
     wayposes - list of wayposes to pass through (including initial pose)
     prox_lim - how close to a waypose counts as "pass through" (in m)
      
     g - gravity vector expressed in global reference frame
      
     d_ab - accelerometer bias drift (in m/s^3)
     d_wb - gyro bias drift (in rad/s^2)

    configuration:
     <none>

    required runtime attributes:
     - inherited from parent class (robot)
        + dt - simulation time step
    """

    class parameters(robot.parameters):
        def __init__(self):
            self.wayposes: list[pose_utils.Pose]
            self.prox_lim: float

            self.g: np.typing.NDArray

            self.d_ab: float
            self.d_wb: float

            super().__init__()
    
    s: INS_robot_state
    s_traj: list[INS_robot_state]

    def __init__(self, params: simple_quad.parameters, config: robot.configuration, sensors: tuple[sensor, ...], filters: tuple[filter, ...]):
        # override parent class types
        self.params: simple_quad.parameters
        
        super().__init__(params, config, sensors, filters)

        self.waypose_steps: list[int]
        self.remaining_wayposes: list[pose_utils.Pose]

    @classmethod
    def get_var_attributes(cls):
        return INS_robot_state.var_attributes
    
    def initialize(self, initial_state: INS_robot_state):
        super().initialize(initial_state)

        # initialize array for recording of time steps when wayposes are reached and list of remaining wayposes
        self.waypose_steps = [0]
        self.remaining_wayposes = self.params.wayposes[1:]

    def _update_state(self):
        if 'dt' not in self.runtime_attributes:
            raise SystemExit('_update_state() called but runtime attribute dt was not set for robot {}.'.format(self.name))

        dt = self.runtime_attributes['dt']

        # compute robot trajectory from waypoints
        done, change, a, w, self.remaining_wayposes = pose_utils.waypoint_nav(
            self.remaining_wayposes, 
            pose_utils.Pose(self.s.q, self.s.p), 
            self.s.v, self.s.a, 
            self.params.g, dt, self.params.prox_lim
        )

        if change:
            self.waypose_steps.append(self.T)   # record step when waypose was reached

        if not done:
            R = quat_utils.quat2Rot(self.s.q)
            self.s.p = self.s.p + dt * self.s.v + .5 * dt * dt * (R @ a + self.params.g)
            self.s.v = self.s.v + dt * (R @ a + self.params.g)
            self.s.q = quat_utils.quat_exp(w, self.s.q, dt)
            self.s.ab = self.s.ab + dt * self.params.d_ab * np.random.randn(3,1)
            self.s.wb = self.s.wb + dt * self.params.d_wb * np.random.randn(3,1)

            self.s.a = a
            self.s.w = w

        return done
