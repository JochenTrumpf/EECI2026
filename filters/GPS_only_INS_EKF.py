import numpy as np
from util import quat_utils, pose_utils

from .filter_state import INS_filter_state
from .filter_covariance import INS_filter_covariance
from .filter import filter

class GPS_only_INS_EKF(filter):
    """
    Error-state EKF for 15 dim state (position, velocity, quaternion, acc bias, gyro bias) with GPS/INS measurements

    filter_state (INS_filter_state):
     p - estimate of 3D position w.r.t. global reference frame expressed in global reference frame
     v - estimate of linear velocity w.r.t. global reference frame expressed in global reference frame
     q - estimate of orientation (quaternion) w.r.t global reference frame expressed in global reference frame; np.array([[1.],[0.],[0.],[0.]]) is identity
     ab - estimate of 3-axis accelerometer bias
     wb - estimate of 3-axis gyro bias
    
    filter_covariance (INS_filter_covariance):
     Pr - estimate of covariance of (robot) state error (15 x 15)

    parameters:
     s_a - estimate of uncertainty of accelerometer measurement a_m in each axis (std.dev. in m/s^2)
     s_w - estimate of uncertainty of gyro measurement w_m in each axis (std.dev in rad/s)

     s_ab - estimate of accelerometer bias drift (= uncertainty of pseudo-measurement) in each axis (std.dev. in m/s^2)
     s_wb - estimate of gyro bias drift (= uncertainty of pseudo-measurement) in each axis (std.dev. in rad/s)

     s_gps - estimate of uncertainty of GPS measurement y_gps in each component (std.dev. in m)

     g - estimate of gravity vector expressed in global reference frame

    configuration:
     <none>
    
    required runtime attributes:
     - inherited from parent class (filter)
        + dt - simulation time step

    Expected measurements:
     + high frequency
       a - measured linear acceleration w.r.t global reference frame expressed in body-fixed frame
       w - measured angular velocity w.r.t global reference frame expressed in body-fixed frame

     + low frequency
       p - measured position w.r.t global reference frame expressed in global reference frame
    """

    class parameters(filter.parameters):
        def __init__(self):
            self.s_a: float
            self.s_w: float

            self.s_ab: float
            self.s_wb: float

            self.s_gps: float

            self.g: np.typing.NDArray

            super().__init__()

    s: INS_filter_state
    s_traj: list[INS_filter_state]
    P: INS_filter_covariance
    P_traj: list[INS_filter_covariance]

    def __init__(self, params: GPS_only_INS_EKF.parameters, config: GPS_only_INS_EKF.configuration):
        # override parent class types
        self.params: GPS_only_INS_EKF.parameters
        self.config: GPS_only_INS_EKF.configuration
        
        super().__init__(params, config)

    @classmethod
    def get_var_attributes(cls):
        return INS_filter_state.var_attributes 
    
    def initialize(self, initial_state: INS_filter_state, initial_covariance: INS_filter_covariance):
        super().initialize(initial_state, initial_covariance)

    def compute_state_error(self, true_state: INS_filter_state, accumulate: bool):
        super().compute_state_error(true_state, accumulate)

    def predict(self, m: dict):
        if 'dt' not in self.runtime_attributes:
            raise SystemExit('predict() called but runtime attribute dt was not set for filter {}.'.format(self.name))

        dt = self.runtime_attributes['dt']

        # check that we have the required measurements, if not, do nothing
        if ('a' not in m) or ('w' not in m):
            return

        # predict nominal (big signal) state estimates (partly Euler)
        R = quat_utils.quat2Rot(self.s.q)
        self.s.p = self.s.p + dt*self.s.v +.5 * dt * dt * (R @ (m['a'] - self.s.ab) + self.params.g) 
        self.s.v = self.s.v + dt * (R @ (m['a'] - self.s.ab) + self.params.g)
        self.s.q = quat_utils.quat_exp(m['w'] - self.s.wb, self.s.q, dt)
    
        # predict error states and covariance (Euler)
        I = np.eye(3)

        #F_x = np.block([ [np.eye(3), dt*np.eye(3), -dt**2*R @ vec2skew(a_m-ab)/2, -dt**2*R/2, dt**3*R @ vec2skew(a_m-ab)/6],
        #                 [np.zeros((3,3)), np.eye(3), -dt*R @ vec2skew(a_m-ab), -dt*R, dt**2*R @ vec2skew(a_m-ab)/2],
        #                 [np.zeros((3,3)), np.zeros((3,3)), R_w.T, np.zeros((3,3)), -dt*np.eye(3)],
        #                 [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,3))],
        #                 [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]  ]) # 15 x 15
        F_x = np.eye(15)

        A = R @ quat_utils.vec2skew(m['a'] - self.s.ab)
        R_w = pose_utils.ang_vel2Rot(m['w'] - self.s.wb, dt)

        F_x[0:3,3:6] = dt*I
        F_x[0:3,6:9] = (-.5*dt*dt)*A
        F_x[0:3,9:12] = (-.5*dt*dt)*R
        F_x[0:3,12:15] = (dt*dt*dt/6)*A

        F_x[3:6,6:9] = -dt*A
        F_x[3:6,9:12] = -dt*R
        F_x[3:6,12:15] = (0.5*dt*dt)*A
    
        F_x[6:9,6:9] = R_w.T
        F_x[6:9,12:15] = -dt*I 

        #F_i = np.block([[np.zeros((3,12))],
        #                [np.eye(3), np.zeros((3,9))],
        #                [np.zeros((3,3)), np.eye(3), np.zeros((3,6))],
        #                [np.zeros((3,6)), np.eye(3), np.zeros((3,3))],
        #                [np.zeros((3,9)), np.eye(3)]]) # 15 x 12
        F_i = np.zeros((15,12))

        F_i[3:6,0:3] = I
        F_i[6:9,3:6] = I
        F_i[9:12,6:9] = I
        F_i[12:15,9:12] = I
    
        #Q_i = np.block([[s_a**2*dt**2*np.eye(3), np.zeros((3,9))],
        #                [np.zeros((3,3)), s_w**2*dt**2*np.eye(3), np.zeros((3,6))],
        #                [np.zeros((3,6)), d_ab**2*dt*np.eye(3), np.zeros((3,3))],
        #                [np.zeros((3,9)), d_wb**2*dt*np.eye(3)]]) # 12x12
        Q_i = np.zeros((12,12))

        Q_i[0:3,0:3] = (self.params.s_a*self.params.s_a*dt*dt)*I
        Q_i[3:6,3:6] = (self.params.s_w*self.params.s_w*dt*dt)*I
        Q_i[6:9,6:9] = (self.params.s_ab*self.params.s_ab)*dt*I
        Q_i[9:12,9:12] = (self.params.s_wb*self.params.s_wb)*dt*I
    
        self.P.Pr =  F_x @ self.P.Pr @ F_x.T + F_i @ Q_i @ F_i.T

    def update(self, m: dict, t_step: int, *_):
        # tell pylance to shut up about the fact that t_step is unused in this implementation
        del t_step 

        # check that we have the required measurements, if not, do nothing
        if ('p' not in m) or (m['p'] is None):
            return
    
        X_dx = np.block([[np.eye(6), np.zeros((6,9))],
                         [np.zeros((4,6)), .5*quat_utils.left_mat(self.s.q) @ np.block([[np.zeros((1,3))],[np.eye(3)]]), np.zeros((4,6))],
                         [np.zeros((6,9)), np.eye(6)]]) #16x15
        y = m['p']
    
        H_x = np.block([[np.eye(3), np.zeros((3,13))]]) #3x16
        H = H_x @ X_dx # 3x15

        K = self.P.Pr @ H.T @ np.linalg.inv(H @ self.P.Pr @ H.T + self.params.s_gps * np.eye(3)) # 15x3
    
        h_x = self.s.p
        dx = K @ (y - h_x)

        # P_new = (np.eye(6) - K @ H) @ P
        # More stable Joseph form (I − KH) P (I − KH)^T + K V K^T
        self.P.Pr = (np.eye(15) - K @ H) @ self.P.Pr @ (np.eye(15) - K @ H).T + K @ (self.params.s_gps * np.eye(3)) @ K.T

        #inject
        self.s.p = self.s.p + dx[0:3, :]
        self.s.v = self.s.v + dx[3:6, :]
        self.s.q = quat_utils.quat_mult(self.s.q, quat_utils.ang_vel2quat(dx[6:9, :], 1.0))
        self.s.ab = self.s.ab + dx[9:12, :]
        self.s.wb = self.s.wb + dx[12:15, :]

        # reset
        G = np.block([[np.eye(6), np.zeros((6,9))],
                      [np.zeros((3,6)), np.eye(3)-quat_utils.vec2skew(.5 * dx[6:9, :]), np.zeros((3,6))],
                      [np.zeros((6,9)), np.eye(6)]])
        self.P.Pr =  G @ self.P.Pr @ G.T
        