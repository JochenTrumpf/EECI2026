import numpy as np
from util import quat_utils, pose_utils

from .filter_state import INS_filter_state
from .filter_covariance import INS_EqF_covariance
from .filter import filter

class GPS_only_INS_EqF(filter):
    """
    Error-state EKF for 15 dim state (position, velocity, quaternion, acc bias, gyro bias) with full GPS/INS measurements

    filter_state (INS_filter_state):
     p - estimate of 3D position w.r.t. global reference frame expressed in global reference frame
     v - estimate of linear velocity w.r.t. global reference frame expressed in global reference frame
     q - estimate of orientation (quaternion) w.r.t global reference frame expressed in global reference frame; np.array([[1.],[0.],[0.],[0.]]) is identity
     ab - estimate of 3-axis accelerometer bias
     wb - estimate of 3-axis gyro bias
    
    filter_covariance (INS_EqF_covariance):
     P - estimate of covariance of full state error (15 x 15)
     Pr - estimate of (marginalized) covariance for state error without bias states (9 x 9)   
     
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
    P: INS_EqF_covariance
    P_traj: list[INS_EqF_covariance]

    def __init__(self, params: GPS_only_INS_EqF.parameters, config: GPS_only_INS_EqF.configuration):
        # override parent class types
        self.params: GPS_only_INS_EqF.parameters
        self.config: GPS_only_INS_EqF.configuration
        
        super().__init__(params, config)

    @classmethod
    def get_var_attributes(cls):
        return INS_filter_state.var_attributes 
    
    def initialize(self, initial_state: INS_filter_state, initial_covariance: INS_EqF_covariance):
        super().initialize(initial_state, initial_covariance)

    def compute_state_error(self, true_state: INS_filter_state, accumulate: bool):
        super().compute_state_error(true_state, accumulate)

    def predict(self, m: dict):
        if 'dt' not in self.runtime_attributes:
            raise SystemExit('predict() called but runtime attribute dt was not set for filter {}.'.format(self.name))

        # check that we have the required measurements, if not, do nothing
        if ('a' not in m) or ('w' not in m):
            return

        # time step
        dt = self.runtime_attributes['dt']

        # current filter state
        Rhat = quat_utils.quat2Rot(self.s.q)
        vhat = self.s.v
        phat = self.s.p
        
        # current filter covariance (9x9)
        Robs_fault = np.block([
            [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)], 
            [np.zeros((3,3)), np.eye(3), np.zeros((3,3))], 
            [np.eye(3), np.zeros((3,3)), np.zeros((3,3))]
        ])
        P = Robs_fault.T @ self.P.Pr @ Robs_fault

        # current accelerometer and gyro measurements
        acc = m['a']
        gyr = m['w']

        # per component standard deviations
        s_a = self.params.s_a   # accel
        s_w = self.params.s_w   # gyro

        # gravity
        g = self.params.g
        
        # START implementation of the EqF predict step here
        # the code will only arrive here when a valid GPS measurement was just received
        # you need to update Rhat, phat and vhat based on p


    
        # END implementation of the EqF predict step here

        # write back the updated state and covariance into the internal representation 
        self.s.q = quat_utils.Rot2quat(Rhat)
        self.s.p = phat
        self.s.v = vhat

        self.P.P = np.block([[Robs_fault.T @ P @ Robs_fault, np.zeros((9,6))], [np.zeros((6,9)), np.eye(6)]]) 

    def update(self, m: dict, t_step: int, *_):
        # tell pylance to shut up about the fact that t_step is unused in this implementation
        del t_step 

        # check that we have the required measurements, if not, do nothing
        if ('p' not in m) or (m['p'] is None):
            return
    
        # current filter state
        Rhat = quat_utils.quat2Rot(self.s.q)
        phat = self.s.p
        vhat = self.s.v
        
        # current filter covariance (9x9)
        Robs_fault = np.block([
            [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)], 
            [np.zeros((3,3)), np.eye(3), np.zeros((3,3))], 
            [np.eye(3), np.zeros((3,3)), np.zeros((3,3))]
        ])
        P = Robs_fault.T @ self.P.Pr @ Robs_fault

        # current GPS measurement
        p = m['p']

        # per component GPS standard deviation
        s_p = self.params.s_gps
        
        # START implementation of the EqF update step here
        # the code will only arrive here when a valid GPS measurement was just received
        # you need to update Rhat, phat and vhat based on p


    
        # END implementation of the EqF update step here

        # write back the updated state and covariance into the internal representation 
        self.s.q = quat_utils.Rot2quat(Rhat)
        self.s.p = phat
        self.s.v = vhat

        self.P.P = np.block([[Robs_fault.T @ P @ Robs_fault, np.zeros((9,6))], [np.zeros((6,9)), np.eye(6)]]) 
