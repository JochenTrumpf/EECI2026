import numpy as np
from util import quat_utils, pose_utils

from .filter_state import phone_filter_state
from .filter_covariance import filter_covariance
from .filter import filter

class Attitude_observer(filter):
    """
    Complementary filter for attitude from IMU measurements

    filter_state (INS_filter_state):
     q - estimate of orientation (quaternion) w.r.t global reference frame expressed in global reference frame; np.array([[1.],[0.],[0.],[0.]]) is identity
     eul - q in Euler angles (roll, pitch, yaw); np.array([[0.], [0.], [0.]]) is identity
    
    filter_covariance (filter_covariance):
     <none>

    parameters:
     k_a - gain for accelerometer measurement
     k_m - gain for magnetometer measurement 
    
     g - estimate of gravity vector expressed in global reference frame
     m - estimate of magnetic field vector expressed in global reference frame

    configuration:
     <none>

    required runtime attributes:
     - inherited from parent class (filter)
        + dt - simulation time step

    Expected measurements:
     + high frequency
       w - measured angular velocity w.r.t global reference frame expressed in body-fixed frame
       y_a - normalized accelerometer measurement vector
       y_m - normalized magnetometer measurement vector
    """

    class parameters(filter.parameters):
        def __init__(self):
            self.k_a: float
            self.k_m: float

            self.g: np.typing.NDArray
            self.m: np.typing.NDArray

            super().__init__()

    s: phone_filter_state
    s_traj: list[phone_filter_state]
    P: filter_covariance
    P_traj: list[filter_covariance]

    def __init__(self, params: Attitude_observer.parameters, config: filter.configuration):
        # override parent class types
        self.params: Attitude_observer.parameters
        self.config: filter.configuration
        
        super().__init__(params, config)

    @classmethod
    def get_var_attributes(cls):
        return phone_filter_state.var_attributes 
    
    def initialize(self, initial_state: phone_filter_state, initial_covariance: filter_covariance):
        super().initialize(initial_state, initial_covariance)

    def compute_state_error(self, true_state: phone_filter_state, accumulate: bool):
        super().compute_state_error(true_state, accumulate)

    def predict(self, m: dict):
        if 'dt' not in self.runtime_attributes:
            raise SystemExit('predict() called but runtime attribute dt was not set for filter {}.'.format(self.name))
        # predict does nothing as we implement all of the observer as an update step

    def update(self, m: dict, t_step: int, *_):
        # tell pylance to shut up about the fact that t_step is unused in this implementation
        del t_step 

        # check that we have the required measurements, if not, do nothing
        if ('w' not in m) or ('y_a' not in m) or ('y_m' not in m):
            return
        if (m['w'] is None) or (m['y_a'] is None) or (m['y_m'] is None):
            return
        
        # observer implementation starts here
        Rhat = quat_utils.quat2Rot(self.s.q)    # observer state
        w = m['w']                              # gyro measurement
        acc = m['y_a']                          # normalized accelerometer measurement
        mag = m['y_m']                          # normalized magnetometer measurement


        # observer implementation ends here

        # convert back to internal formats
        self.s.q = quat_utils.Rot2quat(Rhat)
        self.s.eul = np.array([[np.atan2(Rhat[2,1], Rhat[2,2])], [- np.asin(Rhat[2,0])], [np.atan2(Rhat[1,0],Rhat[0,0])]])
