import numpy as np
from util import quat_utils, pose_utils

from sensors.sensor import sensor

from .robot_state import phone_state
from .robot import robot

from filters.filter import filter

class phone(robot):
    """
    mobile phone rotating in 3D space measuring IMU data
    !!! We are using the IMU data provided as robot parameters to do a direct reconstruction
    !!! of attitude that is presented as "ground truth" 

    robot_state (robot_state):
     q - orientation (quaternion) w.r.t global reference frame expressed in global reference frame; np.array([[1.],[0.],[0.],[0.]]) is identity
     eul - q in Euler angles (roll, pitch, yaw); np.array([[0.], [0.], [0.]]) is the identity

    parameters:
     g - gravity vector expressed in global reference frame

     gyr - list of gyroscope measurements
     acc - list of normalized accelerometer measurements
     mag - list of normalized magnetometer measurements

    configuration:
     <none>

    required runtime attributes:
     - inherited from parent class (robot)
        + dt - simulation time step

     - new runtime attribute
        + max_t - length of state trajectory
    """

    class parameters(robot.parameters):
        def __init__(self):
            self.g: np.typing.NDArray
            self.gyr: list[np.typing.NDArray]
            self.acc: list[np.typing.NDArray]
            self.mag: list[np.typing.NDArray]

            super().__init__()
    
    s: phone_state
    s_traj: list[phone_state]

    def __init__(self, params: phone.parameters, config: robot.configuration, sensors: tuple[sensor, ...], filters: tuple[filter, ...]):
        # override parent class types
        self.params: phone.parameters
        
        super().__init__(params, config, sensors, filters)

        self.max_t = len(self.params.gyr) - 1

    @classmethod
    def get_var_attributes(cls):
        return phone_state.var_attributes
    
    def initialize(self, initial_state: phone_state):
        super().initialize(initial_state)
        R = self._reconstruct_R(0)
        self.s.q = quat_utils.Rot2quat(R)
        self.s.eul = np.array([[np.atan2(R[2,1], R[2,2])], [- np.asin(R[2,0])], [np.atan2(R[1,0],R[0,0])]])

    def _update_state(self):
        if 'dt' not in self.runtime_attributes:
            raise SystemExit('_update_state() called but runtime attribute dt was not set for robot {}.'.format(self.name))

        R = self._reconstruct_R(self.T + 1)
        self.s.q = quat_utils.Rot2quat(R)
        self.s.eul = np.array([[np.atan2(R[2,1], R[2,2])], [- np.asin(R[2,0])], [np.atan2(R[1,0],R[0,0])]])

        # stop once we have run out of sensor data
        done = (self.T >= self.max_t - 1)

        return done

    def _reconstruct_R(self, t_step: int):
        # The i-th column of the attitude matrix corresponds to the i-th inertial basis vector 
        # expressed in the body-fixed frame
        #
        # Since the gravity vector is -e3, we can treat the normalized accelerometer measurement 
        # "acc" as the third column of the attitude matrix
        #
        # Considering that the earth's magnetic field vector points into the north reference
        # direction, the normalized magnetometer measurement "mag" projected onto a plane
        # orthogonal to "acc" will then correspond to the first column of our matrix
        #
        # The second column is chosen to produce a rotation matrix

        acc = self.params.acc[t_step]
        mag = self.params.mag[t_step]

        r1 = mag - (mag.T @ acc) * acc  # project
        r1 = r1 / np.linalg.norm(r1)    # renormalize

        r2 = np.cross(acc, mag, axis = 0)

        return np.column_stack([r1, r2, acc])