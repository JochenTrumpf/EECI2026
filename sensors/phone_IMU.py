import numpy as np

from robots.robot_state import robot_state

from .sensor import sensor

class phone_IMU(sensor):
    """
    Fake sensor getting it's data from data arrays provided as sensor parameters
    !!! Because we have no ground truth, returns ground truth = measurement and error = 0
    
    parameters:
     - inherited from parent class (sensor)
        + freq - default measurement frequency (in Hz)

     - additional parameters
        + gyr - list of gyroscope measurements
        + acc - list of normalized accelerometer measurements
        + mag - list of normalized magnetometer measurements

    required robot_state class:
     robot_state

    required runtime attributes:
     - inherited from parent class (sensor)
        + dt - simulation time step
     
    high frequency return values (a, w):
     w, w_true, w_e - angular velocity measurement/ground truth/measurement error w.r.t global reference frame expressed in body-fixed frame
     y_a, y_a_true; y_a_e - normalized accelerometer measurement vector; ground truth negative gravity vector; measurement error 
                            [approx. measurement of negative of gravity vector in hover flight]
     y_m, y_m_true, y_m_e - normalized magnetometer measurement/ground truth magnetic field vector; measurement error
    """

    class parameters(sensor.parameters):
        def __init__(self):
            self.gyr: list[np.typing.NDArray]
            self.acc: list[np.typing.NDArray]
            self.mag: list[np.typing.NDArray]

            super().__init__()

    def __init__(self, params: phone_IMU.parameters):
        # override parent class type
        self.params: phone_IMU.parameters

        super().__init__(params)

    @classmethod
    def get_var_attributes(cls):
        info = super().get_var_attributes()
        
        # information for additional measured variables
        info.update({
            'w': {
                'var_type': 'nD',
                'var_freq': 'hf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'gyro (rad/s)',
                'error_title': 'gyro error (rad/s)',
                'tab_title': 'Gyroscope'
            },
            'y_a': {
                'var_type': 'nD',
                'var_freq': 'hf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'neg grav dir',
                'error_title': 'grav error (deg)',
                'tab_title': 'Gravity (Accel)'
            },
            'y_m': {
                'var_type': 'nD',
                'var_freq': 'hf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'magn dir',
                'error_title': 'magn error (deg)',
                'tab_title': 'Magnetometer'
            }
        })

        return info 

    def measure(self, s: robot_state, t_step: int):        
        interval = self.get_interval('w')

        if t_step % interval  == 0:
            # read measurements from data arrays
            w_true = self.params.gyr[int(t_step / interval)]
            w_e = 0.0
            w = w_true

            y_a_true = self.params.acc[int(t_step / interval)]
            y_a_e = 0.0
            y_a = y_a_true

            y_m_true = self.params.mag[int(t_step / interval)]
            y_m_e = 0.0
            y_m = y_m_true            
        else:
            w_true = None
            w_e = None
            w = None

            y_a_true = None
            y_a_e = None
            y_a = None

            y_m_true = None
            y_m_e = None
            y_m = None 

        m = {
            'w': w,
            'y_a': y_a,
            'y_m': y_m
        }

        m_true = {
            'w': w_true,
            'y_a': y_a_true,
            'y_m': y_m_true
        }

        m_e = {
            'w': w_e,
            'y_a': y_a_e,
            'y_m': y_m_e
        }

        return m, m_true, m_e
