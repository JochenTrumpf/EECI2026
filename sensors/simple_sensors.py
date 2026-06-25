import math
import numpy as np
from util.quat_utils import quat2Rot

from .sensor import sensor

from robots.robot_state import INS_robot_state

class base_IMU(sensor):
    """
    High frequency 3-axis accelerometer and 3-axis gyroscope
    
    parameters:
     - inherited from parent class (sensor)
        + freq - default measurement frequency (in Hz)
    
     - additional parameters
        + s_a - uncertainty of accelerometer measurement a_m in each component (std.dev. in m/s^2)
        + s_w - uncertainty of gyro measurement w_m in each component (std.dev. in rad/s)

    required robot_state class:
     INS_robot_state

    required runtime attributes:
     - inherited from parent class (sensor)
        + dt - simulation time step

     - additional runtime attributes
        + g - gravity vector
     
    high frequency return values (a, w):
     a, a_true, a_e - linear acceleration measurement/ground truth/measurement error w.r.t global reference frame expressed in body-fixed frame
     w, w_true, w_e - angular velocity measurement/ground truth/measurement error w.r.t global reference frame expressed in body-fixed frame
    """

    class parameters(sensor.parameters):
        def __init__(self):
            self.s_a: float
            self.s_w: float

            super().__init__()

    def __init__(self, params: base_IMU.parameters):
        # override parent class type
        self.params: base_IMU.parameters

        super().__init__(params)

    @classmethod
    def get_var_attributes(cls):
        info = super().get_var_attributes()
        
        # information for additional measured variables
        info.update({
            'a': {
                'var_type': 'nD',
                'var_freq': 'hf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'accel (m/s^2)',
                'error_title': 'accel error (m/s^2)',
                'tab_title': 'Accelerometer'
            },
            'w': {
                'var_type': 'nD',
                'var_freq': 'hf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'gyro (rad/s)',
                'error_title': 'gyro error (rad/s)',
                'tab_title': 'Gyroscope'
            }
        })

        return info 

    def measure(self, s: INS_robot_state, t_step: int):        
        if t_step % self.get_interval('a') == 0:
            # simulate accelerometer and gyro measurements
            a_true = s.a
            a_offset = s.ab + self.params.s_a * np.random.randn(3,1)
            a_e = np.linalg.norm(a_offset)
            a = a_true + a_offset

            w_true = s.w
            w_offset = s.wb + self.params.s_w * np.random.randn(3,1)
            w_e = np.linalg.norm(w_offset)
            w = w_true + w_offset
        else:
            a_true = None
            a_e = None
            a = None

            w_true = None
            w_e = None
            w = None 
            
        m = {
            'a': a,
            'w': w
       }

        m_true = {
            'a': a_true,
            'w': w_true
        }

        m_e = {
            'a': a_e,
            'w': w_e
        }

        return m, m_true, m_e

class simple_IMU(sensor):
    """
    High frequency 3-axis accelerometer and 3-axis gyroscope
    Lower frequency magnetometer
    
    parameters:
     - inherited from parent class (sensor)
        + freq - default measurement frequency (in Hz)
                 MUST be an integer multiple of freq.m 
    
     - additional parameters
        + s_a - uncertainty of accelerometer measurement a_m in each component (std.dev. in m/s^2)
        + s_w - uncertainty of gyro measurement w_m in each component (std.dev. in rad/s)

        + freq_m - measurement frequency of magnetometer (in Hz)
        + s_az - uncertainty of magnetometer azimuth measurement az_m (std.dev. in rad)
        + s_el - uncertainty of magnetometer elevation measurement el_m (std.dev. in rad)

    required robot_state class:
     INS_robot_state

    required runtime attributes:
     - inherited from parent class (sensor)
        + dt - simulation time step

     - additional runtime attributes
        + g - gravity vector
        + m_az - magnetic field azimuth
        + m_el - magnetic field elevation
     
    high frequency return values (a, w):
     a, a_true, a_e - linear acceleration measurement/ground truth/measurement error w.r.t global reference frame expressed in body-fixed frame
     w, w_true, w_e - angular velocity measurement/ground truth/measurement error w.r.t global reference frame expressed in body-fixed frame

    low frequency return values (y_a, y_m)
     y_a, y_a_true; y_a_e - normalized accelerometer measurement vector; ground truth negative gravity vector; measurement error 
                            [approx. measurement of negative of gravity vector in hover flight]
     y_m, y_m_true, y_m_e - normalized magnetometer measurement/ground truth magnetic field vector; measurement error
    """

    class parameters(sensor.parameters):
        def __init__(self):
            self.s_a: float
            self.s_w: float

            self.freq_m: float
            self.s_az: float
            self.s_el: float

            super().__init__()

    def __init__(self, params: simple_IMU.parameters):
        # override parent class type
        self.params: simple_IMU.parameters

        super().__init__(params)

        # check that self.params.freq is an integer multiple of self.params.freq_m
        r: float = self.params.freq / self.params.freq_m
        if not math.isclose(r, round(r)):
            raise SystemExit('params.freq MUST be an integer multiple of params.freq for a simple_IMU sensor.')

    def get_interval(self, var: str):
        if (var == 'y_a') or (var == 'y_m'):
            if 'dt' not in self.runtime_attributes:
                raise SystemExit('get_interval() called but runtime attribute dt was not set for sensor {}.'.format(self.name))
            
            return int(1.0/(self.params.freq_m * self.runtime_attributes['dt']))
        else:
            return super().get_interval(var)

    @classmethod
    def get_var_attributes(cls):
        info = super().get_var_attributes()
        
        # information for additional measured variables
        info.update({
            'a': {
                'var_type': 'nD',
                'var_freq': 'hf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'accel (m/s^2)',
                'error_title': 'accel error (m/s^2)',
                'tab_title': 'Accelerometer'
            },
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
                'var_freq': 'lf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'neg grav dir',
                'error_title': 'grav error (deg)',
                'tab_title': 'Gravity (Accel)'
            },
            'y_m': {
                'var_type': 'nD',
                'var_freq': 'lf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'magn dir',
                'error_title': 'magn error (deg)',
                'tab_title': 'Magnetometer'
            }
        })

        return info 

    def measure(self, s: INS_robot_state, t_step: int):        
        if t_step % self.get_interval('a') == 0:
            # simulate accelerometer and gyro measurements
            a_true = s.a
            a_offset = s.ab + self.params.s_a * np.random.randn(3,1)
            a_e = np.linalg.norm(a_offset)
            a = a_true + a_offset

            w_true = s.w
            w_offset = s.wb + self.params.s_w * np.random.randn(3,1)
            w_e = np.linalg.norm(w_offset)
            w = w_true + w_offset
        else:
            a_true = None
            a_e = None
            a = None

            w_true = None
            w_e = None
            w = None 
            
        if t_step % self.get_interval('y_m') == 0:
            # normalized accelerometer measurement vector
            if 'g' not in self.runtime_attributes:
                raise SystemExit('measure() called for variable y_a but runtime attribute g was not set for sensor {}.'.format(self.name))
            if (a is None) or (a_true is None):
                y_a_true = None
                y_a_e = None
                y_a = None
            else:
                y_a_true = - self.runtime_attributes['g']
                y_a_true /= np.linalg.norm(y_a_true)
                y_a = (a)/np.linalg.norm(a)
                y_a_e = np.rad2deg(np.arccos(y_a_true.T @ y_a))[0, 0]
        
            # normalized magnetometer measurement vector
            if ('m_az' not in self.runtime_attributes) or ('m_el' not in self.runtime_attributes):
                raise SystemExit('measure() called for variable y_m but runtime attributes m_az and m_el were not set for sensor {}.'.format(self.name))

            m_az = self.runtime_attributes['m_az']
            m_el = self.runtime_attributes['m_el']

            y_m_true = np.array([[np.cos(m_el)*np.cos(m_az)],[np.cos(m_el)*np.sin(m_az)],[np.sin(m_el)]])
            az_m = m_az + self.params.s_az * np.random.normal()
            el_m = m_el + self.params.s_el * np.random.normal()
            y_m = np.array([[np.cos(el_m)*np.cos(az_m)],[np.cos(el_m)*np.sin(az_m)],[np.sin(el_m)]])
            y_m_e = np.rad2deg(np.arccos(y_m_true.T @ y_m))[0, 0]
        else:
            y_a_true = None
            y_a_e = None
            y_a = None
            
            y_m_true = None
            y_m_e = None
            y_m = None

        m = {
            'a': a,
            'w': w,
            'y_a': y_a,
            'y_m': y_m
        }

        m_true = {
            'a': a_true,
            'w': w_true,
            'y_a': y_a_true,
            'y_m': y_m_true
        }

        m_e = {
            'a': a_e,
            'w': w_e,
            'y_a': y_a_e,
            'y_m': y_m_e
        }

        return m, m_true, m_e
        
class simple_GPS(sensor):
    """
    GPS position

    parameters:
     - inherited from parent class (sensor)
        + freq - default measurement frequency in Hz 
    
     - additional parameters
        + s_gps - uncertainty of GPS position measurement y_gps in each component (std.dev. in m)

    required robot_state class:
     INS_robot_state

    required runtime attributes:
     - inherited from parent class (sensor)
        + dt - simulation time step

    low frequency return value (p):
     p/p_true/p_e - position measurement/ground truth/measurement error w.r.t global reference frame expressed in global reference frame
    """

    class parameters(sensor.parameters):
        def __init__(self):
            self.s_gps: float

            super().__init__()

    def __init__(self, params: simple_GPS.parameters):
        # override parent class type
        self.params: simple_GPS.parameters

        super().__init__(params)

    @classmethod
    def get_var_attributes(cls):
        info = super().get_var_attributes()
        
        # information for additional measured variables
        info.update({
            'p': {
                'var_type': 'nD',
                'var_freq': 'lf',
                'component_number': 3,
                'component_names': ('x', 'y', 'z'),
                'var_title': 'GPS position (m)',
                'error_title': 'GPS error (m)',
                'tab_title': 'GPS position'
            }
        })

        return info 

    def measure(self, s: INS_robot_state, t_step: int):
        if t_step % self.get_interval('p') == 0:
            # simulate GPS position measurement
            p_true = s.p
            p_offset = self.params.s_gps * np.random.randn(3,1)
            p_e = np.linalg.norm(p_offset)
            p = p_true + p_offset
        else:
            p_true = None
            p_e = None
            p = None

        m = {
            'p': p
        }

        m_true = {
            'p': p_true
        }

        m_e = {
            'p': p_e
        }

        return m, m_true, m_e 
    
class simple_lb_sensor(sensor):
    """
    Azimuth and elevation angle of relative bearings to stationary landmarks

    parameters:
     - inherited from parent class (sensor)
        + freq - default measurement frequency in Hz 
    
     - additional parameters
        + s_az - uncertainty of landmark azimuth measurement (std.dev. in rad)
        + s_el - uncertainty of landmark elevation measurement (std.dev. in rad)

    required robot_state class:
     INS_robot_state

    required runtime attributes:
     - inherited from parent class (sensor)
        + dt - simulation time step

     - additional runtime attributes
        + l - landmark collection
        + b_l - names, colors and indices of measured landmarks in landmark collection l

    low frequency return value (b_l):
     b_l, b_l_true, b_l_e - lists of bearing measurements/ground truth/measurement errors to all landmarks in the same order as self.l.ll
     b_l[idx][0], b_l_true[idx][0], b_l_e[idx][0] - azimuth measurement/ground truth/measurement error to landmark self.l.ll[idx]
     b_l[idx][0], b_l_true[idx][0], b_l_e[idx][0] - elevation measurement/ground truth/measurement error to landmark self.l.ll[idx]
    """

    class parameters(sensor.parameters):
        def __init__(self):
            self.s_az: float
            self.s_el: float

            super().__init__()
    
    def __init__(self, params: simple_lb_sensor.parameters):
        # override parent class type
        self.params: simple_lb_sensor.parameters
        
        super().__init__(params)

    @classmethod
    def get_var_attributes(cls):
        info = super().get_var_attributes()
        
        # information for additional measured variables
        info.update({
            'b_l': {
                'var_type': 'nD_list',
                'var_freq': 'lf',
                'component_number': 2,
                'component_names': ('az', 'el'),
                'var_title': 'bearing angle (rad)',
                'error_title': 'error (deg)',
                'tab_title': 'Landmark bearings'
            }
        })

        return info 

    def measure(self, s: INS_robot_state, t_step: int):
        if t_step % self.get_interval('b_l') == 0:
            if 'l' not in self.runtime_attributes:
                raise SystemExit('measure() called but runtime attribute l was not set for sensor {}.'.format(self.name))

            l = self.runtime_attributes['l']

            b_l_true = []
            b_l_e = []
            b_l = []
        
            for i in range(l.n_l):
                li = l.l[3*i:3*(i+1), :]
                
                li_b = quat2Rot(s.q).T @ (li - s.p)
                li_bn = li_b / np.linalg.norm(li_b)
                
                az_true = np.arctan2(li_bn[1, 0], li_bn[0, 0])
                el_true = np.arctan2(li_bn[2, 0], np.linalg.norm(li_bn[0:2, 0:2]))
                b_true = np.array([[az_true], [el_true]])
                b_l_true.append(b_true)
                
                az = az_true + self.params.s_az * np.random.normal(0,1)
                el = el_true + self.params.s_el * np.random.normal(0,1)
                b = np.array([[az], [el]])
                b_l.append(b)

                li_m = np.array([[np.cos(el)*np.cos(az)],[np.cos(el)*np.sin(az)],[np.sin(el)]])
                b_e = np.rad2deg(np.arccos(li_bn.T @ li_m))[0, 0]
                b_l_e.append(b_e)
        else:
            b_l_true = None
            b_l_e = None
            b_l = None

        m = {
            'b_l': b_l
        }

        m_true = {
            'b_l': b_l_true
        }

        m_e = {
            'b_l': b_l_e
        }

        return m, m_true, m_e
