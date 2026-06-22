import numpy as np

class robot_state():
    """
    This is a container class for the state variables of a robot
    
    Static variable:
     - var_attributes
        + dict with the names of state variables as keys and dicts containing descriptive information as values
        + suggested descriptive information dict keys and values for variables that are numpy n x 1 arrays:
           - 'var_type': str; 'nD'
           - 'var_freq': str; 'lf' or 'hf'
           - 'component_number': integer; e.g. 3 or 4
           - 'component_names': tuple of strings; e.g. ('x', 'y', 'z') or ('w', 'x', 'y', 'z')
           - 'var_title': string; title used for plots of (components of) this variable, e.g. 'position (m)' or 'quaternion'
           - 'tab_title': string; used for tab headings for this variable in the UI, e.g. 'Position' or 'Heading'

    Callable interface:
     - copy()
        + return a deep copy of self

     - default()
        + return a new object with default values  
    """

    var_attributes = {}

    def __init__(self):
        pass

    def copy(self):
        return self.__class__()
    
    @classmethod
    def default(cls):
        return cls()

class INS_robot_state(robot_state):

    var_attributes = {
        'p': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'position (m)',
            'tab_title': 'Position'
        },
        'v': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'velocity (m/s)',
            'tab_title': 'Velocity'
        },
        'q': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 4,
            'component_names': ('w', 'x', 'y', 'z'),
            'var_title': 'quaternion',
            'tab_title': 'Heading'
        },
        'ab': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'accel bias (m/s^2)',
            'tab_title': 'Accel bias'
        },
        'wb': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'gyro bias (rad/s)',
            'tab_title': 'Gyro bias'
        },
        'a': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'accel (m/s^2)',
            'tab_title': 'Acceleration'
        },
        'w': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'ang vel (rad/s)',
            'tab_title': 'Ang velocity'
        }
    }

    def __init__(self, p: np.typing.NDArray, v: np.typing.NDArray, q: np.typing.NDArray, ab: np.typing.NDArray, wb: np.typing.NDArray, a: np.typing.NDArray, w: np.typing.NDArray):
        self.p = p
        self.v = v
        self.q = q
        self.ab = ab
        self.wb = wb

        self.a = a
        self.w = w

    def copy(self):
        return self.__class__(self.p, self.v, self.q, self.ab, self.wb, self.a, self.w)

    @classmethod
    def default(cls):
        pos = np.array([[0.],[0.],[0.]])
        quat = np.array([[1.],[0.],[0.],[0.]])
        return cls(pos, pos, quat, pos, pos, pos, pos)
