import numpy as np

class filter_state():
    """
    This is a container class for the state variables of a filter

    Convention:
     - state variables of the same name MUST have the same variable type (e.g. 'nD' or 'stacked_partial_nD_list')
     - state variables of the same name MUST have the same component number (per element for variables of list type)

    Static variables:
     - var_attributes
        + dict with the names of state variables as keys and dicts containing descriptive information as values
        + suggested descriptive information dict keys and values for variables that are (lists of) numpy n x 1 arrays:
           - 'var_type': str; 'nD', 'nD_list' or 'stacked_partial_nD_list'
           - 'var_freq': str; 'lf' or 'hf'
           - 'component_number': integer; e.g. 3 or 4
           - 'component_names': tuple of strings; e.g. ('x', 'y', 'z') or ('w', 'x', 'y', 'z')
           - 'var_title': string; title used for plots of (components of) this variable, e.g. 'position (m)' or 'quaternion'
           - 'error_function': string; name of error function defined in filters.filter_error.py that can be called to compute a scalar error between two variable instances,
                               e.g. 'l2_error' or 'quat_error' or 'avg_landmark_error'
           - 'error_title': string; title used for error subplots for this variable, e.g. 'error (m)' or 'error (deg)' or 'avg error (m)'
           - 'error_long_title': string; title used for histogram plots for this variable, e.g. 'position error (m)' or 'heading error (deg)' or 'average position error (m)'
           - 'tab_title': string; used for tab headings for this variable in the UI, e.g. 'Position' or 'Heading' or 'Target landmarks'

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

class INS_filter_state(filter_state):

    var_attributes = {
        'p': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'position (m)',
            'error_function': 'l2_error',
            'error_title': 'error (m)',
            'error_long_title': 'position error (m)',
            'tab_title': 'Position'
        },
        'v': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'velocity (m/s)',
            'error_function': 'l2_error',
            'error_title': 'error (m/s)',
            'error_long_title': 'velocity error (m/s)',
            'tab_title': 'Velocity'
        },
        'q': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 4,
            'component_names': ('w', 'x', 'y', 'z'),
            'var_title': 'quaternion',
            'error_function': 'quat_error',
            'error_title': 'error (deg)',
            'error_long_title': 'heading error (deg)',
            'tab_title': 'Heading'
        },
        'ab': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'accel bias (m/s^2)',
            'error_function': 'l2_error',
            'error_title': 'error (m/s^2)',
            'error_long_title': 'accel bias error (m/s^2)',
            'tab_title': 'Accel bias'
        },
        'wb': {
            'var_type': 'nD',
            'var_freq': 'hf',
            'component_number': 3,
            'component_names': ('x', 'y', 'z'),
            'var_title': 'gyro bias (rad/s)',
            'error_function': 'l2_error',
            'error_title': 'error (rad/s)',
            'error_long_title': 'gyro bias error (rad/s)',
            'tab_title': 'Gyro bias'
        }
    }

    def __init__(self, p: np.typing.NDArray, v: np.typing.NDArray, q: np.typing.NDArray, ab: np.typing.NDArray, wb: np.typing.NDArray):
        self.p = p
        self.v = v
        self.q = q
        self.ab = ab
        self.wb = wb

    def copy(self):
        return self.__class__(self.p, self.v, self.q, self.ab, self.wb)

    @classmethod
    def default(cls):
        pos = np.array([[0.],[0.],[0.]])
        quat = np.array([[1.],[0.],[0.],[0.]])
        return cls(pos, pos, quat, pos, pos)
