from abc import ABC, abstractmethod

from .filter_state import filter_state
from .filter_covariance import filter_covariance
from .filter_error import *

class filter(ABC):
    """
    Abstract baseclass for all filter objects. 
    
    Callable interface:
     - __init__(params, config) 
        + set filter parameters self.params to params
        + set filter configuration self.config to config
    
     - get_var_attributes()
        + returns a dict with the names of state variables as keys and dicts containing descriptive information as values

     - set_runtime_attributes(name, color, runtime_attributes)
        + set the filter's name (self.name; string used for UI, e.g. 'EKF')
        + set the filter's color (self.color; RGB string used in UI, e.g. 'rgb(255,0,0)' for red)
        + set the filter attributes that can only be set at runtime (self.runtime_attributes)
          - MUST include required runtime attributes (except possibly dt)
          - dt can be added later with set_dt()

     - set_dt(dt)
        + add the runtime attribute dt (simulation time step)

     - initialize(initial_state, initial_covariance)
        + set filter state self.s to initial_state
        + set filter covariance self.P to initial_covariance
        + initialize state trajectory self.s_traj and covariance trajectory self.P_traj
        + initialize state error self.se to {} and cumulative state error self.cse to zero error

     - compute_state_error(true_state, accumulate)
        + compute the error between self.s and true_state and store the result in self.se
        + if accumulate is set to True, add self.se to self.cse (entry-wise)
        + self.se and self.cse are dictionaries with entries of the form {'<var>': <float>} (error in state variable self.s.<var>) 

     - predict(m) 
        + execute the filter's predict step based on the robot measurement m
     
     - update(m, t_step, *_) 
        + execute the filter's update step at time step t_step based on the robot measurement m
        + *_ argument allows subclasses to require additional arguments, e.g. past_tl_start_idx: bool 
        
     - record(record_state, record_covariance)
        + append self.s to self.s_traj if record_state = True
        + append self.P to self.P_traj if record_covariance = True

    parameters:
     <none>

    configuration:
     <none>
    
    required runtime attributes:
     - dt - simulation time step

    Update class-level type hints in any subclass
     - s - filter state
     - s_traj - filter state trajectory
     - P - filter covariance
     - P_traj - filter covariance trajectory

    Re-implement __init__() in any subclass 
        + provide updated type hints for self.params and self.config (where applicable) to enforce correct typing

    Re-implement the following nested class methods in any subclass to enforce presence of particular attributes
     - parameters.__init__()
        + list the required parameters with type hints, e.g. self.s_a: float or self.g: np.typing.NDArray or self.tl_start_idx: int
     - configuration.__init__()
        + list the required configuration flags, e.g. self.al_first: bool or self.tl_joint: bool

    Implement the following abstract class method in any subclass:
     - get_var_attributes()
        + returns a dict with the names of state variables as keys and dicts containing descriptive information as values (used for UI)
        + usually a pass through of get_var_attributes() of the relevant filter_state class  

    Implement the following abstract methods in any subclass:
     - initialize(initial_state, initial_covariance)
        + particularize initial_state and initial covariance to the correct subclasses matching the type of filter

     - compute_state_error(true_state, accumulate)
        + particularize true_state to the correct subclass matching the type of filter

     - predict(m) 
        + execute the filter's predict step based on the robot measurement m assuming that the time increment is dt (dt: runtime attribute)
     
     - update(m, t_step, *_)
        + execute the filter's update step based on the robot measurement m assuming that the measurement was made at global time t = t_step * dt (dt: runtime attribute)
        + *_ argument allows variable argument list in subclasses
    """

    class parameters:
        def __init__(self):
            pass

    class configuration:
        def __init__(self):
            pass

    s: filter_state
    s_traj: list[filter_state]
    P: filter_covariance
    P_traj: list[filter_covariance]

    def __init__(self, params: parameters, config: configuration):
        # intialize parameters and configuration
        self.params = params
        self.config = config

        # no initial state
        self.initialized = False

        # initialize runtime attributes
        self.name = ''
        self.color = ''
        self.runtime_attributes = {}
            
    def set_runtime_attributes(self, name: str, color: str, runtime_attributes):
        self.name = name
        self.color = color
        self.runtime_attributes = runtime_attributes

    def set_dt(self, dt: float):
        self.runtime_attributes['dt'] = dt

    def record(self, record_state: bool, record_covariance: bool):
        if not self.initialized:
            raise SystemExit('record() called but filter {} was not initialized.'.format(self.name))

        if record_state:
            self.s_traj.append(self.s.copy())
        if record_covariance:
            self.P_traj.append(self.P.copy())

    @classmethod
    @abstractmethod
    def get_var_attributes(cls):
        return filter_state.var_attributes
    
    @abstractmethod
    def initialize(self, initial_state: filter_state, initial_covariance: filter_covariance):
        self.s = initial_state.copy()
        self.P = initial_covariance.copy()
        self.initialized = True

        # initialize state and covariance trajectories
        self.s_traj = []
        self.P_traj = []

        # initialize state error and cumulative state error
        self.se = {}
        self.cse = {}
        for var in self.s.__dict__:
            self.cse[var] = 0.0
    
    @abstractmethod
    def compute_state_error(self, true_state: filter_state, accumulate: bool):
        if not self.initialized:
            raise SystemExit('compute_state_error() called but filter {} was not initialized.'.format(self.name))

        info = self.get_var_attributes()
        for var in self.s.__dict__:
            error_func = info[var]['error_function'] 
            self.se[var] = globals()[error_func](getattr(true_state, var, None), getattr(self.s, var, None))
            if accumulate:
                self.cse[var] += self.se[var]

    @abstractmethod
    def predict(self, m: dict):
        pass

    @abstractmethod
    def update(self, m: dict, t_step: int, *_):
        pass
