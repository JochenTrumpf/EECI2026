from abc import ABC, abstractmethod

from sensors.sensor import sensor

from .robot_state import robot_state

from filters.filter import filter

class robot(ABC):
    """
    Abstract baseclass for all robot objects. 
    
    Callable interface:
     - __init__(params, config, sensors, filters) 
        + set robot parameters self.params to params
        + set robot configuration self.config to config
        + set (immutable) tuple of robot sensors self.sensors to sensors
        + set (immutable) tuple of robot filters self.filters to filters
        + initialize self.T = -1 
        + self.T always contains the largest time step the robot has moved to so far
          (a valid robot state was computed for time steps 0, ..., self.T)
        + initialize state trajectory (if applicable)

     - get_sensor(var)
        + returns the first sensor from self.sensors that measures variable var or None
        
     - get_var_attributes()
        + returns a dict with the names of state variables as keys and dicts containing descriptive information as values

     - set_runtime_attributes(name, color, runtime_attributes)
        + set the robot's name (self.name; string used for UI, e.g. 'robot')
        + set the robot's color (self.color; RGB string used in UI, e.g. 'rgb(0,0,0)' for black)
        + sets the robot attributes that can only be set at runtime (self.runtime_attributes)     
          - MUST include required runtime attributes (except possibly dt)
          - dt can be added later with set_dt()

     - set_dt(dt)
        + add the runtime attribute dt (simulation time step)
        + call set_dt(dt) on all sensors and filters
    
     - initialize(initial_state)
        + set robot state self.s to initial_state
        + set self.initialized to True

     - step() 
        + advance the robot state by one time step
        + self.arrived switches to True once the final target has been reached
        + self.T is adjusted accordingly
        + appends self.s to self.s_traj if self.config.record_state = True
        + no measurement is made

     - record()
        + append self.s to self.s_traj
     
     - measure(t_step = None) 
        + make measurement at time t (default (None) means t_step = self.T)
        + requires self.initialized = True and self.config.record_state = True if used with t_step < self.T
        + return merged measurement/true value/measurement error dicts from all sensors
     
     - precompute_trajectory(initial_state)
        + force config.record_state to True
        + initialize the robot 
        + iteratively execute step() until the final target has been reached
        + no measurements are made
    
     - measure_all() 
        + make consecutive measurements at all time steps 0, ..., self.T
        + requires self.initialized = True and self.config.record_state = True
        + return list of merged measurement/true value/measurement error dicts from all sensors 
        
    parameters:
     <none>
    
    configuration:
     <none>
    
    required runtime attributes:
     - dt - simulation time step

    Update class-level type hints in any subclass 
     - s - robot state
     - s_traj - robot state trajectory
    
    Re-implement __init__() in any subclass 
        + provide updated type hints for self.params and self.config (where applicable) to enforce correct typing

    Re-implement the following nested class methods in any subclass to enforce presence of particular attributes
     - parameters.__init__()
        + list the required parameters with type hints, e.g. self.wayposes: list[util.Pose] or self.prox_lim: float
     - configuration.__init__()
        + list the required configuration flags, e.g. self.blinking_lights: bool

    Implement the following abstract class method in any subclass:
     - get_var_attributes()
        + returns a dict with the names of state variables as keys and dicts containing descriptive information as values (used for UI)
        + usually a pass through of get_var_attributes() of the relevant robot_state class  

    Implement the following abstract methods in any subclass:
     - initialize(initial_state)
        + particularize initial_state to the correct subclass matching the type of robot

     - _update_state(*args) [internal method]
        + change the robot state self.s from the current time step self.T to time step self.T + 1
        + set self.arrived to True once the final target has been reached
        + adjust self.T accordingly
    """

    class parameters:
        def __init__(self):
            pass

    class configuration:
        def __init__(self):
            self.record_state: bool
    
    s: robot_state
    s_traj: list[robot_state]

    def __init__(self, params: parameters, config: configuration, sensors: tuple[sensor, ...], filters: tuple[filter, ...]):
        # intialize time parameters
        self.arrived = False
        self.T = -1

        # intialize robot parameters and configuration
        self.params = params
        self.config = config

        # set sensor suite
        self.sensors = sensors

        # set list of filters
        self.filters = filters

        # no initial state
        self.initialized = False
    
        # state trajectory
        self.precomputed = False

        # initialize runtime attributes
        self.name = '<set name>'
        self.color = ''
        self.runtime_attributes = {}

    def get_sensor(self, var: str):
        for ss in self.sensors:
            if var in ss.get_var_attributes():
                return ss
            
        return None

    def set_runtime_attributes(self, name: str, color: str, runtime_attributes: dict):
        self.name = name
        self.color = color
        self.runtime_attributes = runtime_attributes

    def set_dt(self, dt: float):
        self.runtime_attributes['dt'] = dt
        for ss in self.sensors:
            ss.set_dt(dt)
        for f in self.filters:
            f.set_dt(dt)

    def step(self):
        if not self.initialized:
            raise SystemExit('step() called but robot {} was not initialized.'.format(self.name))
        self.arrived = self._update_state()
        self.T = self.T + 1

    def record(self):
        if not self.initialized:
            raise SystemExit('record() called but robot {} was not initialized.'.format(self.name))
        self.s_traj.append(self.s.copy())

    def measure(self):
        if not self.initialized:
            raise SystemExit('measure() called but robot {} was not initialized.'.format(self.name))

        return self._measure(self.s, self.T)
            
    def _measure(self, s: robot_state, t_step: int):
        m = {}
        m_true = {}
        m_e = {}
        for ss in self.sensors:
            m_new, m_true_new, m_e_new = ss.measure(s, t_step)
            m.update(m_new)
            m_true.update(m_true_new)
            m_e.update(m_e_new)

        return m, m_true, m_e

    def precompute_trajectory(self, initial_state: robot_state):
        self.config.record_state = True
        self.initialize(initial_state)
        self.record()
        while not self.arrived:
            self.step()
            self.record()
        self.precomputed = True

    def measure_all(self):
        # check that we have a precomputed trajectory
        if (not self.precomputed):
            raise SystemExit('measure_all() called but trajectory was not precomputed for robot {}. Call precompute_trajectory(initial_state) first.'.format(self.name))

        m_traj: list[dict] = []
        m_true_traj: list[dict] = []
        m_e_traj: list[dict] = []
        for t_step in range(self.T + 1):
            m, m_true, m_e = self._measure(self.s_traj[t_step], t_step)
            m_traj.append(m)
            m_true_traj.append(m_true)
            m_e_traj.append(m_e)

        return m_traj, m_true_traj, m_e_traj

    @classmethod
    @abstractmethod
    def get_var_attributes(cls):
        return robot_state.var_attributes
    
    @abstractmethod
    def initialize(self, initial_state: robot_state):
        self.s = initial_state.copy()
        self.initialized = True

        # initialize state trajectory
        self.s_traj = []

    @abstractmethod
    def _update_state(self):
        # return True once no further robot state update is needed
        return False
