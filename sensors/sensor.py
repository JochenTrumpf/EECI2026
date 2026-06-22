from abc import ABC, abstractmethod

from robots.robot_state import robot_state

class sensor(ABC):
    """
    Abstract baseclass for all sensor objects. Callable interface:
     - __init__(params)
        + set sensor parameters self.params to params

     - get_interval(var) -> int
        + returns the time between subsequent measurements of variable var expressed in time steps of length dt (dt: runtime attribute)

     - get_var_attributes() -> dict
        + returns a dict with the names of measurement variables as keys and dicts containing descriptive information as values (used for UI)

     - set_runtime_attributes(name, color, runtime_attributes = {})
        + set the sensor's name (string; used for UI, e.g. 'GPS')
        + set the sensor's color (RGB string; used for UI, e.g. 'rgb(0,0,0)' for black)
        + set the sensor attributes that can only be set at runtime (self.runtime_attributes)
          - MUST include required runtime attributes (except possibly dt)
          - dt can be added later with set_dt()

     - set_dt(dt)
        + add the runtime attribute dt (simulation time step)
     
     - measure(s, t_step) -> dict
        + make a measurement for robot state s at global time t = t_step * dt (dt: runtime attribute)

    parameters:
     - freq - default measurement frequency in Hz

    configuration:
     <none>
    
    required runtime attributes:
     - dt - simulation time step

    Re-implement the following nested class method in any subclass to enforce presence of particular attributes
     - parameters.__init__()
        + list the required parameters with type hints, e.g. self.s_a: float

     - configuration.__init__()
        + list the required configuration flags, e.g. self.blinking_lights: bool

    Re-implement the following method in any subclass to return different measurement intervals for different variables 
     - get_interval(var: str)
        + returns the time between subsequent measurements of variable var expressed in time steps of length dt
    
    Implement the following abstract class method in any subclass:
     - get_var_attributes()
        + returns a dict with the names of measured variables as keys and dicts containing descriptive information as values (used for UI)
        + suggested descriptive information dict keys and values for variables:
           - 'var_type': 'nD' or 'nD_list'
           - 'var_freq': str; 'lf' or 'hf'
           - 'component_number': integer; e.g. 3 or 2
           - 'component_names': tuple of strings; e.g. ('x', 'y', 'z') or ('az', 'el')
           - 'var_title': string; title used for plots of (components/elements of) this variable, e.g. 'GPS position (m)' or 'bearing angle (deg)'
           - 'error_title': string, title used for error plots for this variable, e.g. 'error (m)' or 'error (deg)'
           - 'tab_title': string; used for tab headings for this variable in the UI, e.g. 'GPS position' or 'Landmark bearings'

    Implement the following abstract method in any subclass:
     - measure(s: robot_state, t_step: int)
        + return three dicts m, m_true and m_e
        + m contains variable names as keys and measurement values obtained (not necessarily deterministic) in robot state s at global time t as values
        + m_true contains variable names as keys and true measurement values for the corresponding measurement in m as values
        + m_e contains variable names as keys and scalar measurement error values for the corresponding measurement in m as values
        + use value None in all dicts if no measurement is or can be made of the particular variable 
    """

    class parameters:
        def __init__(self):
            self.freq: float

    class configuration:
        def __init__(self):
            pass
    
    def __init__(self, params: parameters):
        self.params = params

        self.name = '<set name>'
        self.color = ''
        self.runtime_attributes = {}

    def set_runtime_attributes(self, name: str, color: str, runtime_attributes: dict):
        self.name = name
        self.color = color
        self.runtime_attributes = runtime_attributes

    def set_dt(self, dt: float):
        self.runtime_attributes['dt'] = dt

    def get_interval(self, var: str):
        if var not in self.get_var_attributes():
            raise SystemExit('get_interval() called for variable {} but sensor {} has no such measurement variable.'.format(var, self.name))
        if 'dt' not in self.runtime_attributes:
            raise SystemExit('get_interval() called but runtime attribute dt was not set for sensor {}.'.format(self.name))

        return round(1.0 / (self.params.freq * self.runtime_attributes['dt']))

    @classmethod
    @abstractmethod
    def get_var_attributes(cls):
        info = {}
        return info

    @abstractmethod
    def measure(self, s: robot_state, t_step: int):
        m = {}
        m_true = {}
        m_e = {}
        return m, m_true, m_e
