from abc import ABC, abstractmethod

class sim(ABC):
    """
    Abstract baseclass for all simulation objects. 
    
    Callable interface:
     - __init__(params, config)     
       + set simulation parameters self.params to params
       + set simulation configuration self.config to config

     - get_var_attributes()
        + returns a dict with the names of result variables as keys and dicts containing descriptive information as values (used for UI)
       
     - sim_run(run_id, enable_replay)
       + reentrant function executing a single simulation run with unique id run_id
       + returns result_data dict
       + if enable_replay is set to True, configuration settings are overridden such that result_data is guaranteed to contain the information needed by replay()
    
     - replay(result_data)
       + function that replays a single simulation run based on result_data

    parameters:
     <none>

    configuration:
     <none>
    
    Re-implement __init__() in any subclass 
       + provide updated type hints for self.params and self.config (where applicable) to enforce correct typing
 
    Re-implement the following nested class methods in any subclass to enforce presence of particular attributes
     - parameters.__init__()
        + list the required parameters with type hints, e.g. self.s_p: float or self.s_ab: float
     - configuration.__init__()
        + list the required configuration flags, e.g. self.compute_time_averaged_errors: bool

    Implement the following abstract class method in any subclass:
     - get_var_attributes()
        + returns a dict with the names of result variables as keys and dicts containing descriptive information as values (used for UI)
        + suggested descriptive information dict keys and values for trajectory error variables that are floats:
           - 'var_title': string; title used for plots of this variable, e.g. 'final error' or 'time-averaged error'
           - 'tab_title': string; used for tab headings for this variable in the UI, e.g. 'final error' or 'time-avgd error'
    
    Implement the following abstract methods in any subclass:
     - sim_run(run_id: int, enable_replay: bool)
       + is NOT intended to separately record robot or filter trajectories
       + MAY record initial values, inputs and measurements to enable robot/filter trajectory replay
       + return a single dict containing at least { 'run_id': run_id } and all simulation data needed for summary plots

     - replay(result_data: dict)
       + uses initial values, inputs and measurements from result_data (previously obtained using sim_run())
       + returns robot/filter/ground truth/error trajectories for plotting
    """

    class parameters:
      def __init__(self):
        pass

    class configuration:
      def __init__(self):
        pass
    
    def __init__(self, params: parameters, config: configuration):
        # intialize parameters and configuration
        self.params = params
        self.config = config

    @classmethod
    @abstractmethod
    def get_var_attributes(cls):
        info = {}
        return info
    
    @abstractmethod
    def sim_run(self, run_id: int, enable_replay: bool):
        return {}

    @abstractmethod
    def replay(self, result_data: dict):
        return {}
