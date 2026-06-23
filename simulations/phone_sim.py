from typing import get_type_hints
import math
import numpy as np
from scipy import linalg

from robots.robot_state import phone_state
from robots.phone import phone

from filters.filter_covariance import filter_covariance
from filters.Attitude_observer import Attitude_observer

from util import quat_utils as util

from .sim import sim

class phone_sim(sim):
    """
    Fixed frequency simulation of a simple_quad (or compatible) robot making GPS/INS measurements

    Additional parameter for __init__():
     - __init__(params, config, robot_data, precompute_robot_trajectories)
        + robot_data, list of pairs of the form (robot, robot initial state)
        + if config.precompute_robot_trajectories is set to true, __init__() calls precompute_trajectory() on every robot

    Callable interface:
     - sim_run(run_id, enable_replay)
       + run one instance of the simulation
       + randomly initialize each robot's filters using the given simulation parameters
       + move robots (if config.precompute_robot_trajectories is set to False), compute sensor measurements and run robot filters
       + compute global error measures 'e_N' and 'e_avg'
       + return result_data dict as configured
         - run_id - unique integer identifier for this simulation run 
            + result_data['run_id']
         - s_traj - robot trajectories indexed by time step t 
            + included if config.precompute_robot_trajectories is set to False
            + result_data['s_traj'][robot_idx][t].var
         - s0s - list of tuples of filter initial states for each robot and each filter 
            + included if config.record_inival or enable_replay is set to True  
            + result_data['s0s'][robot_idx][filter_idx].var
         - P0s - list of tuples of filter initial covariances for each robot and each filter
            + included if config.record_inival or enable_replay is set to True  
            + result_data['P0s'][robot_idx][filter_idx].var
         - m_traj, m_true_traj, m_e_traj - sensor measurement/ground truth/error trajectory data indexed by time step t
            + included if config.record_measurement or enable_replay is set to True
            + result_data['m_traj'][robot_idx][t][var]
            + result_data['m_true_traj'][robot_idx][t][var]
            + result_data['m_e_traj'][robot_idx][t][var]

     - replay(result_data, record_pos_error_ellipsoid)
       + read filter initialization data and measurement data from result_data (output from sim_run) 
       + read ground truth robot trajectory data from robots (if config.precompute_robot_trajectories is set to True) or from result_data (else)
       + re-compute the filter trajectories (for plotting)
       + return replay_data dict as configured
         - s_traj, s_true_traj, s_e_traj - filter state estimate/ground truth/error trajectory data indexed by time step t
            + replay_data['s_traj'][robot_idx][filter_idx][t].var
            + replay_data['s_true_traj'][robot_idx][filter_idx][t].var
            + replay_data['s_e_traj'][robot_idx][filter_idx][t][var] 
         - cov_s_traj, cov_q_traj - main-axes scales and error ellipsoid rotation indexed by time step t
            + included if record_pos_error_ellipsoid is set to True
            + replay_data['cov_s_traj'][robot_idx][filter_idx][t]
            + replay_data['cov_q_traj'][robot_idx][filter_idx][t]

    parameters:
     freq - desired simulation frequency (in Hz)

    configuration:
     precompute_robot_trajectories - if set to True, __init__() calls precompute_trajectory on all robots and individual sim_run calls use the precomputed trajectories
                                     if set to False, each call of sim_run recomputes the robot trajectory (useful for closed loop control)
     record_inival - if set to True, sim_run() returns two lists of tuples, result_data['s0'] and result_data['P0'], containing filter initial values for all robots and filters
     record_measurement - if set to True, sim_run() returns three lists result_data['m_traj'], result_data['m_true_traj'] and result_data['m_e_traj'] 
                          containing the sensor measurement, ground truth and error trajectories for each robot
    """

    class parameters(sim.parameters):
        def __init__(self):
            self.freq: float

    class configuration(sim.configuration):
        def __init__(self):
            # add additional configuration flags
            self.precompute_robot_trajectories: bool
            self.record_inival: bool
            self.record_measurement: bool

    def __init__(self, params: phone_sim.parameters, config: phone_sim.configuration, 
                 robot_data: list[tuple[phone, phone_state]]):
        # override parent class types
        self.params: phone_sim.parameters
        self.config: phone_sim.configuration
        
        super().__init__(params, config)
        
        # unpack robot data
        self.robots: list[phone] = []
        self.robot_inivals: list[phone_state] = []
        for data in robot_data:
            self.robots.append(data[0])
            self.robot_inivals.append(data[1])

        # compute requested simulation time step and ensure that all sensors are compatible
        self.dt = 1.0 / self.params.freq
        for robot in self.robots:
            for ss in robot.sensors:
                r: float = self.params.freq / ss.params.freq
                if not math.isclose(r, round(r)):
                    raise SystemExit('Requested simulation frequency {} is not an integer multiple of the frequency {} of sensor {}.'.format(self.params.freq, ss.params.freq, ss.name))

        # set runtime attribute dt for all robots
        for robot in self.robots:
            robot.set_dt(self.dt)

        # precompute robot trajectories if needed
        if self.config.precompute_robot_trajectories:
            for r_idx, robot in enumerate(self.robots):
                robot.precompute_trajectory(self.robot_inivals[r_idx])

    @classmethod
    def get_var_attributes(cls):
        info = super().get_var_attributes()

        return info
    
    def sim_run(self, run_id: int, enable_replay: bool):
        # initialize result data dict
        result_data = {}
        result_data['run_id'] = run_id

        if self.config.record_inival or enable_replay:
            result_data['s0s'] = []
            result_data['P0s'] = []

        if not self.config.precompute_robot_trajectories:
            result_data['s_traj'] = []

        if self.config.record_measurement or enable_replay:
            result_data['m_traj'] = []    
            result_data['m_true_traj'] = []    
            result_data['m_e_traj'] = []    

        # initialize robots if robot trajectories were not precomputed
        # record robot initial states
        for r_idx, robot in enumerate(self.robots):
            if not self.config.precompute_robot_trajectories:
                robot.initialize(self.robot_inivals[r_idx])
                robot.record()

        # make sensor measurements as appropriate, initialize filters, record filter initial data and 
        # compute initial filter errors
        m_traj_l = []
        m_true_traj_l = []
        m_e_traj_l = []
 
        for r_idx, robot in enumerate(self.robots):
            # make sensor measurements
            if self.config.precompute_robot_trajectories:
                m_traj, m_true_traj, m_e_traj = robot.measure_all()
                m_traj_l.append(m_traj)
                m_true_traj_l.append(m_true_traj)
                m_e_traj_l.append(m_e_traj)
            else:
                m, m_true, m_e = robot.measure()
                m_traj_l.append([m])
                m_true_traj_l.append([m_true])
                m_e_traj_l.append([m_e])

            s0s, P0s = self._get_filter_initial_data(r_idx)

            # record filter initial data 
            if self.config.record_inival or enable_replay:
                result_data['s0s'].append(s0s)
                result_data['P0s'].append(P0s)

            for f_idx, (f, s0, P0) in enumerate(zip(robot.filters, s0s, P0s)):
                # initialize filters
                f.initialize(s0, P0)

        # step robots, record robot state and make sensor measurements unless robot trajectories were precomputed, 
        # run filters and compute filter errors
        t = 0
        done = False
        while not done:
            # increment time step
            t = t + 1

            # step robots, record robot states and make measurements if needed
            done = True
            if self.config.precompute_robot_trajectories:
                for robot in self.robots:
                    if t < robot.T:
                        done = False
            else:
                for robot in self.robots:
                    robot.step()
                    if not robot.arrived:
                        done = False
                    robot.record()
                    m, m_true, m_e = robot.measure()
                    m_traj_l[r_idx].append(m)
                    m_true_traj_l[r_idx].append(m_true)
                    m_e_traj_l[r_idx].append(m_e)

            # run filters and and compute filter errors
            for r_idx, robot in enumerate(self.robots):
                for f_idx, f in enumerate(robot.filters):
                    f.predict(m_traj_l[r_idx][t])
                    f.update(m_traj_l[r_idx][t], t)

        # record state trajectories and waypose steps for robots if needed
        if not self.config.precompute_robot_trajectories:
            for robot in self.robots:
                result_data['s_traj'].append(robot.s_traj)
        
        # record sensor measurements
        if self.config.record_measurement or enable_replay:
            result_data['m_traj'] = m_traj_l
            result_data['m_true_traj'] = m_true_traj_l
            result_data['m_e_traj'] = m_e_traj_l

        return result_data

    def replay(self, result_data: dict, record_pos_error_ellipsoid: bool):
        # check that result_data contains initial values s0 and P0 and measurements m
        if  (not 's0s' in result_data) or (not 'P0s' in result_data) or (not 'm_traj' in result_data):
            raise SystemExit('result_data must contain initial values and measurements to use replay(). Re-run sim_run() with enable_replay set to True.')

        # check that result_data contains robot state trajectories if robot trajectories were not precomputed
        if (not 's_traj' in result_data) and (not self.config.precompute_robot_trajectories):
            raise SystemExit('result_data must contain recorded robot trajectories if they have not been precomputed to use replay().')

        # initialize replay data dict
        replay_data = {}
        replay_data['s_traj'] = []
        replay_data['s_true_traj'] = []
        replay_data['s_e_traj'] = []

        if record_pos_error_ellipsoid:
            replay_data['cov_s_traj'] = []
            replay_data['cov_q_traj'] = []

        for r_idx, robot in enumerate(self.robots):
            # initialize filters and compute initial filter state error
            s_traj_l = []
            s_true_traj_l = []
            s_e_traj_l = []
            if record_pos_error_ellipsoid:
                cov_s_traj_l = []
                cov_q_traj_l = []

            if self.config.precompute_robot_trajectories:
                s_traj = robot.s_traj 
            else:
                s_traj = result_data['s_traj'][r_idx]

            for f_idx, f in enumerate(robot.filters):
                f.initialize(result_data['s0s'][r_idx][f_idx], result_data['P0s'][r_idx][f_idx])
                s_traj_l.append([f.s.copy()])
                true_state = self._get_true_state(r_idx, f_idx, 0, s_traj[0])
                s_true_traj_l.append([true_state])
                f.compute_state_error(true_state, accumulate = False)
                s_e_traj_l.append([f.se.copy()])
                if record_pos_error_ellipsoid:
                    cov_s_traj_l.append([np.array([[0.], [0.], [0.]])])
                    cov_q_traj_l.append([np.array([[1.], [0.], [0.],[0.]])])

            # run filters, record filter trajectories and compute filter state errors
            for t in range(1, robot.T + 1 if self.config.precompute_robot_trajectories else len(result_data['s_traj'][r_idx])):
                for f_idx, f in enumerate(robot.filters):
                    f.predict(result_data['m_traj'][r_idx][t])
                    f.update(result_data['m_traj'][r_idx][t], t)
                    s_traj_l[f_idx].append(f.s.copy())
                    true_state = self._get_true_state(r_idx, f_idx, t, s_traj[t])
                    s_true_traj_l[f_idx].append(true_state)
                    f.compute_state_error(true_state, accumulate = False)
                    s_e_traj_l[f_idx].append(f.se.copy())
                    if record_pos_error_ellipsoid:
                        cov_s_traj_l[f_idx].append(np.array([[0.], [0.], [0.]]))
                        cov_q_traj_l[f_idx].append(np.array([[1.], [0.], [0.],[0.]]))

            replay_data['s_traj'].append(tuple(s_traj_l))
            replay_data['s_true_traj'].append(tuple(s_true_traj_l))
            replay_data['s_e_traj'].append(tuple(s_e_traj_l))
            if record_pos_error_ellipsoid:
                replay_data['cov_s_traj'].append(tuple(cov_s_traj_l))
                replay_data['cov_q_traj'].append(tuple(cov_q_traj_l))

        return replay_data

    def _get_true_state(self, r_idx: int, f_idx: int, t: int, s: phone_state):
        robot = self.robots[r_idx]
        f = robot.filters[f_idx]
        filter_state_type = get_type_hints(f.__class__).get('s')
        true_state = filter_state_type.default() # type: ignore

        for var in true_state.__dict__:
            if var in ['q', 'eul']:
                    value = getattr(s, var)
            else:
                raise SystemExit('_get_true_state() called for unknown variable name {}.'.format(var))
            setattr(true_state, var, value)

        return true_state 

    def _get_filter_initial_data(self, r_idx):
            robot = self.robots[r_idx]

            # initial values for state variables and covariance variables
            s_ini_vals = {
                'q': robot.s_traj[0].q,
                'eul': robot.s_traj[0].eul
            }

            # determine appropriate initial filter states and covariances
            s0_l = []
            P0_l = []

            for f in robot.filters:
                # initial state
                filter_state_type = get_type_hints(f.__class__).get('s')
                s0 = filter_state_type.default() # type: ignore

                for var in s0.__dict__:
                    if var in ['q', 'eul']:
                        setattr(s0, var, s_ini_vals[var])
                    else:
                        raise SystemExit('Encountered unknown state variable name {} in _get_filter_initial_data().'.format(var))

                s0_l.append(s0)

                # initial covariance
                if f.__class__ in [Attitude_observer]:
                    P0 = filter_covariance()
                else:
                    raise SystemExit('Encountered unknown filter type {} in _get_filter_initial_data().'.format(f.__class__))
                
                P0_l.append(P0)

            return tuple(s0_l), tuple(P0_l)
