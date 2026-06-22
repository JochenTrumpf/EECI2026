from typing import cast

import plotly.graph_objects as go
from dash import dcc, html, Input, Output, callback 

import json
import shelve

from robots.robot_state import robot_state, INS_robot_state
from robots.simple_quad import simple_quad

from sensors.simple_sensors import simple_IMU, simple_GPS

from simulations.test_sim import test_sim

from plots.plotly_plots import plotter
from .app import dash_app

class simple_app(dash_app):
    """
    Supports output from:
     + test_sim
    """

    def __init__(self, data_file: str, downloadable_file_buffer: str, unity_height: int = 750, unity_top_margin: int = 25, url_prefix: str = '/', supply_HTTP_server: bool = True,  
                 HTTP_download_url: str = 'http://localhost:8000/single_run'):
        # dash app
        super().__init__(url_prefix, supply_HTTP_server, HTTP_download_url, True)

        # additional app parameters
        self.data_file = data_file
        self.downloadable_file_buffer = downloadable_file_buffer
        self.unity_height = unity_height
        self.unity_top_margin = unity_top_margin

        # read common data
        data = shelve.open(data_file, flag = 'r')
        self.sim: test_sim = data['sim']
        self.n_MC: int = data['n_MC']
        data.close()

        # these are filled by update_run_tabs
        self.result_data = {}
        self.replay_data = {}

        # make a list of unique filter state variable names and their tab titles for each robot
        self.filter_states = [[] for _ in self.sim.robots]
        self.filter_tab_titles = [[] for _ in self.sim.robots]
        for r_idx, robot in enumerate(self.sim.robots):
            for f in robot.filters:
                for f_var in f.get_var_attributes().keys():
                    if f_var not in self.filter_states[r_idx]:
                        self.filter_states[r_idx].append(f_var)
                        self.filter_tab_titles[r_idx].append(f.get_var_attributes()[f_var]['tab_title'])
    
        # make a list of unique sensor measurement variables and their tab titles for each robot
        self.sensor_vars = [[] for _ in self.sim.robots]
        self.sensor_tab_titles = [[] for _ in self.sim.robots]
        for r_idx, robot in enumerate(self.sim.robots):
            for s in robot.sensors:
                for var in s.get_var_attributes().keys():
                    if var not in self.sensor_vars[r_idx]:
                        self.sensor_vars[r_idx].append(var)
                        self.sensor_tab_titles[r_idx].append(s.get_var_attributes()[var]['tab_title'])

        # plotly plotter
        self.plt = plotter()

        # create app layout and add main tabs to app
        main_tabs = []
        for r_idx in range(len(self.sim.robots)):
            main_tabs.append(self.add_robot_tab(r_idx))
        main_tabs.append(self.add_run_tab())
        
        self.app.layout = html.Div(
            [dcc.Tabs(main_tabs, style = { "white-space": "pre" })]
        )

    def add_robot_tab(self, r_idx: int):
        robot = self.sim.robots[r_idx]
        tab_title = robot.name
        sub_tabs = [
            self._add_error_histogram_tab(r_idx, 'e_N'), 
            self._add_error_histogram_tab(r_idx, 'e_avg')
        ]
        # check if robot contains pre-computed trajectory data
        if robot.precomputed:
            sub_tabs.append(self._add_precomputed_robot_traj_tab(r_idx))

        return self._wrap_sub_tabs(sub_tabs, tab_title)
    
    def add_run_tab(self):
        layout = html.Div(
            id = 'run-div', 
            style = { 'display': 'flex' },
            children = [
                html.Div(
                    [
                        html.Div(
                            id = 'download-link',
                            style = { 'text-align': 'center' },
                            hidden = True        
                        ),
                        dcc.Dropdown(
                            [i for i in range(self.n_MC)],
                            id = 'run-dropdown', 
                            clearable = False
                        )
                    ],
                    style = { 'width': '6%' }
                ),
                html.Div(
                    id = 'run-tabs-div',
                    style = { 'display': 'flex', 'flex-grow': '1' },
                    children = [
                        html.Div(
                            'Select simulation run from the dropdown to the left.',
                            style = { 'padding': '10px' }
                        )
                    ]  
                )
            ]
        )

        spun_layout = dcc.Loading(
            layout,
            target_components = { 'run-tabs-div': 'children'}, # type: ignore
            overlay_style = { 'visibility': 'visible', 'filter': 'blur(2px)' },
            style = { 'align-self': 'start' }
        )

        # callback function that updates the sensor and filter tabs when a run is selected from the dropdown
        @callback(
            Output('run-tabs-div', 'children'),
            Output('download-link', 'children'),
            Output('download-link', 'hidden'),
            Input('run-dropdown', 'value'),
            prevent_initial_call = True
        )
        def update_run_tabs(value: int):
            nonlocal self
            run_id = value

            data = shelve.open(self.data_file, flag = 'r')
            self.result_data = data['result_data[{}]'.format(run_id)]
            data.close()

            sub_tabs = []
            
            robot_tabs = self._make_robot_traj_tabs()
            if isinstance(robot_tabs, list):
                for r_idx in range(len(self.sim.robots)):
                    if robot_tabs[r_idx] is not None:
                        sub_tabs.append(robot_tabs[r_idx])
            else:
                precomputed = False
                for r_idx in range(len(self.sim.robots)):
                    if (self.sim.robots[r_idx].precomputed):
                        precomputed = True
                        break
                if (not precomputed):
                    sub_tabs.append(robot_tabs)
            
            sensor_tabs = self._make_sensor_tabs()
            if isinstance(sensor_tabs, list):
                sub_tabs.extend(sensor_tabs)
            else:
                sub_tabs.append(sensor_tabs)

            filter_tabs = self._make_filter_traj_tabs()
            if isinstance(filter_tabs, list):
                sub_tabs.extend(filter_tabs)
            else:
                sub_tabs.append(filter_tabs)

            sub_tabs.append(self._make_unity_tab())

            tabs = dcc.Tabs(sub_tabs, parent_style = { 'width': '100%' }, style = { "white-space": "pre" })          

            # save the run data to a downloadable file on disk
            run = shelve.open(self.downloadable_file_buffer, flag = 'n')
            run['sim'] = self.sim
            run['result_data'] = self.result_data
            run['replay_data'] = self.replay_data
            run.close()

            # create the suggested filename and the download link,
            # if a custom HTTP server is running, pass the suggested filename to it
            filename = '{}.single_run'.format(run_id)
            link = html.A(
                    'Download run {}'.format(run_id),
                    download = filename,
                    href = self.HTTP_download_url,
                    title = 'Click here to download the run data.'    
                )
            
            if self.HTTP_download_server is not None:
                self.HTTP_download_server.suggested_filename = filename

            # return the new tabs and the download link and unhide the link
            return tabs, link, False
        
        return dcc.Tab(
            spun_layout,
            label = 'Simulation\nruns',
            style = { 'padding': '5px' },
            selected_style = { 'padding': '5px' }
        )

    def _add_error_histogram_tab(self, r_idx: int, error_type: str):
        robot = self.sim.robots[r_idx]
        tab_title = robot.name + '\n' + self.sim.get_var_attributes()[error_type]['tab_title']
        data = shelve.open(self.data_file, flag = 'r')
        
        # check that we have the relevant error data
        if error_type not in data:
            return self._make_error_msg_tab(tab_title, '  data file does not contain error data of type {} for robot {}'.format(error_type, robot.name))
        
        error_data: tuple[list[dict]] = data[error_type][r_idx]
        data.close()

        # one tab per unique filter variable
        sub_tabs = []
        for f_var, sub_tab_title in zip(self.filter_states[r_idx], self.filter_tab_titles[r_idx]):
            # check that we have error_data for this variable
            plot = False
            for f_idx in range(len(error_data)):
                if f_var in error_data[f_idx][0]:
                    plot = True

            if plot:
                sub_tabs.append(self._make_sub_tab(self.plt.plot_error_histogram(robot, self.sim, error_type, f_var, error_data), sub_tab_title))

        # check that we plotted something
        if sub_tabs:
            return self._wrap_sub_tabs(sub_tabs, tab_title)
        else:
            return self._make_error_msg_tab(tab_title, '  error data for robot {} does not contain data of type {} for any of the filter state variables'.format(robot.name, error_type))

    def _add_precomputed_robot_traj_tab(self, r_idx: int):
        robot = self.sim.robots[r_idx]
        tab_title = robot.name + '\ntrajectory'

        # one tab per robot state variable
        sub_tabs = []
        for var in self.sim.robots[r_idx].s.__dict__.keys():
                sub_tabs.append(self._make_sub_tab(self.plt.plot_robot_state(robot, var, self.sim.dt), robot.get_var_attributes()[var]['tab_title'], True))

        return self._wrap_sub_tabs(sub_tabs, tab_title, True)
    
    def _make_robot_traj_tabs(self):
        # check that we have robot trajectory data
        if ('s_traj' not in self.result_data):
            return self._make_error_msg_tab('Robot\nerror', '  result_data[{}] contains no robot trajectory data'.format(self.result_data['run_id']))

        s_traj_l: list[list[INS_robot_state] | None] = self.result_data['s_traj']

        tabs = []
        for r_idx, robot in enumerate(self.sim.robots):
            s_traj: list[INS_robot_state] | None = s_traj_l[r_idx]
            if s_traj is not None:
                tab_title = robot.name + '\ntrajectory'

                # one tab per robot state variable
                sub_tabs = []
                for var in s_traj[0].__dict__.keys():
                        sub_tabs.append(self._make_sub_tab(self.plt.plot_robot_state(robot, var, self.sim.dt, cast(list[robot_state], s_traj)), robot.get_var_attributes()[var]['tab_title'], True))

                tabs.append(self._wrap_sub_tabs(sub_tabs, tab_title, True))
            else:
                tabs.append(None)

        return tabs

    def _make_sensor_tabs(self):
        # check that we have measurement data
        if ('m_traj' not in self.result_data) or ('m_true_traj' not in self.result_data) or ('m_e_traj' not in self.result_data):
            return self._make_error_msg_tab('Sensor\nerror', '  result_data[{}] contains no sensor measurement data'.format(self.result_data['run_id']))

        m_traj_l: list[list[dict]] = self.result_data['m_traj']
        m_true_traj_l: list[list[dict]] = self.result_data['m_true_traj']
        m_e_traj_l: list[list[dict]] = self.result_data['m_e_traj']

        tabs = []
        for r_idx, robot in enumerate(self.sim.robots):
            tab_title = '{}\nsensors'.format(robot.name)
            
            m_traj: list[dict] = m_traj_l[r_idx]
            m_true_traj: list[dict] = m_true_traj_l[r_idx]
            m_e_traj: list[dict] = m_e_traj_l[r_idx]

            # one tab per sensor measurement variable
            sub_tabs = []
            for var, sub_tab_title in zip(self.sensor_vars[r_idx], self.sensor_tab_titles[r_idx]):
                # check that we have measurement data for this variable
                if (var in m_traj[0]) and (var in m_true_traj[0]) and (var in m_e_traj[0]): 
                    sub_tabs.append(self._make_sub_tab(self.plt.plot_measurement(robot, var, self.sim.dt, m_traj, m_true_traj, m_e_traj), sub_tab_title, True))

            # check that we plotted something
            if sub_tabs:
                tabs.append(self._wrap_sub_tabs(sub_tabs, tab_title))
            else:
                tabs.append(self._make_error_msg_tab(tab_title, '  result_data[{}] does not contain measurement data for any of the sensor measurement variables for robot {}'.format(self.result_data['run_id'], robot.name)))

        return tabs

    def _make_filter_traj_tabs(self):
        # check that we have filter initialization data
        if ('s0s' not in self.result_data) or ('P0s' not in self.result_data):
            return self._make_error_msg_tab('Filter\nerror', '  result_data[{}] contains no filter initialization data'.format(self.result_data['run_id']))
        # check that we have sensor measurement data
        if ('m_traj' not in self.result_data):
            return self._make_error_msg_tab('Filter\nerror', '  result_data[{}] contains no sensor measurement data'.format(self.result_data['run_id']))
        # check that result_data contains robot state trajectories if robot trajectories were not precomputed
        if (not 's_traj' in self.result_data) and (not self.sim.config.precompute_robot_trajectories):
            raise SystemExit('Filter\nerror', '  result_data[{}] contains no robot trajectory data and robot trajectories have not been precomputed'.format(self.result_data['run_id']))

        # replay trajectories
        self.replay_data = self.sim.replay(self.result_data, record_pos_error_ellipsoid = True)

        tabs = []
        for r_idx, robot in enumerate(self.sim.robots):
            robot = self.sim.robots[r_idx]
            tab_title = '{}\nfilters'.format(robot.name)

            # one tab per filter state variable
            sub_tabs = []
            for var, sub_tab_title in zip(self.filter_states[r_idx], self.filter_tab_titles[r_idx]):
                sub_tabs.append(self._make_sub_tab(self.plt.plot_filter_state(robot, var, self.sim.dt, self.replay_data, r_idx), sub_tab_title, True))

            tabs.append(self._wrap_sub_tabs(sub_tabs, tab_title))

        return tabs

    def _make_unity_tab(self):
        return dcc.Tab(
            html.Div([
                html.Iframe(
                    src = self.unity_player.HTTP_unity_url,
                    style={
                        'width': '100%', 
                        'height': self.unity_height,
                        'margin': { 't': self.unity_top_margin },
                        'border': 'none'
                    }
            )]),
            label = '3D scene'           
        )

    def _make_error_msg_tab(self, tab_title: str, error_msg: str):
        return dcc.Tab(
            error_msg,
            label = tab_title,
            style = { 'padding': '5px' }, 
            selected_style = { 'padding': '5px' } 

        )

    def _make_sub_tab(self, fig: go.Figure, label: str, spinner: bool = False):
        return dcc.Tab(
                        dcc.Loading(dcc.Graph(figure = fig)) 
                    if spinner else 
                        dcc.Graph(figure = fig),
                    label = label,
                    style = { 'padding': '5px' }, 
                    selected_style = { 'padding': '5px' } 
        )
    
    def _wrap_sub_tabs(self, sub_tabs: list, label: str, spinner: bool = False):
        return dcc.Tab(
                dcc.Loading(dcc.Tabs(sub_tabs, parent_style = { 'width': '100%' }, style = { "white-space": "pre" })) 
            if spinner else
                dcc.Tabs(sub_tabs, parent_style = { 'width': '100%' }, style = { "white-space": "pre" }),
            label = label,
            style = { 'padding': '5px' }, 
            selected_style = { 'padding': '5px' } 
        )

    def _assemble_unity_response(self, request):
        try:
            what = request.get('send')
        
            # prepare response
            response = {}
            if what == "sim_data":
                response = {
                    'type': 'sim_data',
                    'value': {
                        'dt': self.sim.dt,
                        'cam_offset': [20.0, 20.0, 10.0],
                        'view_box': [0.0, 0.0, 0.0]
                    }
                }
            elif what == 'object_data':
                object_data = []
                for r_idx in range(len(self.sim.robots)):
                    object_data.extend(self._robot_data_for_unity(r_idx))
                
                response = {
                    'type': 'object_data',
                    'value': object_data
                }
            else:
                response = {
                    'type': 'error',
                    'message': 'Unknown data requested: {}'.format(what)
                }
            return response
        except json.JSONDecodeError:
                raise SystemExit('Received invalid JSON request {} from Unity.'.format(request))

    def _robot_data_for_unity(self, r_idx: int):
        robot = self.sim.robots[r_idx]
        object_data = []

        # find state trajectory data
        if robot.precomputed:
            s_traj = robot.s_traj
        elif 's_traj' in self.result_data:
            s_traj = self.result_data['s_traj'][r_idx]
        else:
            s_traj = []

        # convert to list[list[float]]
        p_traj = [list(s.p.flatten()) for s in s_traj]
        q_traj = [list(s.q.flatten()) for s in s_traj]

        object_data.append({
            'type': 'robot',
            'name': robot.name,
            'color': robot.color,
            't_min': 0,
            't_max': len(s_traj) - 1,

            'robot_type': 'quad' if isinstance(robot, simple_quad) else 'unknown',
            'p_traj': p_traj,
            'q_traj': q_traj
        })

        # add filter data
        for f_idx in range(len(robot.filters)):
            object_data.extend(self._filter_data_for_unity(r_idx, f_idx))

        # add sensor data
        for ss_idx in range(len(robot.sensors)):
            object_data.extend(self._sensor_data_for_unity(r_idx, ss_idx))

        return object_data
    
    def _filter_data_for_unity(self, r_idx: int, f_idx: int):
        robot = self.sim.robots[r_idx]
        filter = robot.filters[f_idx]
        object_data = []

        # find state trajectory data
        if 's_traj' in self.replay_data:
            s_traj = self.replay_data['s_traj'][r_idx][f_idx]
        else:
            s_traj = []

        # find state error trajectory data
        if 's_e_traj' in self.replay_data:
            s_e_traj = self.replay_data['s_e_traj'][r_idx][f_idx]
        else:
            s_e_traj = []

        # find error ellipsoid data
        if 'cov_s_traj' in self.replay_data:
            cov_s_traj = self.replay_data['cov_s_traj'][r_idx][f_idx]
        else:
            cov_s_traj = []

        if 'cov_q_traj' in self.replay_data:
            cov_q_traj = self.replay_data['cov_q_traj'][r_idx][f_idx]
        else:
            cov_q_traj = []

        # convert to list[list[float]]
        p_traj = [list(s.p.flatten()) for s in s_traj]
        q_traj = [list(s.q.flatten()) for s in s_traj]

        # convert to list[float]
        p_e_traj = [s_e['p'] for s_e in s_e_traj]
        q_e_traj = [s_e['q'] for s_e in s_e_traj]

        # convert to list[list[float]]
        cov_s_traj = [list(s.flatten()) for s in cov_s_traj]
        cov_q_traj = [list(q.flatten()) for q in cov_q_traj]

        object_data.append({
            'type': 'filter',
            'name': filter.name,
            'color': filter.color,
            't_min': 0,
            't_max': len(s_traj) - 1,

            'robot': robot.name, 
            'p_traj': p_traj,
            'q_traj': q_traj,
            'p_e_traj': p_e_traj,
            'q_e_traj': q_e_traj,
            'cov_s_traj': cov_s_traj,
            'cov_q_traj': cov_q_traj
        })

        return object_data

    def _sensor_data_for_unity(self, r_idx: int, ss_idx: int):
        robot = self.sim.robots[r_idx]
        sensor = robot.sensors[ss_idx]
        sensor_type = sensor.name
        interval: int = 1
        t_max: int = -1

        supported_sensor = False
        object_data = []
        measurements = []
        params = []

        if sensor.__class__ in [simple_GPS]:
            supported_sensor = True
            interval = sensor.get_interval('p')
            
            # find measurement data
            if 'm_traj' in self.result_data:
                m_traj = self.result_data['m_traj'][r_idx]
            else:
                m_traj = []
            
            # convert to compressed list[list[float]]
            measurements = [list(m['p'].flatten()) for m in m_traj[::interval]]
            t_max = len(measurements) - 1

        elif sensor.__class__ in [simple_IMU]:
            supported_sensor = True
            sensor_type = sensor.name + '.lf'
            interval = sensor.get_interval('y_m')

            # find measurement data
            if 'm_traj' in self.result_data:
                m_traj = self.result_data['m_traj'][r_idx]
            else:
                m_traj = []
            
            # convert to compressed list[list[list[float]]]
            measurements = [
                [list(m['y_a'].flatten()) for m in m_traj[::interval]],
                [list(m['y_m'].flatten()) for m in m_traj[::interval]]
            ]
            t_max = len(measurements[0]) - 1

            # add (time-constant) ground truth vectors as list[list[float]]
            if 'm_true_traj' in self.result_data:
                m_true = self.result_data['m_true_traj'][r_idx][0]
                params = [
                    list(m_true['y_a'].flatten()),
                    list(m_true['y_m'].flatten())
                ]

        if supported_sensor:
            object_data.append({
                'type': 'sensor',
                'name': sensor.name,
                'color': sensor.color,
                't_min': 0,
                't_max': t_max,

                'robot': robot.name,
                'sensor_type': sensor_type,
                'interval': interval,
                'm': measurements,
                'params': params
            })

        return object_data
