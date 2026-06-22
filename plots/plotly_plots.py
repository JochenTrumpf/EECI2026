from collections.abc import Callable
import numpy as np
import textwrap

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from robots.robot_state import robot_state
from robots.robot import robot
from robots.simple_quad import simple_quad

from simulations.sim import sim

class plotter():
    """
    Container class for plotting functions that return plotly figures

    Configuration:
     __init__(fig_height, fig_spacing, top_margin, axes_font_size, x_standoff, component_marker, error_marker, ground_truth_marker)
      + fig_height: height argument for plotly.graph_objects.Figure.update_layout
      + fig_spacing: vertical_spacing argument for plotly.subplots.make_subplots
      + top_margin: margin['t'] argument for plotly.graph_objects.Figure.update_layout

      + axes_font_size: title_font['size'] argument for plotly.graph_objects.Figure.update_(x|y)axes
      + x_standoff: title_standoff argument for plotly.graph_objects.Figure.update_xaxes
      
      + hf_component_marker: marker dict to use for component plots of high frequency variables
      + hf_error_marker: marker dict to use for error plots of high frequency variables
      + hf_ground_truth_marker: marker dict to use for ground truth component plots of high frequency variables
      + lf_component_marker: marker dict to use for component plots of low frequency variables
      + lf_error_marker: marker dict to use for error plots of low frequency variables
      + lf_ground_truth_marker: marker dict to use for ground truth component plots of low frequency variables

    Conventions:
     - legend_group
        robot.name - anything to do with the named robot
        robot.name + '.wayposes' - anything to do with the wayposes for the named robot
        sensor.get_names(var) - anything to do with measurements of the variable var through sensor
        (*.)filter.name - anything to do with the named filter
        *.'ground_truth' -anything to do with ground truth trajectories

     - legend_rank
        0XXXX environmental objects 
        01*** landmarks
        
        1X0** robot number X (X = 0, 1, ...); X = 0 by default if only one robot is involved 
        1X1** wayposes for robot number X
        1X2** measurements for robot number X 
        1X3** filters for robot number X  
        
        90000 extra info

    Top-level functions:
     - plot_robot_state(robot, var, s_traj = None)
        + returns a plotly.graph_objects.Figure with one subplot per component of the robot state variable named var (str)
        + calls robot.get_var_attributes()[var] for axes labels
        + reads the state trajectory from s_traj if supplied

     - plot_measurement(robot, var, m_traj, m_true_traj, m_e_traj)
        + returns a plotly.graph_objects.Figure with one subplot per component of the robot sensor variable named var (str) and one subplot for the measurement error
        + component plots also contain ground truth trajectories
        + calls sensor.get_var_attributes()[var] for axes labels

     - plot_filter_state(robot, var, replay_data, r_idx)
        + returns a plotly.graph_objects.Figure with one subplot per component of the robot filter variable named var (str) and one subplot for the estimation error
        + filter trajectory data is extracted from replay_data['s_traj'][r_idx], replay_data['s_true_traj'][r_idx] and replay_data['s_e_traj'][r_idx]
        + component plots also contain ground truth trajectories
        + calls filter.get_var_attributes()[var] for the first applicable filter in robot.filters for axes labels

     - plot_error_histogram(robot, sim, error_type, var, error_data)
        + returns a plotly.graph_objects.Figure showing overlayed histograms for each filter
        + each histogram shows the error distribution of the error of type error_type (e.g. 'e_N" or 'e_avg') of variable var across multiple sim.sim_run() calls
        + error_data is of type tuple[list[dict]] in the format error_data[f_idx][run_id][var]
        + calls filter.get_var_attributes()[var] for the first applicable filter in robot.filters for part of axes labels
        + calls sim.get_var_attributes()[error_type]['var_title'] for description of error_type for part of axes labels
        
    Component-level functions:
     - add_component_subplot(fig, row, marker, color, type, reader, T, dt, name, legend_group, legend_rank, show_legend, y_min = 0.0, y_max = 0.0)
        + adds a subplot to the given row of fig (plotly.graph_objects_Figure created with plotly.subplots.make_subplots)
        + the subplot shows the return values of Callable reader(t) over the time steps t = 0, ..., T
        + x-axis scale is in seconds (1.0 / dt)
        + uses name, marker and color for legend entry and plot
        + returns updated y_min and y_max values based on y_data

     - add_simple_quad_wayposes(fig, row, robot, legend_group, legend_rank, show_legend, y_min, y_max)
        + adds arrows at the top (y_max) and bottom (y_min) of the subplot in the given row of fig (plotly.graph_objects_Figure created with plotly.subplots.make_subplots)
        + each pair of arrows corresponds to a robot.waypose_steps[i]
        + robot needs to be of type (subclass of) simple_quad
        
     - def add_zero_line(fig, row, marker, color, T, dt, name, legend_group, legend_rank, show_legend)
        + adds a zero line over time steps 0, ..., T to the given row of fig (plotly.graph_objects_Figure created with plotly.subplots.make_subplots)
        + x-axis scale is in seconds (1.0 / dt)
        + uses name, marker and color for legend entry and plot

    Internal methods:
     - reader(t) getters for add_component_subplot
        + _get_component_reader(obj, traj, var, i): reader(t) returns obj.traj[t].var[i][0]
        + _get_measurement_component_reader(traj, var, idx, i): reader(t) returns traj[t][var][i][0] if idx < 0 (type 'nD'), else it returns traj[t][var][idx][i][0] (type 'nD_list') 
        + _get_measurement_error_reader(traj, var, idx): reader(t) returns traj[t][var] if idx < 0 (type 'nD'), else it returns traj[t][var][idx] (type 'nD_list')
        + _get_filter_component_reader(traj, f_idx, var, i): reader(t) returns traj[f_idx][t].var[i][0]
        + _get_filter_error_reader(traj, f_idx, var): reader(t) returns traj[f_idx][t][var]
        
     - _set_axes_data(fig, row, title_text, T, dt)
        + set axes ranges and titles for subplot in the given row of fig

     - _set_figure_data(fig)
        + set figure size and margins
    """

    # Configuration

    def __init__(self, fig_height: int = 750, fig_spacing: float = 0.1, top_margin: int = 25, 
                       axes_font_size: int = 12, x_standoff: int = 5, 
                       hf_component_marker = { 'symbol': 'circle', 'size': 1, 'opacity': 1.0 }, 
                       hf_error_marker = { 'symbol': 'circle', 'size': 1, 'opacity': 1.0 }, 
                       hf_ground_truth_marker = { 'symbol': 'circle-open', 'size': 3, 'opacity': 0.3 },
                       lf_component_marker = { 'symbol': 'circle', 'size': 3, 'opacity': 1.0 }, 
                       lf_error_marker = { 'symbol': 'circle', 'size': 3, 'opacity': 1.0 }, 
                       lf_ground_truth_marker = { 'symbol': 'circle-open', 'size': 6, 'opacity': 0.3 }):

        self.fig_height = fig_height
        self.fig_spacing = fig_spacing
        self.top_margin = top_margin
        
        self.axes_font_size = axes_font_size
        self.x_standoff = x_standoff

        self.hf_component_marker = hf_component_marker
        self.hf_error_marker = hf_error_marker
        self.hf_ground_truth_marker = hf_ground_truth_marker
        self.lf_component_marker = lf_component_marker
        self.lf_error_marker = lf_error_marker
        self.lf_ground_truth_marker = lf_ground_truth_marker

    # Top-level functions

    def plot_robot_state(self, robot: robot, var: str, dt: float, s_traj: None|list[robot_state] = None):
        if s_traj is None:
            # check that we have trajectory data to plot
            if (not robot.precomputed):
                raise SystemExit('plot_robot_state() called without supplied s_traj but the trajectory of robot {} was not precomputed.'.format(robot.name))
            # check that the robot has the required state variable
            if not hasattr(robot.s, var):
                raise SystemExit('Tried to call plot_robot_state() for non-existent state variable {} of robot {}.'.format(var, robot.name))
        else:
            # check that the supplied trajectory has the required state variable
            if not hasattr(s_traj[0], var):
                raise SystemExit('Tried to call plot_robot_state() for state variable {} of robot {} but the supplied s_traj does not have this variable.'.format(var, robot.name))

        # one subplot per component (dimension)
        attr = robot.get_var_attributes()[var]
        dim = attr['component_number']
        fig = make_subplots(rows = dim, cols = 1, vertical_spacing = self.fig_spacing)

        # configure markers
        component_marker = getattr(self, attr['var_freq'] + '_component_marker')
        
        # add component trajectories and set axis ranges and titles; add robot wayposes (if applicable);
        for i in range(dim):
            if s_traj is None:
                reader = self._get_component_reader(robot, 's_traj', var, i)
                T_max = robot.T
            else:
                reader = self._get_traj_reader(s_traj, var, i)
                T_max = len(s_traj) - 1
            y_min, y_max = self.add_component_subplot(fig, i+1, component_marker, robot.color, reader, T_max, dt, 
                                                      robot.name, robot.name, 10000, False if i > 0 else True, True)
            self._set_axes_data(fig, i+1, attr['var_title'] + ' - ' + attr['component_names'][i], T_max, dt)
            if isinstance(robot, simple_quad) and robot.initialized:
                self.add_simple_quad_wayposes(fig, i+1, robot, dt, 
                                              robot.name + '.wayposes', 10100, False if i > 0 else True, y_min, y_max)

        self._set_figure_data(fig)
        return fig

    def plot_measurement(self, robot: robot, var: str, dt: float, m_traj: list[dict], m_true_traj: list[dict], m_e_traj: list[dict]):
        # check that we have the requested measurement in all provided measurement trajectories
        if (not var in m_traj[0]) or (not var in m_true_traj[0]) or (not var in m_e_traj[0]):
            raise SystemExit('Called plot_measurement() with trajectory data that does not contain the requested measurement variable {}.'.format(var))

        # check that the robot has a sensor producing the requested measurement
        sensor = robot.get_sensor(var)
        if sensor is None:
            raise SystemExit('Called plot_measurement() for measurement variable {} but the robot has no such sensor.'.format(var))
        
        # variable attributes
        attr = sensor.get_var_attributes()[var]
        
        # check that we know how to plot var
        var_type = attr['var_type']
        if var_type == 'nD':
            indices = [-1]
            names = [sensor.name]
            colors = [sensor.color]
        elif var_type == 'nD_list':
            indices = sensor.runtime_attributes[var]['indices']
            names = [sensor.name + ' - ' + name for name in sensor.runtime_attributes[var]['names']]
            colors = sensor.runtime_attributes[var]['colors']
        else:
            raise SystemExit('plot_measurement() called with unsupported var_type {}.'.format(var_type))

        # one subplot per component (dimension) plus one error subplot
        n_plots = attr['component_number'] + 1
        fig = make_subplots(rows = n_plots, cols = 1, vertical_spacing = self.fig_spacing)

        # configure markers
        component_marker = getattr(self, attr['var_freq'] + '_component_marker')
        error_marker = getattr(self, attr['var_freq'] + '_error_marker')
        ground_truth_marker = getattr(self, attr['var_freq'] + '_ground_truth_marker')

        # accumulators for y_data limits
        y_min: list[float] = [float('inf')] * n_plots
        y_max: list[float] = [float('-inf')] * n_plots

        for idx, name, color in zip(indices, names, colors):
            # add error trajectory
            reader = self._get_measurement_error_reader(m_e_traj, var, idx)
            y_min[0], y_max[0] = self.add_component_subplot(fig, 1, error_marker, color, reader, len(m_e_traj) - 1, dt, 
                                                            name, name, 10200, True, 'legendonly' if idx > 0 else True, y_min[0], y_max[0])
            # add component trajectories and ground truth component trajectories                                                                  
            for i in range(1, n_plots):
                reader = self._get_measurement_component_reader(m_traj, var, idx, i-1)
                y_min[i], y_max[i] = self.add_component_subplot(fig, i+1, component_marker, color, reader, len(m_traj) - 1, dt, 
                                                                name, name, 10200, False, 'legendonly' if idx > 0 else True, y_min[i], y_max[i])
                reader = self._get_measurement_component_reader(m_true_traj, var, idx, i-1)
                y_min[i], y_max[i] = self.add_component_subplot(fig, i+1, ground_truth_marker, color, reader, len(m_true_traj) - 1, dt, 
                                                                ' ground truth', name + '.ground_truth', 10200, False if i > 1 else True, 'legendonly' if idx > 0 else True, y_min[i], y_max[i])
        
        T_max = max(len(m_e_traj) - 1, len(m_traj) - 1, len(m_true_traj) - 1)
        # set axis ranges and titles and add robot wayposes (if applicable)
        titles = [attr['error_title'], *[attr['var_title'] + ' - ' + attr['component_names'][i] for i in range(n_plots - 1)]]
        for i in range(n_plots):
            self._set_axes_data(fig, i+1, titles[i], T_max, dt)
            if isinstance(robot, simple_quad):
                self.add_simple_quad_wayposes(fig, i+1, robot, dt,
                                              robot.name + '.wayposes', 10100, False if i > 0 else True, y_min[i], y_max[i])

        self._set_figure_data(fig)
        return fig

    def plot_filter_state(self, robot: robot, var: str, dt: float, replay_data: dict, r_idx: int):
        # list of filter indices idx for which robot.filters[idx] estimates var
        f_l = []
        for f_idx, f in enumerate(robot.filters):
            if var in f.get_var_attributes():
                f_l.append(f_idx)

        # check that we have at least one filter tajectory to plot
        if not f_l:
            raise SystemExit('Called plot_filter_state() for robot {} and variable {} but none of the configured filters estimates this.'.format(robot.name, var))

        # check that replay_data contains filter state trajectories
        if ('s_traj' not in replay_data) or ('s_true_traj' not in replay_data) or ('s_e_traj' not in replay_data):
            raise SystemExit('Called plot_filter_state() for robot {} but the provided replay_data is missing filter trajectory recordings.'.format(robot.name))
        
        # variable attributes for first available filter; used for var_type and component_number
        attr = robot.filters[f_l[0]].get_var_attributes()[var]
        
        # check that we know how to plot var
        var_type = attr['var_type']
        dim = attr['component_number']

        names = []
        colors = []
        stack_indices = []

        if (var_type == 'nD'):
            names.append(robot.name)
            colors.append(robot.color)
            stack_indices.append(0)
            filters = [[]]
            var_indices = [[]]
            for f_idx in f_l:
                filters[0].append(f_idx)
                var_indices[0].append(0)
        elif (var_type == 'stacked_partial_nD_list'):
            # combine stack_indices, names and colors accross all available filters
            for f_idx in f_l:
                for j, idx in enumerate(robot.filters[f_idx].runtime_attributes[var]['stack_indices']):
                    if idx not in stack_indices:
                        stack_indices.append(idx)
                        names.append(robot.filters[f_idx].runtime_attributes[var]['names'][j])
                        colors.append(robot.filters[f_idx].runtime_attributes[var]['colors'][j])
            # fill filters and indices list
            filters = [[] for _ in range(len(stack_indices))]
            var_indices = [[] for _ in range(len(stack_indices))]
            idx_l = [0] * len(f_l)
            for j, idx in enumerate(stack_indices):
                for k, f_idx in enumerate(f_l):
                    if idx in robot.filters[f_idx].runtime_attributes[var]['stack_indices']:
                        filters[j].append(f_idx)
                        var_indices[j].append(idx_l[k])
                        idx_l[k] = idx_l[k] + dim
        else:
            raise SystemExit('plot_filter_state() called with unsupported var_type {}.'.format(var_type))

        # one subplot per component plus one error subplot
        n_plots = dim + 1
        fig = make_subplots(rows = n_plots, cols = 1, vertical_spacing = self.fig_spacing)
        
        # configure markers
        component_marker = getattr(self, attr['var_freq'] + '_component_marker')
        error_marker = getattr(self, attr['var_freq'] + '_error_marker')

        # accumulators for y_data limits
        y_min: list[float] = [float('inf')] * n_plots
        y_max: list[float] = [float('-inf')] * n_plots

        # add error trajectories
        T_max = 0 
        for f_idx in f_l:
            reader = self._get_filter_error_reader(replay_data['s_e_traj'][r_idx], f_idx, var)
            y_min[0], y_max[0] = self.add_component_subplot(fig, 1, error_marker, robot.filters[f_idx].color, reader, len(replay_data['s_e_traj'][r_idx][f_idx]) - 1, dt, 
                                                            robot.filters[f_idx].name + ' error', robot.filters[f_idx].name, 10300, False if len(names) == 1 else True, True, y_min[0], y_max[0])
            T_max = max(T_max, len(replay_data['s_e_traj'][r_idx][f_idx]) - 1)

        # add component trajectories
        for j, (name, color) in enumerate(zip(names, colors)):
            # add dummy legend entry as title for this list element
            fig.add_trace(go.Scatter(
                    x = [None], 
                    y = [None],
                    mode = 'lines',
                    line_width = 0,
                    name = '<b>' + name + '</b>',
                    legendrank = 10302,
                    showlegend = True,
                    visible = True
                ), 
                row = 1, col = 1
            )
            
            # add filter component trajectories
            for k, f_idx in enumerate(filters[j]):
                for i in range(1, n_plots):
                    legend_group = robot.filters[f_idx].name if len(names) == 1 else name + '.' + robot.filters[f_idx].name
                    reader = self._get_filter_component_reader(replay_data['s_traj'][r_idx], f_idx, var, var_indices[j][k] + i - 1)
                    y_min[i], y_max[i] = self.add_component_subplot(fig, i+1, component_marker, robot.filters[f_idx].color, reader, len(replay_data['s_traj'][r_idx][f_idx]) - 1, dt, 
                                                                    robot.filters[f_idx].name, legend_group, 10302, False if i > 1 else True, 'legendonly' if j > 0 else True, y_min[i], y_max[i])
                T_max = max(T_max, len(replay_data['s_traj'][r_idx][f_idx]) - 1)

            # add ground truth component trajectories from first applicable filter
            f_idx = filters[j][0]
            for i in range(1, n_plots):
                reader = self._get_filter_component_reader(replay_data['s_true_traj'][r_idx], f_idx, var, var_indices[j][k] + i - 1)
                y_min[i], y_max[i] = self.add_component_subplot(fig, i+1, component_marker, color, reader, len(replay_data['s_true_traj'][r_idx][f_idx]) - 1, dt, 
                                                                'ground truth', name + '.ground_truth', 10302, False if i > 1 else True, 'legendonly' if j > 0 else True, y_min[i], y_max[i])
            T_max = max(T_max, len(replay_data['s_true_traj'][r_idx][f_idx]) - 1)

        # set axis ranges and titles and add robot wayposes (if applicable)
        titles = [attr['error_title'], *[attr['var_title'] + ' - ' + attr['component_names'][i] for i in range(n_plots - 1)]]
        for i in range(n_plots):
            self._set_axes_data(fig, i+1, titles[i], T_max, dt)
            if isinstance(robot, simple_quad):
                self.add_simple_quad_wayposes(fig, i+1, robot, dt,
                                              robot.name + '.wayposes', 10100, False if i > 0 else True, y_min[i], y_max[i])

        # adjust legend spacing
        fig.update_layout(
            legend_traceorder = 'normal'
        )
        
        self._set_figure_data(fig)
        return fig

    def plot_error_histogram(self, robot: robot, sim: sim, error_type: str, var: str, error_data: tuple[list[dict]]):
        # list of filter indices idx for which robot.filters[idx] estimates var
        f_l = []
        for f_idx, f in enumerate(robot.filters):
            if var in f.get_var_attributes():
                f_l.append(f_idx)

        # check that we have at least one filter to plot data for
        if not f_l:
            raise SystemExit('Called plot_error_histogram() for robot {} and variable {} but none of the configured filters estimates this.'.format(robot.name, var))

        # check that error_data contains the required data
        for f_idx in f_l:
            if var not in error_data[f_idx][0]:
                raise SystemExit('Called plot_error_histogram() for robot {} but the provided error_data is missing filter error recordings.'.format(robot.name))
        
        # variable attributes for first available filter
        attr = robot.filters[f_l[0]].get_var_attributes()[var]

        fig = go.Figure()
        for f_idx in f_l:
            # prepare histogram data
            x_data = [d[var] for d in error_data[f_idx]]                        
            
            # add histogram
            fig.add_trace(go.Histogram(
                x = x_data,
                name = robot.filters[f_idx].name + ' mean: {:.2f}, std.dev: {:.2f}'.format(np.nanmean(x_data), np.nanstd(x_data)),
                marker_color = robot.filters[f_idx].color,
                hovertemplate = 'Value: %{x}<br>Count: %{y}<br>Runs: %{pointNumbers}<extra></extra>'
            ))
            
            # add extra info to legend
            nan_indices = np.where(np.isnan(np.array(x_data)))[0].tolist()
            info = ', '.join([str(i) for i in nan_indices])
            info = '<br>'.join(textwrap.wrap(info, width = 25, initial_indent = 'NaN: ', subsequent_indent = '     '))
            fig.add_scatter(
                x = [None],
                y = [None],
                name = info,
                mode = 'lines',
                line_width = 0,
                showlegend = True
            )

        fig.update_layout(
            xaxis_title_text = attr['error_long_title'] + ' - ' + sim.get_var_attributes()[error_type]['var_title'],
            yaxis_title_text = 'number of occurrences'
        )

        self._set_figure_data(fig)
        return fig
    
    # Component-level functions

    def add_component_subplot(self, fig: go.Figure, row: int, marker: dict, color: str, reader: Callable[[int], float | None], T: int, dt: float,
                                    name: str, legend_group: str, legend_rank: int, show_legend: bool, visible: bool | str, y_min: float = float('inf'), y_max: float = float('-inf')):
        # prepare plotting data
        x_data = []
        y_data = []
        for t in range(T + 1):
            d = reader(t)
            if d is not None:
                x_data.append(t * dt)
                y_data.append(d)
                y_min = min(y_min, d)
                y_max = max(y_max, d)                      

        # add plot
        fig.add_trace(go.Scatter(
                x = x_data, 
                y = y_data,
                mode = 'markers',
                marker = marker,
                marker_color = color,
                name = name,
                legendgroup = legend_group,
                legendrank = legend_rank,
                showlegend = show_legend,
                visible = visible
            ), 
            row = row, col = 1
        )

        return y_min, y_max
    
    def add_simple_quad_wayposes(self, fig: go.Figure, row: int, robot: simple_quad, dt: float,
                                       legend_group: str, legend_rank: int, show_legend: bool, y_min: float, y_max: float):
        # prepare plotting data
        x_data = [t * dt for t in robot.waypose_steps]
        y_data_min = [y_min] * len(robot.waypose_steps)
        y_data_max = [y_max] * len(robot.waypose_steps)

        # add plots
        fig.add_trace(go.Scatter(
                x = x_data,
                y = y_data_min,
                mode = 'markers',
                marker_symbol = 'arrow-up',
                marker_color = robot.color,
                name = 'wayposes',
                legendgroup = legend_group,
                legendrank = legend_rank,
                showlegend = show_legend
            ),
            row = row, col = 1
        )
        fig.add_trace(go.Scatter(
                x = x_data,
                y = y_data_max,
                mode = 'markers',
                marker_symbol = 'arrow-down',
                marker_color = robot.color,
                name = 'wayposes',
                legendgroup = legend_group,
                legendrank = legend_rank,
                showlegend = False
            ),
            row = row, col = 1
        )

    def add_zero_line(self, fig: go.Figure, row: int, marker: dict, color: str, T: int, dt: float, 
                            name: str, legend_group: str, legend_rank: int, show_legend: bool, visible: bool | str, y_min: float = float('inf'), y_max: float = float('-inf')):
        # prepare plotting data
        x_data = [t * dt for t in range(T + 1)]
        y_data = [0.0] * len(x_data)
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)

        # add plot
        fig.add_trace(go.Scatter(
                x = x_data,
                y = y_data,
                mode = 'markers',
                marker = marker,
                marker_color = color,
                name = name,
                legendgroup = legend_group,
                legendrank = legend_rank,
                showlegend = show_legend,
                visible = visible
            ),
            row = row, col = 1
        )

        return y_min, y_max

    # internal methods
    def _get_component_reader(self, obj, traj: str, var: str, i: int):
        return lambda t: getattr(getattr(obj, traj)[t], var)[i][0]
    
    def _get_traj_reader(self, traj: list, var: str, i: int):
        return lambda t: getattr(traj[t], var)[i][0]
    
    def _get_measurement_component_reader(self, traj: list[dict], var: str, idx: int, i: int):
        if idx < 0:
            return lambda t: (traj[t][var][i][0] if traj[t][var] is not None else None)
        else:
            return lambda t: (traj[t][var][idx][i][0] if (traj[t][var] is not None) and (traj[t][var][idx] is not None) else None)

    def _get_measurement_error_reader(self, traj: list[dict], var: str, idx: int):
        if idx < 0:
            return lambda t: traj[t][var]
        else:
            return lambda t: (traj[t][var][idx] if traj[t][var] is not None else None)

    def _get_filter_component_reader(self, traj: list[tuple], f_idx: int, var: str, i: int):
        return lambda t: getattr(traj[f_idx][t], var)[i][0]

    def _get_filter_error_reader(self, traj: list[tuple], f_idx: int, var: str):
        return lambda t: traj[f_idx][t][var]

    def _set_axes_data(self, fig: go.Figure, row: int, title_text: str, T: int, dt: float):
        # set subplot axis ranges and titles
        fig.update_xaxes(
            range = [0.0, T * dt],
            title_text = 'time (s)', 
            title_font = { 'size': self.axes_font_size },
            title_standoff = self.x_standoff,
            row = row, col = 1
        )
        
        fig.update_yaxes(
            title_text = title_text, 
            title_font = { 'size': self.axes_font_size },
            row = row, col = 1
        )

    def _set_figure_data(self, fig: go.Figure):
        # set figure height and top margin
        fig.update_layout(
            height = self.fig_height,
            margin = { 't': self.top_margin }
        )
