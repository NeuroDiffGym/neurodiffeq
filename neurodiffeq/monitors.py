import math
import torch
import warnings
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
from abc import ABC, abstractmethod

from ._version_utils import deprecated_alias
from .function_basis import RealSphericalHarmonics as _RealSphericalHarmonics
from .generators import Generator1D as _Generator1D
from .generators import Generator2D as _Generator2D
from .generators import Generator3D as _Generator3D
from .conditions import IrregularBoundaryCondition as _IrregularBC
from .operators import grad


def _updatable_contour_plot_available():
    from packaging.version import parse as vparse
    from matplotlib import __version__
    return vparse(__version__) >= vparse('3.3.0')


class BaseMonitor(ABC):
    r"""A tool for checking the status of the neural network during training.

    A monitor keeps track of a matplotlib.figure.Figure instance and updates the plot
    whenever its ``check()`` method is called (usually by a ``neurodiffeq.solvers.BaseSolver`` instance).

    .. note::
        Currently, the ``check()`` method can only run synchronously.
        It blocks the training / validation process, so don't call the ``check()`` method too often.
    """

    def __init__(self, check_every=None):
        self.check_every = check_every or 100
        self.fig = ...
        self.using_non_gui_backend = (matplotlib.get_backend() == 'agg')

        if matplotlib.get_backend() == 'module://ipykernel.pylab.backend_inline':
            warnings.warn(
                "You seem to be using jupyter notebook with '%matplotlib inline' "
                "which can lead to monitor plots not updating. "
                "Consider using '%matplotlib notebook' or '%matplotlib widget' instead.",
                UserWarning)

    @abstractmethod
    def check(self, nets, conditions, history):
        pass  # pragma: no cover

    def to_callback(self, fig_dir=None, format=None, logger=None):
        r"""Return a callback that updates the monitor plots, which will be run

        1. Every ``self.check_every`` epochs; and
        2. After the last local epoch.

        :param fig_dir: Directory for saving monitor figs; if not specified, figs will not be saved.
        :type fig_dir: str
        :param format: Format for saving figures: {'jpg', 'png' (default), ...}.
        :type format: str
        :param logger: The logger (or its name) to be used for the returned callback. Defaults to the 'root' logger.
        :type logger: str or ``logging.Logger``
        :return: The callback that updates the monitor plots.
        :rtype: neurodiffeq.callbacks.BaseCallback
        """
        # to avoid circular import
        from .callbacks import MonitorCallback, PeriodLocal, OnLastLocal
        action_cb = MonitorCallback(self, fig_dir=fig_dir, format=format, logger=logger)
        condition_cb = OnLastLocal(logger=logger)
        if self.check_every:
            condition_cb = condition_cb | PeriodLocal(self.check_every, logger=logger)
        return condition_cb.set_action_callback(action_cb)


class MonitorSpherical(BaseMonitor):
    r"""A monitor for checking the status of the neural network during training.

    :param r_min:
        The lower bound of radius,
        i.e., radius of interior boundary.
    :type r_min: float
    :param r_max:
        The upper bound of radius,
        i.e., radius of exterior boundary.
    :type r_max: float
    :param check_every:
        The frequency of checking the neural network represented by the number of epochs between two checks.
        Defaults to 100.
    :type check_every: int, optional
    :param var_names:
        Names of dependent variables.
        If provided, shall be used for plot titles.
        Defaults to None.
    :type var_names: list[str]
    :param shape:
        Shape of mesh for visualizing the solution.
        Defaults to (10, 10, 10).
    :type shape: tuple[int]
    :param r_scale:
        'linear' or 'log'.
        Controls the grid point in the :math:`r` direction.
        Defaults to 'linear'.
    :type r_scale: str
    :param theta_min:
        The lower bound of polar angle.
        Defaults to :math:`0`.
    :type theta_min: float
    :param theta_max:
        The upper bound of polar angle.
        Defaults to :math:`\pi`.
    :type theta_max: float
    :param phi_min:
        The lower bound of azimuthal angle.
        Defaults to :math:`0`.
    :type phi_min: float
    :param phi_max:
        The upper bound of azimuthal angle.
        Defaults to :math:`2\pi`.
    :type phi_max: float
    """

    def __init__(self, r_min, r_max, check_every=None, var_names=None, shape=(10, 10, 10), r_scale='linear',
                 theta_min=0.0, theta_max=math.pi, phi_min=0.0, phi_max=math.pi * 2):
        """Initializer method
        """
        super(MonitorSpherical, self).__init__(check_every=check_every)
        self.contour_plot_available = _updatable_contour_plot_available()
        if not self.contour_plot_available:
            warnings.warn("Warning: contourf plot only available for matplotlib version >= v3.3.0 "
                          "switching to matshow instead")
        self.fig = None
        self.axs = []  # subplots
        self.ax_metrics = None
        self.ax_loss = None
        self.cbs = []  # color bars
        self.names = var_names
        self.shape = shape
        # input for neural network

        if r_scale == 'log':
            r_min, r_max = np.log(r_min), np.log(r_max)

        gen = _Generator3D(
            grid=shape,
            xyz_min=(r_min, theta_min, phi_min),
            xyz_max=(r_max, theta_max, phi_max),
            method='equally-spaced'
        )
        rs, thetas, phis = gen.get_examples()  # type: torch.Tensor, torch.Tensor, torch.Tensor

        if r_scale == 'log':
            rs = torch.exp(rs)

        self.r_tensor = rs.reshape(-1, 1)
        self.theta_tensor = thetas.reshape(-1, 1)
        self.phi_tensor = phis.reshape(-1, 1)

        self.r_label = rs.reshape(-1).detach().cpu().numpy()
        self.theta_label = thetas.reshape(-1).detach().cpu().numpy()
        self.phi_label = phis.reshape(-1).detach().cpu().numpy()

        self.n_vars = None

    @staticmethod
    def _longitude_formatter(value, count):
        value = int(round(value / math.pi * 180)) - 180
        if value == 0 or abs(value) == 180:
            marker = ''
        elif value > 0:
            marker = 'E'
        else:
            marker = 'W'
        return f'{abs(value)}°{marker}'

    @staticmethod
    def _latitude_formatter(value, count):
        value = int(round(value / math.pi * 180)) - 90
        if value == 0:
            marker = ''
        elif value > 0:
            marker = 'N'
        else:
            marker = 'S'
        return f'{abs(value)}°{marker}'

    def _compute_us(self, nets, conditions):
        r, theta, phi = self.r_tensor, self.theta_tensor, self.phi_tensor
        return [
            cond.enforce(net, r, theta, phi).detach().cpu().numpy()
            for net, cond in zip(nets, conditions)
        ]

    @deprecated_alias(loss_history='history')
    def check(self, nets, conditions, history, analytic_mse_history=None):
        r"""Draw (3n + 2) plots

         1. For each function :math:`u_i(r, \phi, \theta)`, there are 3 axes:

            - one ax for :math:`u`-:math:`r` curves grouped by :math:`\phi`
            - one ax for :math:`u`-:math:`r` curves grouped by :math:`\theta`
            - one ax for :math:`u`-:math:`\theta`-:math:`\phi` contour heat map

         2. Additionally, one ax for training and validaiton loss, another for the rest of the metrics

        :param nets:
            The neural networks that approximates the PDE.
        :type nets: list [`torch.nn.Module`]
        :param conditions:
            The initial/boundary condition of the PDE.
        :type conditions: list [`neurodiffeq.conditions.BaseCondition`]
        :param history:
            A dict of history of training metrics and validation metrics,
            where keys are metric names (str) and values are list of metrics values (list[float]).
            It must contain a 'train_loss' key and a 'valid_loss' key.
        :type history: dict[str, list[float]]
        :param analytic_mse_history:
            **[DEPRECATED]**
            Include 'train_analytic_mse' and 'valid_analytic_mse' in ``history`` instead.
        :type analytic_mse_history: dict['train': list[float], 'valid': list[float]], deprecated

        .. note::
            ``check`` is meant to be called by ``neurodiffeq.solvers.BaseSolver``.
        """

        for key in ['train', 'valid']:
            if key in history:
                warnings.warn(f'`{key}` is deprecated, use `{key}_loss` instead', FutureWarning)
                history[key + '_loss'] = history.pop(key)

        if ('train_loss' not in history) or ('valid_loss' not in history):
            raise ValueError("Either 'train_loss' or 'valid_loss' not present in `history`.")

        # initialize the figure and axes here so that the Monitor knows the number of dependent variables and
        # shape of the figure, number of the subplots, etc.
        n_vars = len(nets) if self.n_vars is None else self.n_vars
        n_row = (n_vars + 2) if len(history) > 2 else (n_vars + 1)
        n_col = 3

        if analytic_mse_history is not None:
            warnings.warn(
                "`analytic_mse_history` is deprecated. "
                "Include 'train_analytic_mse' and 'valid_analytic_mse' in ``history`` instead.",
                FutureWarning,
            )
            history['train_analytic_mse'] = analytic_mse_history['train']
            history['valid_analytic_mse'] = analytic_mse_history['valid']

        if not self.fig:
            self.fig = plt.figure(figsize=(24, 6 * n_row))
            self.fig.tight_layout()
            self.axs = self.fig.subplots(nrows=n_row, ncols=n_col, gridspec_kw={'width_ratios': [1, 1, 2]})
            # remove 1-1-2 empty axes, which will be replaced by ax_loss and ax_metrics
            for row in self.axs[n_vars:]:
                for ax in row:
                    ax.remove()
            self.cbs = [None] * n_vars
            if len(history) > 2:
                self.ax_loss = self.fig.add_subplot(n_row, 1, n_row - 1)
                self.ax_metrics = self.fig.add_subplot(n_row, 1, n_row)
            else:
                self.ax_loss = self.fig.add_subplot(n_row, 1, n_row)

        us = self._compute_us(nets, conditions)

        for i, u in enumerate(us):
            try:
                var_name = self.names[i]
            except (TypeError, IndexError):
                var_name = f"u[{i}]"

            # prepare data for plotting
            u_across_r = u.reshape(*self.shape).mean(0)
            df = pd.DataFrame({
                '$r$': self.r_label,
                '$\\theta$': self.theta_label,
                '$\\phi$': self.phi_label,
                'u': u.reshape(-1),
            })

            # u-r curve grouped by phi
            ax = self.axs[i][0]
            self._update_r_plot_grouped_by_phi(var_name, ax, df)

            # u-r curve grouped by theta
            ax = self.axs[i][1]
            self._update_r_plot_grouped_by_theta(var_name, ax, df)

            # u-theta-phi heatmap/contourf depending on matplotlib version
            ax = self.axs[i][2]
            self._update_contourf(var_name, ax, u_across_r, colorbar_index=i)

        self._refresh_history(
            self.ax_loss,
            {name: history[name] for name in history if name in ['train_loss', 'valid_loss']},
            x_label='Epochs',
            y_label='Loss Value',
            title='Loss (Mean Squared Residual)',
        )

        if len(history) > 2:
            self._refresh_history(
                self.ax_metrics,
                {name: history[name] for name in history if name not in ['train_loss', 'valid_loss']},
                x_label='Epochs',
                y_label='Metric Values',
                title='Other metrics',
            )

        self.customization()
        self.fig.canvas.draw()
        # for command-line, interactive plots, not pausing can lead to graphs not being displayed at all
        # see https://stackoverflow.com/questions/
        # 19105388/python-2-7-mac-osx-interactive-plotting-with-matplotlib-not-working
        if not self.using_non_gui_backend:
            plt.pause(0.05)

    def customization(self):
        """Customized tweaks can be implemented by overwriting this method.
        """
        pass

    @staticmethod
    def _update_r_plot_grouped_by_phi(var_name, ax, df):
        ax.clear()
        sns.lineplot(x='$r$', y='u', hue='$\\phi$', data=df, ax=ax)
        ax.set_title(f'{var_name}($r$) grouped by $\\phi$')
        ax.set_ylabel(var_name)

    @staticmethod
    def _update_r_plot_grouped_by_theta(var_name, ax, df):
        ax.clear()
        sns.lineplot(x='$r$', y='u', hue='$\\theta$', data=df, ax=ax)
        ax.set_title(f'{var_name}($r$) grouped by $\\theta$')
        ax.set_ylabel(var_name)

    # _update_contourf cannot be defined as a static method since it depends on self.contourf_plot_available
    def _update_contourf(self, var_name, ax, u, colorbar_index):
        ax.clear()
        ax.set_xlabel('$\\phi$')
        ax.set_ylabel('$\\theta$')

        ax.set_title(f'{var_name} averaged across $r$')
        if self.contour_plot_available:
            # matplotlib has problems plotting repeatedly `contourf` until version 3.3
            # see https://github.com/matplotlib/matplotlib/issues/15986
            theta = self.theta_label.reshape(*self.shape)[0, :, 0]
            phi = self.phi_label.reshape(*self.shape)[0, 0, :]
            cax = ax.contourf(phi, theta, u, cmap='magma', levels=max(self.shape[-2:]))
            ax.xaxis.set_major_locator(plt.MultipleLocator(math.pi / 6))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(math.pi / 12))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(self._longitude_formatter))
            ax.yaxis.set_major_locator(plt.MultipleLocator(math.pi / 6))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(math.pi / 12))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(self._latitude_formatter))
            ax.grid(which='major', linestyle='--', linewidth=0.5)
            ax.grid(which='minor', linestyle=':', linewidth=0.5)
        else:
            # use matshow() to plot a heatmap instead
            cax = ax.matshow(u, cmap='magma', interpolation='nearest')

        if self.cbs[colorbar_index]:
            self.cbs[colorbar_index].remove()
        self.cbs[colorbar_index] = self.fig.colorbar(cax, ax=ax)

    @staticmethod
    def _refresh_history(ax, history, x_label='Epochs', y_label=None, title=None):
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        for metric in history:
            ax.plot(history[metric], label=metric)
        # By default, metrics are plotted using log-scale
        # If there are negative values in metrics, override `self.customization()` to change to linear-scale
        ax.set_yscale('log')
        ax.legend()

    def new(self):
        self.fig = None
        self.axs = []
        self.cbs = []
        self.ax_metrics = None
        self.ax_loss = None
        return self

    def set_variable_count(self, n):
        r"""Manually set the number of scalar fields to be visualized;
        If not set, defaults to length of ``nets`` passed to ``self.check()`` every time ``self.check()`` is called.

        :param n: number of scalar fields to overwrite default
        :type n: int
        :return: self
        """
        self.n_vars = n
        return self

    def unset_variable_count(self):
        r"""Manually unset the number of scalar fields to be visualized;
        Once unset, the number defaults to length of ``nets``
        passed to ``self.check()`` every time ``self.check()`` is called.

        :return: self
        """
        self.n_vars = None
        return self


class MonitorSphericalHarmonics(MonitorSpherical):
    r"""A monitor for checking the status of the neural network during training.

    :param r_min:
        The lower bound of radius, i.e., radius of interior boundary.
    :type r_min: float
    :param r_max:
        The upper bound of radius, i.e., radius of exterior boundary.
    :type r_max: float
    :param check_every:
        The frequency of checking the neural network represented by the number of epochs between two checks.
        Defaults to 100.
    :type check_every: int, optional
    :param var_names:
        The names of dependent variables; if provided, shall be used for plot titles.
        Defaults to None
    :type var_names: list[str]
    :param shape:
        Shape of mesh for visualizing the solution.
        Defaults to (10, 10, 10).
    :type shape: tuple[int]
    :param r_scale:
        'linear' or 'log'.
        Controls the grid point in the :math:`r` direction.
        Defaults to 'linear'.
    :type r_scale: str
    :param harmonics_fn:
        A mapping from :math:`\theta` and :math:`\phi` to basis functions, e.g., spherical harmonics.
    :type harmonics_fn: callable
    :param theta_min:
        The lower bound of polar angle.
        Defaults to :math:`0`
    :type theta_min: float
    :param theta_max:
        The upper bound of polar angle.
        Defaults to :math:`\pi`.
    :type theta_max: float
    :param phi_min:
        The lower bound of azimuthal angle.
        Defaults to :math:`0`.
    :type phi_min: float
    :param phi_max:
        The upper bound of azimuthal angle.
        Defaults to :math:`2\pi`.
    :type phi_max: float
    :param max_degree:
        **DEPRECATED and SUPERSEDED** by ``harmonics_fn``.
        Highest used for the harmonic basis.
    :type max_degree: int
    """

    def __init__(self, r_min, r_max, check_every=None, var_names=None, shape=(10, 10, 10), r_scale='linear',
                 harmonics_fn=None, theta_min=0.0, theta_max=math.pi, phi_min=0.0, phi_max=math.pi * 2,
                 # DEPRECATED
                 max_degree=None):
        super(MonitorSphericalHarmonics, self).__init__(
            r_min,
            r_max,
            check_every=check_every,
            var_names=var_names,
            shape=shape,
            r_scale=r_scale,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
        )

        if (harmonics_fn is None) and (max_degree is None):
            raise ValueError("harmonics_fn should be specified")

        if max_degree is not None:
            warnings.warn("`max_degree` is DEPRECATED; pass `harmonics_fn` instead, which takes precedence")
            self.harmonics_fn = _RealSphericalHarmonics(max_degree=max_degree)

        if harmonics_fn is not None:
            self.harmonics_fn = harmonics_fn

    def _compute_us(self, nets, conditions):
        r, theta, phi = self.r_tensor, self.theta_tensor, self.phi_tensor
        us = []
        for net, cond in zip(nets, conditions):
            products = cond.enforce(net, r) * self.harmonics_fn(theta, phi)
            u = torch.sum(products, dim=1, keepdim=True).detach().cpu().numpy()
            us.append(u)
        return us

    @property
    def max_degree(self):
        try:
            ret = self.harmonics_fn.max_degree
        except AttributeError as e:
            warnings.warn(f"Error caught when accessing {self.__class__.__name__}, returning None:\n{e}")
            ret = None
        return ret


class Monitor1D(BaseMonitor):
    """A monitor for checking the status of the neural network during training.

    :param t_min:
        The lower bound of time domain that we want to monitor.
    :type t_min: float
    :param t_max:
        The upper bound of time domain that we want to monitor.
    :type t_max: float
    :param check_every:
        The frequency of checking the neural network represented by the number of epochs between two checks.
        Defaults to 100.
    :type check_every: int, optional
    """

    def __init__(self, t_min, t_max, check_every=None):
        """Initializer method
        """
        super(Monitor1D, self).__init__(check_every=check_every)
        self.fig = plt.figure(figsize=(30, 8))
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        # input for plotting
        self.ts_plt = np.linspace(t_min, t_max, 100)
        # input for neural network
        self.ts_ann = torch.linspace(t_min, t_max, 100, requires_grad=True).reshape((-1, 1))

    def check(self, nets, conditions, history):
        r"""Draw 2 plots: One shows the shape of the current solution.
        The other shows the history training loss and validation loss.

        :param nets:
            The neural networks that approximates the ODE (system).
        :type nets: list[`torch.nn.Module`]
        :param conditions:
            The initial/boundary conditions of the ODE (system).
        :type conditions: list[`neurodiffeq.ode.BaseCondition`]
        :param history:
            The history of training loss and validation loss.
            The 'train_loss' entry is a list of training loss and 'valid_loss' entry is a list of validation loss.
        :type history: dict['train': list[float], 'valid': list[float]]

        .. note::
            `check` is meant to be called by the function `solve` and `solve_system`.
        """
        us = [
            cond.enforce(net, self.ts_ann).detach().cpu().numpy()
            for cond, net in zip(conditions, nets)
        ]

        self.ax1.clear()
        for i, u in enumerate(us):
            self.ax1.plot(self.ts_plt, u, label=f'variable {i}')
        self.ax1.legend()
        self.ax1.set_title('solutions')

        self.ax2.clear()
        self.ax2.plot(history['train_loss'], label='training loss')
        self.ax2.plot(history['valid_loss'], label='validation loss')
        self.ax2.set_title('loss during training')
        self.ax2.set_ylabel('loss')
        self.ax2.set_xlabel('epochs')
        self.ax2.set_yscale('log')
        self.ax2.legend()

        self.ax3.clear()
        for metric_name, metric_values in history.items():
            if metric_name == 'train_loss' or metric_name == 'valid_loss':
                continue
            self.ax3.plot(metric_values, label=metric_name)
        self.ax3.set_title('metrics during training')
        self.ax3.set_ylabel('metrics')
        self.ax3.set_xlabel('epochs')
        self.ax3.set_yscale('log')
        # if there's not custom metrics, then there won't be any labels in this axis
        if len(history) > 2:
            self.ax3.legend()

        self.fig.canvas.draw()
        if not self.using_non_gui_backend:
            plt.pause(0.05)


class Monitor2D(BaseMonitor):
    r"""A monitor for checking the status of the neural network during training.
    The number and layout of subplots (matplotlib axes) will be finalized after the first ``.check()`` call.

    :param xy_min:
        The lower bound of 2 dimensions.
        If we only care about :math:`x \geq x_0` and :math:`y \geq y_0`, then `xy_min` is `(x_0, y_0)`.
    :type xy_min: tuple[float, float], optional
    :param xy_max:
        The upper bound of 2 dimensions.
        If we only care about :math:`x \leq x_1` and :math:`y \leq y_1`, then `xy_min` is `(x_1, y_1)`.
    :type xy_max: tuple[float, float], optional
    :param check_every:
        The frequency of checking the neural network represented by the number of epochs between two checks.
        Defaults to 100.
    :type check_every: int, optional
    :param valid_generator:
        The generator used to sample points from the domain when visualizing the solution.
        The generator is only called once (during instantiating the generator), and its outputs are stored.
        Defaults to a 32x32 ``Generator2D`` with method 'equally-spaced'.
    :type valid_generator: neurodiffeq.generators.BaseGenerator
    :param solution_style:

        - If set to 'heatmap', solution visualization will be a contour heat map of
          :math:`u` w.r.t. :math:`x` and :math:`y`. Useful when visualizing a 2-D spatial solution.
        - If set to 'curves', solution visualization will be :math:`u`-:math:`x` curves instead of a 2d heat map.
          Each curve corresponds to a :math:`t` value. Useful when visualizing 1D spatio-temporal solution.
          The first coordinate is interpreted as :math:`x` and the second as :math:`t`.

        Defaults to 'heatmap'.
    :type solution_style: str
    :param equal_aspect:
        Whether to set aspect ratio to 1:1 for heatmap. Defaults to True.
        Ignored if `solutions_style` is 'curves'.
    :type equal_aspect: bool
    :param ax_width:
        Width for each solution visualization. Note that this is different from width for metrics history,
        which is equal to ``ax_width`` :math:`\times` ``n_cols``.
    :type ax_width: float
    :param ax_height: Height for each solution visualization and metrics history plot.
    :type ax_height: float
    :param n_col:
        Number of solution visualizations to plot in each row.
        Note there is always only 1 plot for metrics history plot per row.
    :type n_col: int
    :param levels: Number of levels to plot with contourf (heatmap). Defaults to 20.
    :type levels: int
    """

    def __init__(self, xy_min, xy_max, check_every=None, valid_generator=None, solution_style='heatmap',
                 equal_aspect=True, ax_width=5.0, ax_height=4.0, n_col=2, levels=20):
        """Initializer method
        """
        super(Monitor2D, self).__init__(check_every=check_every)
        if solution_style not in ['heatmap', 'curves']:
            raise ValueError(f"Unsupported 'solution_style' = {solution_style}")
        if not _updatable_contour_plot_available() and solution_style == 'heatmap':
            warnings.warn("Heatmap-style solution does not work with your matplotlib version. "
                          "Please upgrade matplotlib to v3.3.0 or higher. "
                          "Otherwise you may experience buggy behavior.",
                          UserWarning)
        self.solution_style = solution_style
        self.fig = None
        self.ax_width = ax_width
        self.ax_height = ax_height
        self.n_col = n_col
        self.equal_aspect = equal_aspect
        self.axs = []  # subplots
        # self.caxs = []  # colorbars
        self.cbs = []  # color bars
        # input for neural network
        if valid_generator is None:
            valid_generator = _Generator2D([32, 32], xy_min, xy_max, method='equally-spaced')
        xs_ann, ys_ann = valid_generator.get_examples()
        self.xs_ann, self.ys_ann = xs_ann.reshape(-1, 1), ys_ann.reshape(-1, 1)
        self.xs_plot = self.xs_ann.detach().cpu().numpy().flatten()
        self.ys_plot = self.ys_ann.detach().cpu().numpy().flatten()
        self.levels = levels

    # draw a contour plot of the surface (xs, ys) -> zs
    def _create_contour(self, ax, xs, ys, zs, condition):
        triang = tri.Triangulation(xs, ys)
        xs = xs[triang.triangles].mean(axis=1)
        ys = ys[triang.triangles].mean(axis=1)
        if condition:
            xs, ys = torch.tensor(xs), torch.tensor(ys)
            if isinstance(condition, _IrregularBC):
                in_domain = condition.in_domain(xs, ys)
                triang.set_mask(~in_domain)

        contour = ax.tricontourf(triang, zs, cmap='coolwarm', levels=self.levels)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if self.equal_aspect:
            ax.set_aspect('equal', adjustable='box')
        return contour

    def check(self, nets, conditions, history):
        r"""Draw 2 plots: One shows the shape of the current solution (with heat map).
        The other shows the history training loss and validation loss.

        :param nets:
            The neural networks that approximates the PDE.
        :type nets: list [`torch.nn.Module`]
        :param conditions:
            The initial/boundary condition of the PDE.
        :type conditions: list [`neurodiffeq.conditions.BaseCondition`]
        :param history:
            The history of training loss and validation loss.
            The 'train' entry is a list of training loss and 'valid' entry is a list of validation loss.
        :type history: dict['train': list[float], 'valid': list[float]]

        .. note::
            `check` is meant to be called by the function `solve2D`.
        """

        if not self.fig:
            # initialize the figure and axes here so that the Monitor knows the number of dependent variables and
            # size of the figure, number of the subplots, etc.

            # one for each dependent variable, plus one for training and validation loss, plus one for metrics
            n_func = len(conditions)
            n_col = self.n_col
            n_row_sols = math.ceil(n_func / n_col)
            n_row = n_row_sols + 2
            self.fig = plt.figure(figsize=(self.ax_width * n_col, self.ax_height * n_row))
            self.fig.tight_layout()
            # axes and color bars for solutions (aka dependent variables)
            for i in range(n_func):
                self.axs.append(self.fig.add_subplot(n_row, n_col, i + 1))
                self.cbs.append(None)
            # axes for history plot of loss and other metrics, these plots should take the whole row
            self.axs.append(self.fig.add_subplot(n_row, 1, n_row_sols + 1))
            self.axs.append(self.fig.add_subplot(n_row, 1, n_row_sols + 2))

        us = [
            con.enforce(net, self.xs_ann, self.ys_ann)
            for con, net in zip(conditions, nets)
        ]

        for i, (ax, u, con) in enumerate(zip(self.axs[:-2], us, conditions)):
            ax.clear()
            u = u.detach().cpu().numpy().flatten()
            if self.solution_style == 'heatmap':
                cs = self._create_contour(ax, self.xs_plot, self.ys_plot, u, con)
                if self.cbs[i] is not None:
                    self.cbs[i].remove()
                self.cbs[i] = self.fig.colorbar(cs, format='%.0e', ax=ax)
                ax.set_title(f'u[{i}](x, y)')
            elif self.solution_style == 'curves':
                df = pd.DataFrame(dict(u=u, x=self.xs_plot, t=self.ys_plot))
                sns.lineplot(x='x', y='u', data=df, hue='t', ax=ax)
                ax.set_title(f'u[{i}](x) across different t')

        self.axs[-2].clear()
        self.axs[-2].plot(history['train_loss'], label='training loss')
        self.axs[-2].plot(history['valid_loss'], label='validation loss')
        self.axs[-2].set_title('loss during training')
        self.axs[-2].set_ylabel('loss')
        self.axs[-2].set_xlabel('epochs')
        self.axs[-2].set_yscale('log')
        self.axs[-2].legend()

        self.axs[-1].clear()
        for metric_name, metric_values in history.items():
            if metric_name == 'train_loss' or metric_name == 'valid_loss':
                continue
            self.axs[-1].plot(metric_values, label=metric_name)
        self.axs[-1].set_title('metrics during training')
        self.axs[-1].set_ylabel('metrics')
        self.axs[-1].set_xlabel('epochs')
        self.axs[-1].set_yscale('log')
        # if there's not custom metrics, then there won't be any labels in this axis
        if len(history) > 2:
            self.axs[-1].legend()

        self.fig.canvas.draw()
        if not self.using_non_gui_backend:
            plt.pause(0.05)


class MetricsMonitor(BaseMonitor):
    r"""A monitor for visualizing the loss and other metrics.
    This monitor does not visualize the solution.

    :param check_every:
        The frequency of checking the neural network represented by the number of epochs between two checks.
        Defaults to 100.
    :type check_every: int, optional
    """

    def __init__(self, check_every=None):
        super().__init__(check_every=check_every)
        self.fig = plt.figure(figsize=(12, 6), dpi=125)
        self.ax1, self.ax2 = self.fig.subplots(1, 2)

    def check(self, nets, conditions, history):
        self.ax1.clear()
        self.ax1.plot(history['train_loss'], label='training loss')
        self.ax1.plot(history['valid_loss'], label='validation loss')
        self.ax1.set_title('loss during training')
        self.ax1.set_ylabel('loss')
        self.ax1.set_xlabel('epochs')
        self.ax1.set_yscale('log')
        self.ax1.legend()

        self.ax2.clear()
        for metric_name, metric_values in history.items():
            if metric_name == 'train_loss' or metric_name == 'valid_loss':
                continue
            self.ax2.plot(metric_values, label=metric_name)
        self.ax2.set_title('metrics during training')
        self.ax2.set_ylabel('metrics')
        self.ax2.set_xlabel('epochs')
        self.ax2.set_yscale('log')
        # if there're not custom metrics, then there won't be any labels in this axis
        if len(history) > 2:
            self.ax2.legend()

        self.fig.canvas.draw()
        if not self.using_non_gui_backend:
            plt.pause(0.05)


class StreamPlotMonitor2D(BaseMonitor):
    def __init__(self, xy_min, xy_max, pairs, nx=32, ny=32, check_every=None, mask_fn=None,
                 ax_width=13.0, ax_height=10.0, n_col=2, stream_kwargs=None, equal_aspect=True, field_names=None):
        super(StreamPlotMonitor2D, self).__init__(check_every=check_every)
        self.pairs = pairs
        self.field_names = field_names or [f'Field[{i}]' for i, _ in enumerate(pairs)]
        if len(self.field_names) != len(self.pairs):
            raise ValueError(f"Length of field_names ({len(self.field_names)}) != Length of pairs({len(self.pairs)})")
        n_row = int(np.ceil(len(self.pairs) / n_col))
        self.nx, self.ny = nx, ny
        self.fig = plt.figure(figsize=(n_col * ax_width, n_row * ax_height))
        self.axes = self.fig.subplots(n_row, n_col).reshape(-1)
        self.cbs = [None] * len(pairs)  # colorbars
        _xs_ann, _ys_ann = torch.meshgrid([
            torch.linspace(xy_min[0], xy_max[0], nx, requires_grad=True),
            torch.linspace(xy_min[1], xy_max[1], ny, requires_grad=True),
        ])
        self.xs_ann, self.ys_ann = _xs_ann.reshape(-1, 1), _ys_ann.reshape(-1, 1)
        self.xs_plot = _xs_ann.detach().cpu().numpy()
        self.ys_plot = _ys_ann.detach().cpu().numpy()
        self.xlim = xy_min[0], xy_max[0]
        self.ylim = xy_min[1], xy_max[1]

        if mask_fn:
            self.mask = mask_fn(self.xs_plot, self.ys_plot)
            # TODO use antialiasing
            _pcolor_x, _pcolor_y = np.meshgrid(
                np.linspace(xy_min[0], xy_max[0], nx * 8),
                np.linspace(xy_min[1], xy_max[1], ny * 8),
            )
            _pcolor_mask = mask_fn(_pcolor_x, _pcolor_y)
            self._pcolor_args = (_pcolor_x, _pcolor_y, ~_pcolor_mask)
        else:
            self.mask = None
            self._pcolor_args = ()
        self.stream_kwargs = dict(
            density=(self.nx / 30, self.ny / 30),
        )
        self.stream_kwargs.update(stream_kwargs or {})
        self.equal_aspect = equal_aspect

    def _plot_streamlines(self, ax, us, vs, norms, cb_idx, is_grad=False):
        ax.clear()
        if self.mask is not None:
            # FIXME if mask covers all points in the meshgrid, the following ValueError will be raised
            # "Need at least one array to concatenate"
            us[~self.mask] = np.nan
            vs[~self.mask] = np.nan
            ax.pcolor(*self._pcolor_args, shading='auto', cmap='Purples')
        kwargs = dict(color=norms.transpose())
        kwargs.update(self.stream_kwargs)
        stream = ax.streamplot(self.xs_plot[:, 0], self.ys_plot[0, :], us.transpose(), vs.transpose(), **kwargs)
        if self.cbs[cb_idx] is not None:
            self.cbs[cb_idx].remove()
        self.cbs[cb_idx] = self.fig.colorbar(stream.lines, ax=ax)
        if self.equal_aspect:
            ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        if is_grad:
            ax.set_title(f'Gradient of {self.field_names[cb_idx]}')
        else:
            ax.set_title(f'Stream Plot of {self.field_names[cb_idx]}')

    def check(self, nets, conditions, history):
        for idx, pair in enumerate(self.pairs):
            if isinstance(pair, int):
                p = conditions[pair].enforce(nets[pair], self.xs_ann, self.ys_ann)
                us, vs = grad(p, self.xs_ann, self.ys_ann)
                us = us.reshape(self.nx, self.ny)
                vs = vs.reshape(self.nx, self.ny)
                is_grad = True
            else:
                ui, vi = pair
                us = conditions[ui].enforce(nets[ui], self.xs_ann, self.ys_ann).reshape(self.nx, self.ny)
                vs = conditions[vi].enforce(nets[vi], self.xs_ann, self.ys_ann).reshape(self.nx, self.ny)
                is_grad = False

            norms = torch.sqrt(us ** 2 + vs ** 2)

            self._plot_streamlines(
                ax=self.axes[idx],
                us=us.detach().cpu().numpy(),
                vs=vs.detach().cpu().numpy(),
                norms=norms.detach().cpu().numpy(),
                cb_idx=idx,
                is_grad=is_grad,
            )
