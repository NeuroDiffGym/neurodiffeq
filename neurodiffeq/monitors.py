import math
import torch
import warnings
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

from ._version_utils import deprecated_alias
from .function_basis import RealSphericalHarmonics
from .generators import Generator3D


class BaseMonitor(ABC):
    @abstractmethod
    def check(self, nets, conditions, history):
        pass


class MonitorSpherical:
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

    def __init__(self, r_min, r_max, check_every=100, var_names=None, shape=(10, 10, 10), r_scale='linear',
                 theta_min=0.0, theta_max=math.pi, phi_min=0.0, phi_max=math.pi * 2):
        """Initializer method
        """
        self.contour_plot_available = self._matplotlib_version_satisfies()
        if not self.contour_plot_available:
            warnings.warn("Warning: contourf plot only available for matplotlib version >= v3.3.0 "
                          "switching to matshow instead")
        self.using_non_gui_backend = (matplotlib.get_backend() == 'agg')
        self.check_every = check_every
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

        gen = Generator3D(
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
    def _matplotlib_version_satisfies():
        from packaging.version import parse as vparse
        from matplotlib import __version__
        return vparse(__version__) >= vparse('3.3.0')

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
            cax = ax.contourf(phi, theta, u, cmap='magma')
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

    def __init__(self, r_min, r_max, check_every=100, var_names=None, shape=(10, 10, 10), r_scale='linear',
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
            self.harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)

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

