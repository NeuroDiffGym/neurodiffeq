# Consider these convention:
# For user facing part, use 1-d tensor (when representing a collection of 2-d points, use 2 1-d tensors).
# For non-user facing part, avoid reshaping and use 1-d tensor as much as possible.
# In function signatures, let u comes before x, let x comes before t
# Use x and t for distinguish values for x or t; Use xx and tt when corresponding entries of xx and tt are
# supposed to be paired to represent a point. ([xx, tt] is often the Cartesian product of x and t)
from abc import ABC, abstractmethod
import torch
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from copy import deepcopy

# return the Cartesian product of x and t.
def _cartesian_prod_dims(x, t, x_grad=True, t_grad=True):
    xt = torch.cartesian_prod(x, t)
    xx = torch.squeeze(xt[:, 0])
    xx.requires_grad = x_grad
    tt = torch.squeeze(xt[:, 1])
    tt.requires_grad = t_grad
    return xx, tt

class Approximator(ABC):
    """The base class of approximators. An approximator is an approximation of the
    differential equation's solution. It knows the parameters in the neural network, 
    and how to calculate the loss function and the metrics.
    """
    @abstractmethod
    def __call__(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def parameters(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def calculate_loss(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def calculate_metrics(self):
        raise NotImplementedError  # pragma: no cover


class SingleNetworkApproximator1DSpatialTemporal(Approximator):
    """An approximator to approximate the solution of a 1D time-dependent problem.
    The boundary condition will be enforced by a regularization term in the loss function
    and the initial condition will be enforced by transforming the output of the
    neural network.

    :param single_network: A neural network with 2 input nodes (x, t) and 1 output node
    :type single_network: `torch.nn.Module`
    :param pde: The PDE to solve. If the PDE is :math:`F(u, x, t) = 0` then `pde` 
        should be a function that maps :math:`(u, x, t)` to :math:`F(u, x, t)`.
    :type pde: function
    :param initial_condition: A first order initial condition
    :type initial_condition: `temporal.FirstOrderInitialCondition`
    :param boundary_conditions: A list of boundary conditions
    :type boundary_conditions: list[`temporal.BoundaryCondition`]
    :param boundary_strictness: The regularization parameter, defaults to 1.
        a larger regularization parameter enforces the boundary conditions more strictly.
    :type boundary_strictness: float
    """
    def __init__(self, single_network, pde, initial_condition, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, tt):
        xx = torch.unsqueeze(xx, dim=1)
        tt = torch.unsqueeze(tt, dim=1)
        xt = torch.cat((xx, tt), dim=1)
        uu = torch.exp(-tt) * self.initial_condition.u0(xx) + (1 - torch.exp(-tt)) * self.single_network(xt)
        return torch.squeeze(uu)

    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, tt, x, t):
        uu = self.__call__(xx, tt)

        equation_mse = torch.mean(self.pde(uu, xx, tt)**2)

        boundary_mse = self.boundary_strictness * sum(self._boundary_mse(t, bc) for bc in self.boundary_conditions)

        return equation_mse + boundary_mse

    def _boundary_mse(self, t, bc):
        x = next(bc.points_generator)

        xx, tt = _cartesian_prod_dims(x, t, x_grad=True, t_grad=False)
        uu = self.__call__(xx, tt)
        return torch.mean(bc.form(uu, xx, tt)**2)

    def calculate_metrics(self, xx, tt, x, t, metrics):
        uu = self.__call__(xx, tt)

        return {
            metric_name: metric_func(uu, xx, tt)
            for metric_name, metric_func in metrics.items()
        }


class SingleNetworkApproximator2DSpatial(Approximator):
    """An approximator to approximate the solution of a 2D steady-state problem.
    The boundary condition will be enforced by a regularization term in the loss function.

    :param single_network: A neural network with 2 input nodes (x, y) and 1 output node
    :type single_network: `torch.nn.Module`
    :param pde: The PDE to solve. If the PDE is :math:`F(u, x, y) = 0` then `pde` 
        should be a function that maps :math:`(u, x, y)` to :math:`F(u, x, y)`.
    :type pde: function
    :param boundary_conditions: A list of boundary conditions
    :type boundary_conditions: list[`temporal.BoundaryCondition`]
    :param boundary_strictness: The regularization parameter, defaults to 1.
        a larger regularization parameter enforces the boundary conditions more strictly.
    :type boundary_strictness: float
    """
    def __init__(self, single_network, pde, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, yy):
        xx = torch.unsqueeze(xx, dim=1)
        yy = torch.unsqueeze(yy, dim=1)
        xy = torch.cat((xx, yy), dim=1)
        uu = self.single_network(xy)
        return torch.squeeze(uu)

    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy):
        uu = self.__call__(xx, yy)

        equation_mse = torch.mean(self.pde(uu, xx, yy)**2)

        boundary_mse = self.boundary_strictness * sum(self._boundary_mse(bc) for bc in self.boundary_conditions)

        return equation_mse + boundary_mse

    def _boundary_mse(self, bc):
        xx, yy = next(bc.points_generator)
        uu = self.__call__(xx, yy)
        return torch.mean(bc.form(uu, xx, yy) ** 2)

    def calculate_metrics(self, xx, yy, metrics):
        uu = self.__call__(xx, yy)

        return {
            metric_name: metric_func(uu, xx, yy)
            for metric_name, metric_func in metrics.items()
        }


class SingleNetworkApproximator2DSpatialSystem(Approximator):
    """An approximator to approximate the solution of a 2D steady-state differential equation system.
    The boundary condition will be enforced by a regularization term in the loss function.

    :param single_network: A neural network with 2 input nodes (x, y) and n output node (n is the number of
        dependent variables in the differential equation system)
    :type single_network: `torch.nn.Module`
    :param pde: The PDE system to solve. If the PDE is :math:`F_i(u_1, u_2, ..., u_n, x, y) = 0`
        where :math:`u_i` is the i-th dependent variable,
        then `pde` should be a function that maps :math:`(u_1, u_2, ..., u_n, x, y)` to
        a list where the i-th entry is :math:`F_i(u_1, u_2, ..., u_n, x, y)`.
    :type pde: function
    :param boundary_conditions: A list of boundary conditions
    :type boundary_conditions: list[`temporal.BoundaryCondition`]
    :param boundary_strictness: The regularization parameter, defaults to 1.
        a larger regularization parameter enforces the boundary conditions more strictly.
    :type boundary_strictness: float
    """
    def __init__(self, single_network, pde, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, yy):
        xx = torch.unsqueeze(xx, dim=1)
        yy = torch.unsqueeze(yy, dim=1)
        xy = torch.cat((xx, yy), dim=1)
        uu = self.single_network(xy)
        return tuple(torch.squeeze(uu[:, i]) for i in range(uu.shape[1]))

    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy):
        uu = self.__call__(xx, yy)

        equation_mse = sum(
            torch.mean(eq**2)
            for eq in self.pde(*uu, xx, yy)
        )

        boundary_mse = self.boundary_strictness * sum(self._boundary_mse(bc) for bc in self.boundary_conditions)

        return equation_mse + boundary_mse

    def _boundary_mse(self, bc):
        xx, yy = next(bc.points_generator)
        uu = self.__call__(xx, yy)
        return torch.mean(bc.form(*uu, xx, yy) ** 2)

    def calculate_metrics(self, xx, yy, metrics):
        uu = self.__call__(xx, yy)

        return {
            metric_name: metric_func(*uu, xx, yy)
            for metric_name, metric_func in metrics.items()
        }


class SingleNetworkApproximator2DSpatialTemporal(Approximator):
    """An approximator to approximate the solution of a 2D time-dependent problem.
    The boundary condition will be enforced by a regularization term in the loss function
    and the initial condition will be enforced by transforming the output of the
    neural network.

    :param single_network: A neural network with 3 input nodes (x, y, t) and 1 output node.
    :type single_network: `torch.nn.Module`
    :param pde: The PDE system to solve. If the PDE is :math:`F(u, x, y, t) = 0`
        then `pde` should be a function that maps :math:`(u, x, y, t)` to :math:`F(u, x, y, t)`.
    :type pde: function
    :param initial_condition: A first order initial condition
    :type initial_condition: `temporal.FirstOrderInitialCondition` or `temporal.SecondOrderInitialCondition`
    :param boundary_conditions: A list of boundary conditions
    :type boundary_conditions: list[`temporal.BoundaryCondition`]
    :param boundary_strictness: The regularization parameter, defaults to 1.
        a larger regularization parameter enforces the boundary conditions more strictly.
    :type boundary_strictness: float
    """
    def __init__(self, single_network, pde, initial_condition, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        self.u0 = initial_condition.u0
        self.u0dot = initial_condition.u0dot if hasattr(initial_condition, 'u0dot') else None
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, yy, tt):
        xx = torch.unsqueeze(xx, dim=1)
        yy = torch.unsqueeze(yy, dim=1)
        tt = torch.unsqueeze(tt, dim=1)
        xyt = torch.cat((xx, yy, tt), dim=1)
        if self.u0dot is None:
            uu = torch.exp(-tt) * self.u0(xx, yy) + (1 - torch.exp(-tt)) * self.single_network(xyt)
        else:
            # not sure about this line
            uu = (1 - (1 - torch.exp(-tt))**2) * self.u0(xx, yy) + (1 - torch.exp(-tt)) * self.u0dot(xx, yy) + (1 - torch.exp(-tt))**2 * self.single_network(xyt)
        return torch.squeeze(uu)

    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy, tt, x, y, t):
        uu = self.__call__(xx, yy, tt)

        equation_mse = torch.mean(self.pde(uu, xx, yy, tt)**2)

        boundary_mse = self.boundary_strictness * sum(self._boundary_mse(t, bc) for bc in self.boundary_conditions)

        return equation_mse + boundary_mse

    def _boundary_mse(self, t, bc):
        x, y = next(bc.points_generator)

        xx, tt = _cartesian_prod_dims(x, t, x_grad=True, t_grad=False)
        yy, tt = _cartesian_prod_dims(y, t, x_grad=True, t_grad=False)
        uu = self.__call__(xx, yy, tt)
        return torch.mean(bc.form(uu, xx, yy, tt) ** 2)

    def calculate_metrics(self, xx, yy, tt, x, y, t, metrics):
        uu = self.__call__(xx, yy, tt)

        return {
            metric_name: metric_func(uu, xx, yy, tt)
            for metric_name, metric_func in metrics.items()
        }


class FirstOrderInitialCondition:
    """A first order initial condition. It is used to initialize ``temporal.Approximator``\s.

    :param u0: A function representing the initial condition. If we are solving for
        is :math:`u`, then `u0` is :math:`u\\bigg|_{t=0}`. The input of the function
        dependes on where it is used. If it is used as the input for
        `temporal.SingleNetworkApproximator1DSpatialTemporal`, then `u0` should map
        :math:`x` to :math:`u(x, t)\\bigg|_{t = 0}`. If it is used as the input for
        `temporal.SingleNetworkApproximator2DSpatialTemporal`, then `u0` should map
        :math:`(x, y)` to :math:`u(x, y, t)\\bigg|_{t = 0}`.
    :type u0: function
    """
    def __init__(self, u0):
        self.u0 = u0


class SecondOrderInitialCondition:
    """A second order initial condition. It is used to initialize ``temporal.Approximator``\s.

    :param u0: A function representing the initial condition. If we are solving for
        is :math:`u`, then ``u0`` is :math:`u\\bigg|_{t=0}`. The input of the function
        dependes on where it is used. If it is used as the input for
        ``temporal.SingleNetworkApproximator1DSpatialTemporal``, then ``u0`` should map
        :math:`x` to :math:`u(x, t)\\bigg|_{t = 0}`. If it is used as the input for
        ``temporal.SingleNetworkApproximator2DSpatialTemporal``, then ``u0`` should map
        :math:`(x, y)` to :math:`u(x, y, t)\\bigg|_{t = 0}`.
    :type u0: function
    :param u0dot: A function representing the initial derivative w.r.t. time. If we are solving for
        is :math:``u``, then ``u0dot`` is :math:`\\dfrac{\\partial u}{\\partial t}\\bigg|_{t=0}`.
        The input of the function
        depends on where it is used. If it is used as the input for
        ``temporal.SingleNetworkApproximator1DSpatialTemporal``, then ``u0`` should map
        :math:`x` to :math:`\\dfrac{\\partial u}{\\partial t}\\bigg|_{t = 0}`. If it is used as the input for
        `temporal.SingleNetworkApproximator2DSpatialTemporal`, then ``u0`` should map
        :math:`(x, y)` to :math:`\\dfrac{\\partial u}{\\partial t}\\bigg|_{t = 0}`.
    :type u0dot: function
    """
    def __init__(self, u0, u0dot):
        self.u0 = u0
        self.u0dot = u0dot


class BoundaryCondition:
    """A boundary condition. It is used to initialize ``temporal.Approximator``\s.

    :param form: The form of the boundary condition. For a 1D time-dependent problem, if the boundary condition demands that :math:`B(u, x) = 0`, then ``form`` should be a function that maps :math:`u, x, t` to :math:`B(u, x)`. For a 2D steady-state problem, if the boundary condition demands that :math:`B(u, x, y) = 0`, then ``form`` should be a function that maps :math:`u, x, y` to :math:`B(u, x, y)`. For a 2D steady-state system, if the boundary condition demands that :math:`B(u_i, x, y) = 0`, then `form` should be a function that maps :math:`u_1, u_2, ..., u_n, x, y` to `B(u_i, x, y)`. For 2D time-dependent problem, if the boundary condition demands that :math:`B(u, x, y) = 0`, then `form` should be a function that maps :math:`u, x, y, t` to `B(u_i, x, y)`. Basically the function signature of ``form`` should be the same as the ``pde`` function of the given ``temporal.Approximator``.
    :type form: callable
    :param points_generator: A generator that generates points on the boundary.
        It can be a `temporal.generator_1dspatial`, `temporal.generator_2dspatial_segment`,
        or a generator written by user.
    :type points_genrator: generator
    """
    def __init__(self, form, points_generator):
        self.form = form
        self.points_generator = points_generator


def generator_1dspatial(size, x_min, x_max, random=True):
    """Return a generator that generates 1D points range from x_min to x_max

    :param size: number of points to generated when `__next__` is invoked
    :type size: int
    :param x_min: Lower bound of x
    :type x_min: float
    :param x_max: Upper bound of x
    :type x_max: float
    :param random: If set to False, then return eqally spaced points range from
        x_min to x_max. If set to True then generate points randomly. Defaults to True
    :type random: bool
    """
    seg_len = (x_max-x_min) / size
    linspace_lo = x_min + seg_len*0.5
    linspace_hi = x_max - seg_len*0.5
    center = torch.linspace(linspace_lo, linspace_hi, size)
    noise_lo = -seg_len*0.5
    while True:
        if random:
            noise = seg_len*torch.rand(size) + noise_lo
            yield center + noise
        else:
            yield center


def generator_2dspatial_segment(size, start, end, random=True):
    """Return a generator that generates 2D points in a line segment.

    :param size: number of points to generated when `__next__` is invoked
    :type size: int
    :param x_min: Lower bound of x
    :type x_min: float
    :param x_max: Upper bound of x
    :type x_max: float
    :param y_min: Lower bound of y
    :type y_min: float
    :param y_max: Upper bound of y
    :type y_max: float
    :param random: If set to False, then return a grid where the points are eqally
        spaced in the x and y dimension. If set to True then generate points randomly. 
        Defaults to True.
    :type random: bool
    """
    x1, y1 = start
    x2, y2 = end
    step = 1./size
    center = torch.linspace(0. + 0.5*step, 1. - 0.5*step, size)
    noise_lo = -step*0.5
    while True:
        if random:
            noise = step*torch.rand(size) + noise_lo
            center = center + noise
        yield x1 + (x2-x1)*center, y1 + (y2-y1)*center


def generator_2dspatial_rectangle(size, x_min, x_max, y_min, y_max, random=True):
    """Return a generator that generates 2D points in a rectangle.

    :param size: number of points to generated when `__next__` is invoked
    :type size: int
    :param start: the starting point of the line segment
    :type start: tuple[float, float]
    :param end: the ending point of the line segment
    :type end: tuple[float, float]
    :param random: If set to False, then return eqally spaced points range from
        `start` to `end`. If set to Rrue then generate points randomly. Defaults to True.
    :type random: bool
    """
    x_size, y_size = size
    x_generator = generator_1dspatial(x_size, x_min, x_max, random)
    y_generator = generator_1dspatial(y_size, y_min, y_max, random)
    while True:
        x = next(x_generator)
        y = next(y_generator)
        xy = torch.cartesian_prod(x, y)
        xx = torch.squeeze(xy[:, 0])
        yy = torch.squeeze(xy[:, 1])
        yield xx, yy


def generator_temporal(size, t_min, t_max, random=True):
    """Return a generator that generates 1D points range from t_min to t_max

    :param size: number of points to generated when `__next__` is invoked
    :type size: int
    :param t_min: Lower bound of t
    :type t_min: float
    :param t_max: Upper bound of t
    :type t_max: float
    :param random: If set to False, then return eqally spaced points range from
        t_min to t_max. If set to True then generate points randomly. Defaults to True
    :type random: bool
    """
    seg_len = (t_max - t_min) / size
    linspace_lo = t_min + seg_len * 0.5
    linspace_hi = t_max - seg_len * 0.5
    center = torch.linspace(linspace_lo, linspace_hi, size)
    noise_lo = -seg_len * 0.5
    while True:
        if random:
            noise = seg_len * torch.rand(size) + noise_lo
            yield center + noise
        else:
            yield center


class MonitorMinimal:
    """A monitor that shows the loss function and custom metrics.
    """
    def __init__(self, check_every):
        self.using_non_gui_backend = matplotlib.get_backend() == 'agg'
        self.check_every = check_every

        self.fig = plt.figure(figsize=(20, 8))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

    def check(self, approximator, history):

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
        # if there's not custom metrics, then there won't be any labels in this axis
        if len(history) > 2:
            self.ax2.legend()

        self.fig.canvas.draw()
        if not self.using_non_gui_backend:
            plt.pause(0.05)   # pragma: no cover (we are not using gui backend for testing)


class Monitor1DSpatialTemporal:
    """A monitor for 1D time-dependent problems.
    """
    def __init__(self, check_on_x, check_on_t, check_every):
        self.using_non_gui_backend = matplotlib.get_backend() == 'agg'

        self.xx_tensor, self.tt_tensor = _cartesian_prod_dims(check_on_x, check_on_t)
        self.x_array = check_on_x.clone().detach().cpu().numpy()
        self.t_array = check_on_t.clone().detach().cpu().numpy()
        self.check_every = check_every
        self.t_color = torch.linspace(0, 1, len(check_on_t)).detach().cpu().numpy()

        self.fig = plt.figure(figsize=(30, 8))
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

    def check(self, approximator, history):
        uu_array = approximator(self.xx_tensor, self.tt_tensor).detach().cpu().numpy()

        self.ax1.clear()
        for i, t_c in enumerate(zip(self.t_array, self.t_color)):
            u_t = uu_array[i::len(self.t_array)]
            t, c = t_c
            t = float(t)
            c = cm.viridis(c)

            self.ax1.plot(self.x_array, u_t, color=c, label=f't = {t:.2E}')
        self.ax1.legend()
        self.ax1.set_title('approximation')

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
            plt.pause(0.05)   # pragma: no cover (we are not using gui backend for testing)


class Monitor2DSpatialTemporal:
    """A monitor for 2D time-dependent problems.
    """
    def __init__(self, check_on_x, check_on_y, check_on_t, check_every):
        self.using_non_gui_backend = matplotlib.get_backend() == 'agg'

        xy_tensor = torch.cartesian_prod(check_on_x, check_on_y)
        self.xx_tensor = torch.squeeze(xy_tensor[:, 0])
        self.yy_tensor = torch.squeeze(xy_tensor[:, 1])
        self.tt_tensors = [
            torch.ones(len(xy_tensor)) * t
            for t in check_on_t
        ]
        self.xx_array = self.xx_tensor.clone().detach().cpu().numpy()
        self.yy_array = self.yy_tensor.clone().detach().cpu().numpy()
        self.t_array = check_on_t.clone().detach().cpu().numpy()
        self.check_every = check_every

        self.fig = None
        self.axs = []  # subplots
        self.cbs = []  # color bars

    @staticmethod
    def _create_contour(ax, xx, yy, uu):
        triang = tri.Triangulation(xx, yy)
        contour = ax.tricontourf(triang, uu, cmap='coolwarm')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        return contour

    def check(self, approximator, history):
        if not self.fig:
            # initialize the figure and axes here so that the Monitor knows the number of dependent variables and
            # size of the figure, number of the subplots, etc.
            n_axs = len(self.t_array)+2  # one for each time slice, plus one for training and validation loss, plus one for metrics
            n_row, n_col = (n_axs+1) // 2, 2
            self.fig = plt.figure(figsize=(20, 8*n_row))
            for i in range(n_axs):
                self.axs.append(self.fig.add_subplot(n_row, n_col, i+1))
            for i in range(n_axs-2):
                self.cbs.append(None)

        for i, ax in enumerate(self.axs[:-2]):
            ax.clear()
            uu_array = approximator(self.xx_tensor, self.yy_tensor, self.tt_tensors[i]).detach().cpu().numpy()
            cs = self._create_contour(ax, self.xx_array, self.yy_array, uu_array)
            if self.cbs[i] is None:
                self.cbs[i] = self.fig.colorbar(cs, format='%.0e', ax=ax)
            else:
                self.cbs[i].mappable.set_clim(vmin=uu_array.min(), vmax=uu_array.max())
            ax.set_title(f'approximation t = {self.t_array[i]:.2E}')

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
            plt.pause(0.05)  # pragma: no cover (we are not using gui backend for testing)


class Monitor2DSpatial:
    """A Monitor for 2D steady-state problems
    """
    def __init__(self, check_on_x, check_on_y, check_every):
        self.using_non_gui_backend = matplotlib.get_backend() == 'agg'

        xy_tensor = torch.cartesian_prod(check_on_x, check_on_y)
        self.xx_tensor = torch.squeeze(xy_tensor[:, 0])
        self.yy_tensor = torch.squeeze(xy_tensor[:, 1])

        self.xx_array = self.xx_tensor.clone().detach().cpu().numpy()
        self.yy_array = self.yy_tensor.clone().detach().cpu().numpy()

        self.check_every = check_every

        self.fig = plt.figure(figsize=(30, 8))
        self.ax1 = self.fig.add_subplot(131)
        self.cb1 = None
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)


    @staticmethod
    def _create_contour(ax, xx, yy, uu):
        triang = tri.Triangulation(xx, yy)
        contour = ax.tricontourf(triang, uu, cmap='coolwarm')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        return contour

    def check(self, approximator, history):
        self.ax1.clear()
        uu_array = approximator(self.xx_tensor, self.yy_tensor).detach().cpu().numpy()
        cs = self._create_contour(self.ax1, self.xx_array, self.yy_array, uu_array)
        if self.cb1 is None:
            self.cb1 = self.fig.colorbar(cs, format='%.0e', ax=self.ax1)
        else:
            self.cb1.mappable.set_clim(vmin=uu_array.min(), vmax=uu_array.max())
        self.ax1.set_title(f'approximation')

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
            plt.pause(0.05)  # pragma: no cover (we are not using gui backend for testing)


def _solve_1dspatial_temporal(
    train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor
):
    """Solve a 1D time-dependent problem

    :param train_generator_spatial: a generator to generate 1D spatial points for training
    :type train_generator_spatial: generator
    :param train_generator_temporal: a generator to generate 1D temporal points for training
    :type train_generator_temporal: generator
    :param valid_generator_spatial: a generator to generate 1D spatial points for validation
    :type valid_generator_spatial: generator
    :param valid_generator_temporal: a generator to generate 1D temporal points for validation
    :type valid_generator_temporal: generator
    :param approximator: an approximator for 1D time-dependent problem
    :type approximator: `temporal.SingleNetworkApproximator1DSpatialTemporal` or a custom `temporal.Approximator`
    :param optimizer: The optimization method to use for training
    :type optimizer: `torch.optim.Optimizer`
    :param batch_size: The size of the mini-batch to use
    :type batch_size: int
    :param max_epochs: The maximum number of epochs to train
    :type max_epochs: int
    :param shuffle: Whether to shuffle the training examples every epoch
    :type shuffle: bool
    :param metrics: Metrics to keep track of during training. The metrics should be passed as a dictionary where the keys are the names of the metrics, and the values are the corresponding function.
        The input functions should be the same as `pde` (of the approximator) and the output should be a numeric value. The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, function]
    :param monitor: The monitor to check the status of nerual network during training
    :type monitor: `temporal.Monitor1DSpatialTemporal` or `temporal.MonitorMinimal`
    """
    return _solve_spatial_temporal(
        train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
        approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,
        train_routine=_train_1dspatial_temporal, valid_routine=_valid_1dspatial_temporal
    )


def _solve_2dspatial_temporal(
    train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor
):
    """Solve a 2D time-dependent problem

    :param train_generator_spatial: a generator to generate 2D spatial points for training
    :type train_generator_spatial: generator
    :param train_generator_temporal: a generator to generate 1D temporal points for training
    :type train_generator_temporal: generator
    :param valid_generator_spatial: a generator to generate 2D spatial points for validation
    :type valid_generator_spatial: generator
    :param valid_generator_temporal: a generator to generate 1D temporal points for validation
    :type valid_generator_temporal: generator
    :param approximator: an approximator for 2D time-dependent problem
    :type approximator: `temporal.SingleNetworkApproximator2DSpatialTemporal` or a custom `temporal.Approximator`
    :param optimizer: The optimization method to use for training
    :type optimizer: `torch.optim.Optimizer`
    :param batch_size: The size of the mini-batch to use
    :type batch_size: int
    :param max_epochs: The maximum number of epochs to train
    :type max_epochs: int
    :param shuffle: Whether to shuffle the training examples every epoch
    :type shuffle: bool
    :param metrics: Metrics to keep track of during training. The metrics should be passed as a dictionary where the keys are the names of the metrics, and the values are the corresponding function.
        The input functions should be the same as `pde` (of the approximator) and the output should be a numeric value. The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, function]
    :param monitor: The monitor to check the status of nerual network during training
    :type monitor: `temporal.Monitor2DSpatialTemporal` or `temporal.MonitorMinimal`
    """
    return _solve_spatial_temporal(
        train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
        approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,
        train_routine=_train_2dspatial_temporal, valid_routine=_valid_2dspatial_temporal
    )

def _solve_2dspatial(
    train_generator_spatial, valid_generator_spatial,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor
):
    return _solve_spatial_temporal(
        train_generator_spatial, None, valid_generator_spatial, None,
        approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,
        train_routine=_train_2dspatial, valid_routine=_valid_2dspatial
    )
    """Solve a 2D steady-state problem

    :param train_generator_spatial: a generator to generate 2D spatial points for training
    :type train_generator_spatial: generator
    :param valid_generator_spatial: a generator to generate 2D spatial points for validation
    :type valid_generator_spatial: generator
    :param approximator: an approximator for 2D time-state problem
    :type approximator: `temporal.SingleNetworkApproximator2DSpatial`, `temporal.SingleNetworkApproximator2DSpatialSystem`, or a custom `temporal.Approximator`
    :param optimizer: The optimization method to use for training
    :type optimizer: `torch.optim.Optimizer`
    :param batch_size: The size of the mini-batch to use
    :type batch_size: int
    :param max_epochs: The maximum number of epochs to train
    :type max_epochs: int
    :param shuffle: Whether to shuffle the training examples every epoch
    :type shuffle: bool
    :param metrics: Metrics to keep track of during training. The metrics should be passed as a dictionary where the keys are the names of the metrics, and the values are the corresponding function.
        The input functions should be the same as `pde` (of the approximator) and the output should be a numeric value. The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, function]
    :param monitor: The monitor to check the status of nerual network during training
    :type monitor: `temporal.Monitor2DSpatial` or `temporal.MonitorMinimal`
    """


# _solve_1dspatial_temporal, _solve_2dspatial_temporal, _solve_2dspatial all call this function in the end
def _solve_spatial_temporal(
    train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,
    train_routine, valid_routine
):
    history = {'train_loss': [], 'valid_loss': []}
    for metric_name, _ in metrics.items():
        history['train_' + metric_name] = []
        history['valid_' + metric_name] = []

    for epoch in range(max_epochs):
        train_epoch_loss, train_epoch_metrics = train_routine(
            train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size
        )
        history['train_loss'].append(train_epoch_loss)
        for metric_name, metric_value in train_epoch_metrics.items():
            history['train_' + metric_name].append(metric_value)

        valid_epoch_loss, valid_epoch_metrics = valid_routine(
            valid_generator_spatial, valid_generator_temporal, approximator, metrics
        )
        history['valid_loss'].append(valid_epoch_loss)
        for metric_name, metric_value in valid_epoch_metrics.items():
            history['valid_' + metric_name].append(metric_value)

        if monitor and epoch % monitor.check_every == 0:
            monitor.check(approximator, history)

    return approximator, history


# training phase for 1D time-dependent problems
def _train_1dspatial_temporal(train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size):
    x = next(train_generator_spatial)
    t = next(train_generator_temporal)
    xx, tt = _cartesian_prod_dims(x, t)
    training_set_size = len(xx)
    idx = torch.randperm(training_set_size) if shuffle else torch.arange(training_set_size)

    batch_start, batch_end = 0, batch_size
    while batch_start < training_set_size:
        if batch_end > training_set_size:
            batch_end = training_set_size
        batch_idx = idx[batch_start:batch_end]
        batch_xx = xx[batch_idx]
        batch_tt = tt[batch_idx]

        batch_loss = approximator.calculate_loss(batch_xx, batch_tt, x, t)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_start += batch_size
        batch_end += batch_size

    epoch_loss = approximator.calculate_loss(xx, tt, x, t).item()

    epoch_metrics = approximator.calculate_metrics(xx, tt, x, t, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


# training phase for 2D steady-state problems
def _train_2dspatial(train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size):
    xx, yy = next(train_generator_spatial)
    xx.requires_grad = True
    yy.requires_grad = True
    training_set_size = len(xx)
    idx = torch.randperm(training_set_size) if shuffle else torch.arange(training_set_size)

    batch_start, batch_end = 0, batch_size
    while batch_start < training_set_size:
        if batch_end > training_set_size:
            batch_end = training_set_size
        batch_idx = idx[batch_start:batch_end]
        batch_xx = xx[batch_idx]
        batch_yy = yy[batch_idx]

        batch_loss = approximator.calculate_loss(batch_xx, batch_yy)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_start += batch_size
        batch_end += batch_size

    epoch_loss = approximator.calculate_loss(xx, yy).item()

    epoch_metrics = approximator.calculate_metrics(xx, yy, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


# validation phase for 2D steady-state problems
def _valid_2dspatial(valid_generator_spatial, valid_generator_temporal, approximator, metrics):
    xx, yy = next(valid_generator_spatial)
    xx.requires_grad = True
    yy.requires_grad = True

    epoch_loss = approximator.calculate_loss(xx, yy).item()

    epoch_metrics = approximator.calculate_metrics(xx, yy, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


# training phase for 2D time-dependent problems
def _train_2dspatial_temporal(train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size):
    x, y = next(train_generator_spatial)
    t = next(train_generator_temporal)
    xx, tt = _cartesian_prod_dims(x, t)
    yy, tt = _cartesian_prod_dims(y, t)
    training_set_size = len(xx)
    idx = torch.randperm(training_set_size) if shuffle else torch.arange(training_set_size)

    batch_start, batch_end = 0, batch_size
    while batch_start < training_set_size:
        if batch_end > training_set_size:
            batch_end = training_set_size
        batch_idx = idx[batch_start:batch_end]
        batch_xx = xx[batch_idx]
        batch_yy = yy[batch_idx]
        batch_tt = tt[batch_idx]

        batch_loss = approximator.calculate_loss(batch_xx, batch_yy, batch_tt, x, y, t)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_start += batch_size
        batch_end += batch_size

    # TODO: this can give us the real loss after an epoch, but can be very memory intensive
    epoch_loss = approximator.calculate_loss(xx, yy, tt, x, y, t).item()

    epoch_metrics = approximator.calculate_metrics(xx, yy, tt, x, y, t, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


# validation phase for 1D time-dependent problems
def _valid_1dspatial_temporal(valid_generator_spatial, valid_generator_temporal, approximator, metrics):
    x = next(valid_generator_spatial)
    t = next(valid_generator_temporal)
    xx, tt = _cartesian_prod_dims(x, t)

    epoch_loss = approximator.calculate_loss(xx, tt, x, t).item()

    epoch_metrics = approximator.calculate_metrics(xx, tt, x, t, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


# validation phase for 2D time-dependent problems
def _valid_2dspatial_temporal(valid_generator_spatial, valid_generator_temporal, approximator, metrics):
    x, y = next(valid_generator_spatial)
    t = next(valid_generator_temporal)
    xx, tt = _cartesian_prod_dims(x, t)
    yy, tt = _cartesian_prod_dims(y, t)

    epoch_loss = approximator.calculate_loss(xx, yy, tt, x, y, t).item()

    epoch_metrics = approximator.calculate_metrics(xx, yy, tt, x, y, t, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics
