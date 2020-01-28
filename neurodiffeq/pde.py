import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .networks import FCNN
from .neurodiffeq import diff
from copy import deepcopy

def _network_output_2input(net, xs, ys, ith_unit):
    xys = torch.cat((xs, ys), 1)
    nn_output = net(xys)
    if ith_unit is not None:
        return nn_output[:, ith_unit].reshape(-1, 1)
    else:
        return nn_output

def _trial_solution_2input(single_net, nets, xs, ys, conditions):
    if single_net:  # using a single net
        us = [
            con.enforce(single_net, xs, ys)
            for con in conditions
        ]
    else:  # using multiple nets
        us = [
            con.enforce(net, xs, ys)
            for con, net in zip(conditions, nets)
        ]
    return us

class Condition:

    def __init__(self):
        self.ith_unit = None

    def set_impose_on(self, ith_unit):
        self.ith_unit = ith_unit

class NoCondition2D(Condition):

    def __init__(self):
        super().__init__()

    def enforce(self, net, x, y):
        return _network_output_2input(net, x, y, self.ith_unit)


class DirichletBVP2D(Condition):
    """An Dirichlet boundary value problem on a 2-D orthogonal box where :math:`x\\in[x_0, x_1]` and :math:`y\\in[y_0, y_1]`
        We are solving :math:`u(x, t)` given:
        :math:`u(x, y)\\bigg|_{x = x_0} = f_0(y)`;
        :math:`u(x, y)\\bigg|_{x = x_1} = f_1(y)`;
        :math:`u(x, y)\\bigg|_{y = y_0} = g_0(x)`;
        :math:`u(x, y)\\bigg|_{y = y_1} = g_1(x)`.

        :param x_min: The lower bound of x, the :math:`x_0`.
        :type x_min: float
        :param x_min_val: The boundary value when :math:`x = x_0`, the :math:`f_0(y)`.
        :type x_min_val: function
        :param x_max: The upper bound of x, the :math:`x_1`.
        :type x_max: float
        :param x_max_val: The boundary value when :math:`x = x_1`, the :math:`f_1(y)`.
        :type x_max_val: function
        :param y_min: The lower bound of y, the :math:`y_0`.
        :type y_min: float
        :param y_min_val: The boundary value when :math:`y = y_0`, the :math:`g_0(x)`.
        :type y_min_val: function
        :param y_max: The upper bound of y, the :math:`y_1`.
        :type y_max: float
        :param y_max_val: The boundary value when :math:`y = y_1`, the :math:`g_1(x)`.
        :type y_max_val: function
    """

    def __init__(self, x_min, x_min_val, x_max, x_max_val, y_min, y_min_val, y_max, y_max_val):
        """Initializer method
        """
        super().__init__()
        self.x_min, self.x_min_val = x_min, x_min_val
        self.x_max, self.x_max_val = x_max, x_max_val
        self.y_min, self.y_min_val = y_min, y_min_val
        self.y_max, self.y_max_val = y_max, y_max_val

    def enforce(self, net, x, y):
        r"""Enforce the output of a neural network to satisfy the boundary condition.

            :param net: The neural network that approximates the ODE.
            :type net: `torch.nn.Module`
            :param x: X-coordinates of the points where the neural network output is evaluated.
            :type x: `torch.tensor`
            :param y: Y-coordinates of the points where the neural network output is evaluated.
            :type y: `torch.tensor`
            :return: The modified output which now satisfies the boundary condition.
            :rtype: `torch.tensor`

            .. note::
                `enforce` is meant to be called by the function `solve2D`.
        """
        u = _network_output_2input(net, x, y, self.ith_unit)
        x_tilde = (x-self.x_min) / (self.x_max-self.x_min)
        y_tilde = (y-self.y_min) / (self.y_max-self.y_min)
        Axy = (1-x_tilde)*self.x_min_val(y) + x_tilde*self.x_max_val(y) + \
              (1-y_tilde)*( self.y_min_val(x) - ((1-x_tilde)*self.y_min_val(self.x_min * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_min_val(self.x_max * torch.ones_like(x_tilde))) ) + \
                 y_tilde *( self.y_max_val(x) - ((1-x_tilde)*self.y_max_val(self.x_min * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_max_val(self.x_max * torch.ones_like(x_tilde))) )
        return Axy + x_tilde*(1-x_tilde)*y_tilde*(1-y_tilde)*u


class IBVP1D(Condition):
    """An initial boundary value problem on a 1-D range where :math:`x\\in[x_0, x_1]` and time starts at :math:`t_0`
            We are solving :math:`u(x, t)` given:
            :math:`u(x, t)\\bigg|_{t = t_0} = u_0(x)`;
            :math:`u(x, t)\\bigg|_{x = x_0} = g(t)` or :math:`\\displaystyle\\frac{\\partial u(x, t)}{\\partial x}\\bigg|_{x = x_0} = g(t)`;
            :math:`u(x, t)\\bigg|_{x = x_1} = h(t)` or :math:`\\displaystyle\\frac{\\partial u(x, t)}{\\partial x}\\bigg|_{x = x_1} = h(t)`.

            :param x_min: The lower bound of x, the :math:`x_0`.
            :type x_min: float
            :param x_max: The upper bound of x, the :math:`x_1`.
            :type x_max: float
            :param t_min: The initial time, the :math:`t_0`.
            :type t_min: float
            :param t_min_val: The initial condition, the :math:`u_0(x)`.
            :type t_min_val: function
            :param x_min_val: The Dirichlet boundary condition when :math:`x = x_0`, the :math:`u(x, t)\\bigg|_{x = x_0}`, defaults to None.
            :type x_min_val: function, optional
            :param x_min_prime: The Neumann boundary condition when :math:`x = x_0`, the :math:`\\displaystyle\\frac{\\partial u(x, t)}{\\partial x}\\bigg|_{x = x_0}`, defaults to None.
            :type x_min_prime: function, optional
            :param x_max_val: The Dirichlet boundary condition when :math:`x = x_1`, the :math:`u(x, t)\\bigg|_{x = x_1}`, defaults to None.
            :type x_max_val: function, optioonal
            :param x_max_prime: The Neumann boundary condition when :math:`x = x_1`, the :math:`\\displaystyle\\frac{\\partial u(x, t)}{\\partial x}\\bigg|_{x = x_1}`, defaults to None.
            :type x_max_prime: function, optional
            :raises NotImplementedError: When unimplemented boundary conditions are configured.
        """

    def __init__(
            self, x_min, x_max, t_min, t_min_val,
            x_min_val=None, x_min_prime=None,
            x_max_val=None, x_max_prime=None,

    ):
        r"""Initializer method

        .. note::
            A instance method `enforce` is dynamically created to enforce initial and boundary conditions. It will be called by the function `solve2D`.
        """
        super().__init__()
        self.x_min, self.x_min_val, self.x_min_prime = x_min, x_min_val, x_min_prime
        self.x_max, self.x_max_val, self.x_max_prime = x_max, x_max_val, x_max_prime
        self.t_min, self.t_min_val = t_min, t_min_val
        n_conditions = sum(c is None for c in [x_min_val, x_min_prime, x_max_val, x_max_prime])
        if n_conditions != 2:
            raise NotImplementedError('Sorry, this boundary condition is not implemented.')
        if self.x_min_val and self.x_max_val:
            self.enforce = self._enforce_dd
        elif self.x_min_val and self.x_max_prime:
            self.enforce = self._enforce_dn
        elif self.x_min_prime and self.x_max_val:
            self.enforce = self._enforce_nd
        elif self.x_min_prime and self.x_max_prime:
            self.enforce = self._enforce_nn
        else:
            raise NotImplementedError('Sorry, this boundary condition is not implemented.')

    def _enforce_dd(self, net, x, t):
        uxt = _network_output_2input(net, x, t, self.ith_unit)

        t_ones = torch.ones_like(t, requires_grad=True)
        t_ones_min = self.t_min * t_ones

        x_tilde = (x - self.x_min) / (self.x_max - self.x_min)
        t_tilde = t - self.t_min

        Axt = self.t_min_val(x) + \
            x_tilde     * (self.x_max_val(t) - self.x_max_val(t_ones_min)) + \
            (1-x_tilde) * (self.x_min_val(t) - self.x_min_val(t_ones_min))
        return Axt + x_tilde * (1 - x_tilde) * (1 - torch.exp(-t_tilde)) * uxt

    def _enforce_dn(self, net, x, t):
        uxt = _network_output_2input(net, x, t, self.ith_unit)

        x_ones = torch.ones_like(x, requires_grad=True)
        t_ones = torch.ones_like(t, requires_grad=True)
        x_ones_max = self.x_max * x_ones
        t_ones_min = self.t_min * t_ones
        uxmaxt = _network_output_2input(net, x_ones_max, t, self.ith_unit)

        x_tilde = (x-self.x_min) / (self.x_max-self.x_min)
        t_tilde = t-self.t_min

        Axt = (self.x_min_val(t) - self.x_min_val(t_ones_min)) + self.t_min_val(x) + \
            x_tilde * (self.x_max-self.x_min) * (self.x_max_prime(t) - self.x_max_prime(t_ones_min))
        return Axt + x_tilde*(1-torch.exp(-t_tilde))*(
            uxt - (self.x_max-self.x_min)*diff(uxmaxt, x_ones_max) - uxmaxt
        )

    def _enforce_nd(self, net, x, t):
        uxt = _network_output_2input(net, x, t, self.ith_unit)

        x_ones = torch.ones_like(x, requires_grad=True)
        t_ones = torch.ones_like(t, requires_grad=True)
        x_ones_min = self.x_min * x_ones
        t_ones_min = self.t_min * t_ones
        uxmint = _network_output_2input(net, x_ones_min, t, self.ith_unit)

        x_tilde = (x - self.x_min) / (self.x_max - self.x_min)
        t_tilde = t - self.t_min

        Axt = (self.x_max_val(t) - self.x_max_val(t_ones_min)) + self.t_min_val(x) + \
              (x_tilde - 1) * (self.x_max - self.x_min) * (self.x_min_prime(t) - self.x_min_prime(t_ones_min))
        return Axt + (1 - x_tilde) * (1 - torch.exp(-t_tilde)) * (
                uxt + (self.x_max - self.x_min) * diff(uxmint, x_ones_min) - uxmint
        )

    def _enforce_nn(self, net, x, t):
        uxt = _network_output_2input(net, x, t, self.ith_unit)

        x_ones = torch.ones_like(x, requires_grad=True)
        t_ones = torch.ones_like(t, requires_grad=True)
        x_ones_min = self.x_min * x_ones
        x_ones_max = self.x_max * x_ones
        t_ones_min = self.t_min * t_ones
        uxmint = _network_output_2input(net, x_ones_min, t, self.ith_unit)
        uxmaxt = _network_output_2input(net, x_ones_max, t, self.ith_unit)

        x_tilde = (x - self.x_min) / (self.x_max - self.x_min)
        t_tilde = t - self.t_min

        Axt = self.t_min_val(x) - 0.5 * (1 - x_tilde) ** 2 * (self.x_max - self.x_min) * (
                self.x_min_prime(t) - self.x_min_prime(t_ones_min)
        ) + 0.5 * x_tilde ** 2 * (self.x_max - self.x_min) * (
                      self.x_max_prime(t) - self.x_max_prime(t_ones_min)
              )
        return Axt + (1 - torch.exp(-t_tilde)) * (
                uxt - x_tilde * (self.x_max - self.x_min) * diff(uxmint, x_ones_min) \
                + 0.5 * x_tilde ** 2 * (self.x_max - self.x_min) * (
                        diff(uxmint, x_ones_min) - diff(uxmaxt, x_ones_max)
                ))


class ExampleGenerator2D:
    """An example generator for generating 2-D training points.

        :param grid: The discretization of the 2 dimensions, if we want to generate points on a :math:`m \\times n` grid, then `grid` is `(m, n)`, defaults to `(10, 10)`.
        :type grid: tuple[int, int], optional
        :param xy_min: The lower bound of 2 dimensions, if we only care about :math:`x \\geq x_0` and :math:`y \\geq y_0`, then `xy_min` is `(x_0, y_0)`, defaults to `(0.0, 0.0)`.
        :type xy_min: tuple[float, float], optional
        :param xy_max: The upper boound of 2 dimensions, if we only care about :math:`x \\leq x_1` and :math:`y \\leq y_1`, then `xy_min` is `(x_1, y_1)`, defaults to `(1.0, 1.0)`.
        :type xy_max: tuple[float, float], optional
        :param method: The distribution of the 2-D points generated.
            If set to 'equally-spaced', the points will be fixed to the grid specified.
            If set to 'equally-spaced-noisy', a normal noise will be added to the previously mentioned set of points, defaults to 'equally-spaced-noisy'.
        :type method: str, optional
        :param xy_noise_std: the standard deviation of the noise on the x and y dimension, if not specified, the default value will be (grid step size on x dimension / 4, grid step size on y dimension / 4)
        :type xy_noise_std: tuple[int, int], optional, defaults to None
        :raises ValueError: When provided with an unknown method.
    """

    def __init__(self, grid=(10, 10), xy_min=(0.0, 0.0), xy_max=(1.0, 1.0), method='equally-spaced-noisy', xy_noise_std=None):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 2-D training points. It will be called by the function `solve2D`.
        """
        self.size = grid[0] * grid[1]

        if method == 'equally-spaced':
            x = torch.linspace(xy_min[0], xy_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xy_min[1], xy_max[1], grid[1], requires_grad=True)
            grid_x, grid_y = torch.meshgrid(x, y)
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()

            self.get_examples = lambda: (self.grid_x, self.grid_y)

        elif method == 'equally-spaced-noisy':
            x = torch.linspace(xy_min[0], xy_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xy_min[1], xy_max[1], grid[1], requires_grad=True)
            grid_x, grid_y = torch.meshgrid(x, y)
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()

            if xy_noise_std:
                self.noise_xstd, self.noise_ystd = xy_noise_std
            else:
                self.noise_xstd = ((xy_max[0] - xy_min[0]) / grid[0]) / 4.0
                self.noise_ystd = ((xy_max[1] - xy_min[1]) / grid[1]) / 4.0
            self.get_examples = lambda: (
                torch.normal(mean=self.grid_x, std=self.noise_xstd),
                torch.normal(mean=self.grid_y, std=self.noise_ystd)
            )
        else:
            raise ValueError(f'Unknown method: {method}')


class Monitor2D:
    """A monitor for checking the status of the neural network during training.

    :param xy_min: The lower bound of 2 dimensions, if we only care about :math:`x \\geq x_0` and :math:`y \\geq y_0`, then `xy_min` is `(x_0, y_0)`.
    :type xy_min: tuple[float, float], optional
    :param xy_max: The upper boound of 2 dimensions, if we only care about :math:`x \\leq x_1` and :math:`y \\leq y_1`, then `xy_min` is `(x_1, y_1)`.
    :type xy_max: tuple[float, float], optional
    :param check_every: The frequency of checking the neural network represented by the number of epochs between two checks, defaults to 100.
    :type check_every: int, optional
    """

    def __init__(self, xy_min, xy_max, check_every=100):
        """Initializer method
        """
        self.using_non_gui_backend = matplotlib.get_backend() is 'agg'
        self.check_every = check_every
        self.fig = None
        self.axs = []  # subplots
        self.cbs = []  # color bars
        # input for neural network
        gen = ExampleGenerator2D([32, 32], xy_min, xy_max, method='equally-spaced')
        xs_ann, ys_ann = gen.get_examples()
        self.xs_ann, self.ys_ann = xs_ann.reshape(-1, 1), ys_ann.reshape(-1, 1)


    def check(self, single_net, nets, conditions, history):
        r"""Draw 2 plots: One shows the shape of the current solution (with heat map). The other shows the history training loss and validation loss.

        :param single_net: The neural network that approximates the PDE.
        :type single_net: `torch.nn.Module`
        :param nets: The neural networks that approximates the PDE.
        :type nets: list [`torch.nn.Module`]
        :param conditions: The initial/boundary condition of the PDE.
        :type conditions: list [`neurodiffeq.pde.DirichletBVP2D` or `neurodiffeq.pde.IBVP1D` or `neurodiffeq.pde.NoCondition`]
        :param history: The history of training loss and validation loss. The 'train' entry is a list of training loss and 'valid' entry is a list of validation loss.
        :type history: dict['train': list[float], 'valid': list[float]]

        .. note::
            `check` is meant to be called by the function `solve2D`.
        """

        if not self.fig:
            # initialize the figure and axes here so that the Monitor knows the number of dependent variables and
            # size of the figure, number of the subplots, etc.
            n_axs = len(conditions)+2  # one for each dependent variable, plus one for training and validation loss, plus one for metrics
            n_row, n_col = (n_axs+1) // 2, 2
            self.fig = plt.figure(figsize=(20, 8*n_row))
            for i in range(n_axs):
                self.axs.append( self.fig.add_subplot(n_row, n_col, i+1) )
            for i in range(n_axs-1):
                self.cbs.append(None)

        us = _trial_solution_2input(single_net, nets, self.xs_ann, self.ys_ann, conditions)

        for i, ax_u in enumerate( zip(self.axs[:-1], us) ):
            ax, u = ax_u
            ax.clear()
            u_as_mat = u.detach().cpu().numpy().reshape((32, 32))
            cax = ax.matshow(u_as_mat, cmap='hot', interpolation='nearest')
            if self.cbs[i]:
                self.cbs[i].remove()
            self.cbs[i] = self.fig.colorbar(cax, ax=ax)
            ax.set_title(f'u[{i}](x, y)')

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
        self.axs[-1].legend()

        self.fig.canvas.draw()
        if not self.using_non_gui_backend:
            plt.pause(0.05)


def solve2D(
        pde, condition, xy_min=None, xy_max=None,
        net=None, train_generator=None, shuffle=True, valid_generator=None, optimizer=None, criterion=None, additional_loss_term=None, metrics=None,
        batch_size=16,
        max_epochs=1000,
        monitor=None, return_internal=False, return_best=False
):
    """Train a neural network to solve a PDE with 2 independent variables.

    :param pde: The PDE to solve. If the PDE is :math:`F(u, x, y) = 0` where :math:`u` is the dependent variable and :math:`x` and :math:`y` are the independent variables,
        then `pde` should be a function that maps :math:`(u, x, y)` to :math:`F(u, x, y)`.
    :type pde: function
    :param condition: The initial/boundary condition.
    :type condition: `neurodiffeq.pde.DirichletBVP2D` or `neurodiffeq.pde.IBVP1D` or `neurodiffeq.pde.NoCondition`
    :param xy_min: The lower bound of 2 dimensions, if we only care about :math:`x \\geq x_0` and :math:`y \\geq y_0`, then `xy_min` is `(x_0, y_0)`, only needed when train_generator and valid_generator are not specified, defaults to None
    :type xy_min: tuple[float, float], optional
    :param xy_max: The upper bound of 2 dimensions, if we only care about :math:`x \\leq x_1` and :math:`y \\leq y_1`, then `xy_min` is `(x_1, y_1)`, only needed when train_generator and valid_generator are not specified, defaults to None
    :type xy_max: tuple[float, float], optional
    :param net: The neural network used to approximate the solution, defaults to None.
    :type net: `torch.nn.Module`, optional
    :param train_generator: The example generator to generate 1-D training points, default to None.
    :type train_generator: `neurodiffeq.pde.ExampleGenerator2D`, optional
    :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
    :type shuffle: bool, optional
    :param valid_generator: The example generator to generate 1-D validation points, default to None.
    :type valid_generator: `neurodiffeq.pde.ExampleGenerator2D`, optional
    :param optimizer: The optimization method to use for training, defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion: The loss function to use for training, defaults to None.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param additional_loss_term: Extra terms to add to the loss function besides the part specified by `criterion`. The input of `additional_loss_term` should be the same as `pde_system`
    :type additional_loss_term: function
    :param metrics: Metrics to keep track of during training. The metrics should be passed as a dictionary where the keys are the names of the metrics, and the values are the corresponding function.
        The input functions should be the same as `pde` and the output should be a numeric value. The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, function]
    :param batch_size: The size of the mini-batch to use, defaults to 16.
    :type batch_size: int, optional
    :param max_epochs: The maximum number of epochs to train, defaults to 1000.
    :type max_epochs: int, optional
    :param monitor: The monitor to check the status of nerual network during training, defaults to None.
    :type monitor: `neurodiffeq.pde.Monitor2D`, optional
    :param return_internal: Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function, defaults to False.
    :type return_internal: bool, optional
    :param return_best: Whether to return the nets that achieved the lowest validation loss, defaults to False.
    :type return_best: bool, optional
    :return: The solution of the PDE. The history of training loss and validation loss.
        Optionally, the nets, conditions, training generator, validation generator, optimizer and loss function.
        The solution is a function that has the signature `solution(xs, ys, as_type)`.
    :rtype: tuple[`neurodiffeq.pde.Solution`, dict]; or tuple[`neurodiffeq.pde.Solution`, dict, dict]
    """
    nets = None if not net else [net]
    return solve2D_system(
        pde_system=lambda u, x, y: [pde(u, x, y)], conditions=[condition],
        xy_min=xy_min, xy_max=xy_max, nets=nets,
        train_generator=train_generator, shuffle=shuffle, valid_generator=valid_generator,
        optimizer=optimizer, criterion=criterion, additional_loss_term=additional_loss_term, metrics=metrics, batch_size=batch_size,
        max_epochs=max_epochs, monitor=monitor, return_internal=return_internal, return_best=return_best
    )


def solve2D_system(
        pde_system, conditions, xy_min=None, xy_max=None,
        single_net=None, nets=None, train_generator=None, shuffle=True, valid_generator=None,
        optimizer=None, criterion=None, additional_loss_term=None, metrics=None, batch_size=16,
        max_epochs=1000,
        monitor=None, return_internal=False, return_best=False
):
    """Train a neural network to solve a PDE with 2 independent variables.

        :param pde_system: The PDEsystem to solve. If the PDE is :math:`F_i(u_1, u_2, ..., u_n, x, y) = 0` where :math:`u_i` is the i-th dependent variable and :math:`x` and :math:`y` are the independent variables,
            then `pde_system` should be a function that maps :math:`(u_1, u_2, ..., u_n, x, y)` to a list where the i-th entry is :math:`F_i(u_1, u_2, ..., u_n, x, y)`.
        :type pde_system: function
        :param conditions: The initial/boundary conditions. The ith entry of the conditions is the condition that :math:`x_i` should satisfy.
        :type conditions: list[`neurodiffeq.pde.DirichletBVP2D` or `neurodiffeq.pde.IBVP1D` or `neurodiffeq.pde.NoCondition`]
        :param xy_min: The lower bound of 2 dimensions, if we only care about :math:`x \\geq x_0` and :math:`y \\geq y_0`, then `xy_min` is `(x_0, y_0)`, only needed when train_generator or valid_generator are not specified, defaults to None
        :type xy_min: tuple[float, float], optional
        :param xy_max: The upper bound of 2 dimensions, if we only care about :math:`x \\leq x_1` and :math:`y \\leq y_1`, then `xy_min` is `(x_1, y_1)`, only needed when train_generator or valid_generator are not specified, defaults to None
        :type xy_max: tuple[float, float], optional
        :param single_net: The single neural network used to approximate the solution. Only one of `single_net` and `nets` should be specified, defaults to None
        :param single_net: `torch.nn.Module`, optional
        :param nets: The neural networks used to approximate the solution, defaults to None.
        :type nets: list[`torch.nn.Module`], optional
        :param train_generator: The example generator to generate 1-D training points, default to None.
        :type train_generator: `neurodiffeq.pde.ExampleGenerator2D`, optional
        :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
        :type shuffle: bool, optional
        :param valid_generator: The example generator to generate 1-D validation points, default to None.
        :type valid_generator: `neurodiffeq.pde.ExampleGenerator2D`, optional
        :param optimizer: The optimization method to use for training, defaults to None.
        :type optimizer: `torch.optim.Optimizer`, optional
        :param criterion: The loss function to use for training, defaults to None.
        :type criterion: `torch.nn.modules.loss._Loss`, optional
        :param additional_loss_term: Extra terms to add to the loss function besides the part specified by `criterion`. The input of `additional_loss_term` should be the same as `pde_system`
        :type additional_loss_term: function
        :param metrics: Metrics to keep track of during training. The metrics should be passed as a dictionary where the keys are the names of the metrics, and the values are the corresponding function.
            The input functions should be the same as `pde_system` and the output should be a numeric value. The metrics are evaluated on both the training set and validation set.
        :type metrics: dict[string, function]
        :param batch_size: The size of the mini-batch to use, defaults to 16.
        :type batch_size: int, optional
        :param max_epochs: The maximum number of epochs to train, defaults to 1000.
        :type max_epochs: int, optional
        :param monitor: The monitor to check the status of nerual network during training, defaults to None.
        :type monitor: `neurodiffeq.pde.Monitor2D`, optional
        :param return_internal: Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function, defaults to False.
        :type return_internal: bool, optional
        :param return_best: Whether to return the nets that achieved the lowest validation loss, defaults to False.
        :type return_best: bool, optional
        :return: The solution of the PDE. The history of training loss and validation loss.
            Optionally, the nets, conditions, training generator, validation generator, optimizer and loss function.
            The solution is a function that has the signature `solution(xs, ys, as_type)`.
        :rtype: tuple[`neurodiffeq.pde.Solution`, dict]; or tuple[`neurodiffeq.pde.Solution`, dict, dict]
        """

    ########################################### subroutines ###########################################
    def train(train_generator, net, nets, pde_system, conditions, criterion, additional_loss_term, metrics, shuffle, optimizer):
        train_examples_x, train_examples_y = train_generator.get_examples()
        train_examples_x, train_examples_y = train_examples_x.reshape((-1, 1)), train_examples_y.reshape((-1, 1))
        n_examples_train = train_generator.size
        idx = np.random.permutation(n_examples_train) if shuffle else np.arange(n_examples_train)

        batch_start, batch_end = 0, batch_size
        while batch_start < n_examples_train:
            if batch_end > n_examples_train:
                batch_end = n_examples_train
            batch_idx = idx[batch_start:batch_end]
            xs, ys = train_examples_x[batch_idx], train_examples_y[batch_idx]

            train_loss_batch = calculate_loss(xs, ys, net, nets, pde_system, conditions, criterion, additional_loss_term)

            optimizer.zero_grad()
            train_loss_batch.backward()
            optimizer.step()

            batch_start += batch_size
            batch_end += batch_size

        train_loss_epoch = calculate_loss(train_examples_x, train_examples_y, net, nets, pde_system, conditions, criterion, additional_loss_term)

        train_metrics_epoch = calculate_metrics(train_examples_x, train_examples_y, net, nets, conditions, metrics)
        return train_loss_epoch, train_metrics_epoch

    def valid(valid_generator, net, nets, pde_system, conditions, criterion, additional_loss_term, metrics):
        valid_examples_x, valid_examples_y = valid_generator.get_examples()
        valid_examples_x, valid_examples_y = valid_examples_x.reshape((-1, 1)), valid_examples_y.reshape((-1, 1))
        valid_loss_epoch = calculate_loss(valid_examples_x, valid_examples_y, net, nets, pde_system, conditions, criterion, additional_loss_term)
        valid_loss_epoch = valid_loss_epoch.item()

        valid_metrics_epoch = calculate_metrics(valid_examples_x, valid_examples_y, net, nets, conditions, metrics)
        return valid_loss_epoch, valid_metrics_epoch

    def calculate_loss(xs, ys, net, nets, pde_system, conditions, criterion, additional_loss_term):
        us = _trial_solution_2input(net, nets, xs, ys, conditions)
        Fuxys = pde_system(*us, xs, ys)
        loss = sum(
            criterion(Fuxy, torch.zeros_like(xs))
            for Fuxy in Fuxys
        )
        if additional_loss_term is not None:
            loss += additional_loss_term(*us, xs, ys)
        return loss

    def calculate_metrics(xs, ys, net, nets, conditions, metrics):
        us = _trial_solution_2input(net, nets, xs, ys, conditions)
        metrics_ = {
            metric_name: metric_function(*us, xs, ys).item()
            for metric_name, metric_function in metrics.items()
        }
        return metrics_
    ###################################################################################################

    if single_net and nets:
        raise RuntimeError('Only one of net and nets should be specified')
    # defaults to use a single neural network
    if (not single_net) and (not nets):
        net = FCNN(n_input_units=2, n_output_units=len(conditions), n_hidden_units=32, n_hidden_layers=1, actv=nn.Tanh)
    if single_net:
        # mark the Conditions so that we know which condition correspond to which output unit
        for ith, con in enumerate(conditions):
            con.set_impose_on(ith)
    if not train_generator:
        if (xy_min is None) or (xy_max is None):
            raise RuntimeError('Please specify xy_min and xy_max when train_generator is not specified')
        train_generator = ExampleGenerator2D((32, 32), xy_min, xy_max, method='equally-spaced-noisy')
    if not valid_generator:
        if (xy_min is None) or (xy_max is None):
            raise RuntimeError('Please specify xy_min and xy_max when valid_generator is not specified')
        valid_generator = ExampleGenerator2D((32, 32), xy_min, xy_max, method='equally-spaced')
    if (not optimizer) and single_net:  # using a single net
        optimizer = optim.Adam(single_net.parameters(), lr=0.001)
    if (not optimizer) and nets:  # using multiple nets
        all_parameters = []
        for net in nets:
            all_parameters += list(net.parameters())
        optimizer = optim.Adam(all_parameters, lr=0.001)
    if not criterion:
        criterion = nn.MSELoss()

    history = {}
    history['train_loss'] = []
    history['valid_loss'] = []
    for metric_name, _ in metrics.items():
        history['train__'+metric_name] = []
        history['valid__'+metric_name] = []

    if return_best:
        valid_loss_epoch_min = np.inf
        solution_min = None

    for epoch in range(max_epochs):
        train_loss_epoch, train_metrics_epoch = train(train_generator, single_net, nets, pde_system, conditions, criterion, additional_loss_term, metrics, shuffle, optimizer)
        history['train_loss'].append(train_loss_epoch)
        for metric_name, metric_value in train_metrics_epoch.items():
            history['train__'+metric_name].append(metric_value)

        valid_loss_epoch, valid_metrics_epoch = valid(valid_generator, single_net, nets, pde_system, conditions, criterion, additional_loss_term, metrics)
        history['valid_loss'].append(valid_loss_epoch)
        for metric_name, metric_value in valid_metrics_epoch.items():
            history['valid__'+metric_name].append(metric_value)

        if monitor and epoch % monitor.check_every == 0:
            monitor.check(single_net, nets, conditions, history)

        if return_best and valid_loss_epoch < valid_loss_epoch_min:
            valid_loss_epoch_min = valid_loss_epoch
            solution_min = Solution(single_net, nets, conditions)

    if return_best:
        solution = solution_min
    else:
        solution = Solution(single_net, nets, conditions)

    if return_internal:
        internal = {
            'single_net': single_net,
            'nets': nets,
            'conditions': conditions,
            'train_generator': train_generator,
            'valid_generator': valid_generator,
            'optimizer': optimizer,
            'criterion': criterion
        }
        return solution, history, internal
    else:
        return solution, history


class Solution:
    """A solution to an PDE (system)

    :param single_net: The neural networks that approximates the PDE.
    :type single_net: `torch.nn.Module`
    :param nets: The neural networks that approximates the PDE.
    :type nets: list[`torch.nn.Module`]
    :param conditions: The initial/boundary conditions of the ODE (system).
    :type conditions: list[`neurodiffeq.ode.IVP` or `neurodiffeq.ode.DirichletBVP` or `neurodiffeq.pde.NoCondition`]
    """
    def __init__(self, single_net, nets, conditions):
        """Initializer method
        """
        self.single_net = deepcopy(single_net)
        self.nets = deepcopy(nets)
        self.conditions = deepcopy(conditions)

    def __call__(self, xs, ys, as_type='tf'):
        """Evaluate the solution at certain points.

        :param xs: the x-coordinates of points on which the dependent variables are evaluated.
        :type xs: `torch.tensor` or sequence of number
        :param ys: the y-coordinates of points on which the dependent variables are evaluated.
        :type ys: `torch.tensor` or sequence of number
        :param as_type: Whether the returned value is a `torch.tensor` ('tf') or `numpy.array` ('np').
        :type as_type: str
        :return: dependent variables are evaluated at given points.
        :rtype: list[`torch.tensor` or `numpy.array` (when there is more than one dependent variables)
            `torch.tensor` or `numpy.array` (when there is only one dependent variable).
        """
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs, dtype=torch.float32)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys, dtype=torch.float32)
        original_shape = xs.shape
        xs, ys = xs.reshape(-1, 1), ys.reshape(-1, 1)
        if as_type not in ('tf', 'np'):
            raise ValueError("The valid return types are 'tf' and 'np'.")

        us = _trial_solution_2input(self.single_net, self.nets, xs, ys, self.conditions)
        us = [u.reshape(original_shape) for u in us]
        if as_type == 'np':
            us = [u.detach().cpu().numpy() for u in us]

        return us if len(us) > 1 else us[0]


def make_animation(solution, xs, ts):
    """Create animation of 1-D time-dependent problems.

    :param solution: solution function returned by `solve2D` (for a 1-D time-dependent problem).
    :type solution: function
    :param xs: The locations to evaluate solution.
    :type xs: `numpy.array`
    :param ts: The time points to evaluate solution.
    :type ts: `numpy.array`
    :return: The animation.
    :rtype: `matplotlib.animation.FuncAnimation`
    """

    xx, tt = np.meshgrid(xs, ts)
    sol_net = solution(xx, tt, as_type='np')

    def u_gen():
        for i in range( len(sol_net) ):
            yield sol_net[i]

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    umin, umax = sol_net.min(), sol_net.max()
    scale = umax - umin
    ax.set_ylim(umin-scale*0.1, umax+scale*0.1)
    ax.set_xlim(xs.min(), xs.max())
    def run(data):
        line.set_data(xs, data)
        return line,

    return animation.FuncAnimation(
        fig, run, u_gen, blit=True, interval=50, repeat=False
    )
