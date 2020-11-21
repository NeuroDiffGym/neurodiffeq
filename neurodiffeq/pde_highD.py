import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as tri

from .networks import FCNN
from copy import deepcopy

# make a function to set global state
FLOAT_DTYPE=torch.float32

def set_default_dtype(dtype):
    """Set the default `dtype` of `torch.tensor` used in `neurodiffeq`.

    :param dtype: `torch.float`, `torch.double`, etc
    """
    global FLOAT_DTYPE
    FLOAT_DTYPE=dtype
    torch.set_default_dtype(FLOAT_DTYPE)

# Calculate the output of a neural network with 3 input.
# In the case where the neural network has multiple output unit,
# `ith_unit` specifies which unit do we want.
def _network_output_2input(net, xs, ys, ts, ith_unit):
    xyts = torch.cat((xs, ys, ts), 1)
    nn_output = net(xyts)
    if ith_unit is not None:
        return nn_output[:, ith_unit].reshape(-1, 1)
    else:
        return nn_output

# Adjust the output of the neural network with trial solutions
# coded into `conditions`.
def _trial_solution_2input(single_net, nets, xs, ys, ts, conditions):
    if single_net:  # using a single net
        us = [
            con.enforce(single_net, xs, ys, ts)
            for con in conditions
        ]
    else:  # using multiple nets
        us = [
            con.enforce(net, xs, ys, ts)
            for con, net in zip(conditions, nets)
        ]
    return us


# The base class for conditions. Conditions carry the knowledge of
# the trial solutions. They how to adjust the neural network so that
# the transformed output satisfies the initial/boundary conditions.
class Condition:

    def __init__(self):
        self.ith_unit = None

    def set_impose_on(self, ith_unit):
        self.ith_unit = ith_unit

    def enforce(self, net, *dimensions):
        raise NotImplementedError

    def in_domain(self, *dimensions):
        return np.ones_like(dimensions[0], dtype=bool)


class NoCondition2D(Condition):
    """A stand-in `Condition` that does not impose any initial/boundary conditions.
    """

    def __init__(self):
        super().__init__()

    def enforce(self, net, x, y, t):
        return _network_output_2input(net, x, y, t, self.ith_unit)

class IBVP2D(Condition):
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
            self, x_min, x_max, y_min, y_max, t_min, t_min_val,
            x_min_val=None, x_min_prime=None,
            x_max_val=None, x_max_prime=None,
            y_min_val=None, y_min_prime=None,
            y_max_val=None, y_max_prime=None,
    ):
        r"""Initializer method

        .. note::
            A instance method `enforce` is dynamically created to enforce initial and boundary conditions. It will be called by the function `solve2D`.
        """
        super().__init__()
        self.x_min, self.x_min_val, self.x_min_prime = x_min, x_min_val, x_min_prime
        self.x_max, self.x_max_val, self.x_max_prime = x_max, x_max_val, x_max_prime
        self.y_min, self.y_min_val, self.y_min_prime = y_min, y_min_val, y_min_prime
        self.y_max, self.y_max_val, self.y_max_prime = y_max, y_max_val, y_max_prime
        self.t_min, self.t_min_val = t_min, t_min_val
        n_conditions = sum(c is None for c in [x_min_val, x_min_prime, x_max_val, x_max_prime,
                                               y_min_val, y_min_prime, y_max_val, y_max_prime])
        if n_conditions != 4:
            raise NotImplementedError('Sorry, this boundary condition is not implemented.')

        # only Dirichlet everywhere at this point
        if self.x_min_val and self.x_max_val and self.y_min_val and self.y_max_val:
            self.enforce = self._enforce_dd
        else:
            raise NotImplementedError('Sorry, this boundary condition is not implemented.')

    # When we have Dirichlet boundary conditions on both ends of the domain:
    def _enforce_dd(self, net, x, y, t):
        uxyt = _network_output_2input(net, x, y, t, self.ith_unit)

        x_tilde = (x-self.x_min) / (self.x_max-self.x_min)
        y_tilde = (y-self.y_min) / (self.y_max-self.y_min)
        t_tilde = t - self.t_min

        t_ones = torch.ones_like(t, requires_grad=True)
        t_ones_min = self.t_min * t_ones

        Axyt = self.t_min_val(x, y)
              #  +\
              #  (1-x_tilde)*(self.x_min_val(y) - self.x_min_val(t_ones_min)) +\
              #  x_tilde*(self.x_max_val(y) - self.x_max_val(t_ones_min)) + \
              # (1-y_tilde)*( self.y_min_val(x) - self.y_min_val(t_ones_min) - ((1-x_tilde)*self.y_min_val(self.x_min * torch.ones_like(x_tilde))
              #                                     + x_tilde *self.y_min_val(self.x_max * torch.ones_like(x_tilde))) ) + \
              #    y_tilde *( self.y_max_val(x) - self.y_max_val(t_ones_min) - ((1-x_tilde)*self.y_max_val(self.x_min * torch.ones_like(x_tilde))
              #                                     + x_tilde *self.y_max_val(self.x_max * torch.ones_like(x_tilde))) )

        return Axyt + x_tilde*(1-x_tilde)*y_tilde*(1-y_tilde)*(1 - torch.exp(-t_tilde))*uxyt

class ExampleGenerator3D:
    """An example generator for generating 3-D training points.

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

    def __init__(self, grid=(10, 10, 10), xyt_min=(0.0, 0.0, 0.0), xyt_max=(1.0, 1.0, 1.0), method='equally-spaced-noisy',
                 xyt_noise_std=None):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 2-D training points. It will be called by the function `solve2D`.
        """
        self.size = grid[0] * grid[1] * grid[2]

        if method == 'equally-spaced':
            x = torch.linspace(xyt_min[0], xyt_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xyt_min[1], xyt_max[1], grid[1], requires_grad=True)
            t = torch.linspace(xyt_min[2], xyt_max[2], grid[2], requires_grad=True)
            grid_x, grid_y, grid_t = torch.meshgrid(x, y, t)
            self.grid_x, self.grid_y, self.grid_t = grid_x.flatten(), grid_y.flatten(), grid_t.flatten()

            self.get_examples = lambda: (self.grid_x, self.grid_y, self.grid_t)

        elif method == 'equally-spaced-noisy':
            x = torch.linspace(xyt_min[0], xyt_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xyt_min[1], xyt_max[1], grid[1], requires_grad=True)
            t = torch.linspace(xyt_min[2], xyt_max[2], grid[2], requires_grad=True)
            grid_x, grid_y, grid_t = torch.meshgrid(x, y, t)
            self.grid_x, self.grid_y, self.grid_t = grid_x.flatten(), grid_y.flatten(), grid_t.flatten()

            if xyt_noise_std:
                self.noise_xstd, self.noise_ystd, self.noise_tstd = xyt_noise_std
            else:
                self.noise_xstd = ((xyt_max[0] - xyt_min[0]) / grid[0]) / 4.0
                self.noise_ystd = ((xyt_max[1] - xyt_min[1]) / grid[1]) / 4.0
                self.noise_tstd = ((xyt_max[2] - xyt_min[2]) / grid[2]) / 4.0
            self.get_examples = lambda: (
                torch.normal(mean=self.grid_x, std=self.noise_xstd),
                torch.normal(mean=self.grid_y, std=self.noise_ystd),
                torch.normal(mean=self.grid_t, std=self.noise_tstd)
            )
        else:
            raise ValueError(f'Unknown method: {method}')


class PredefinedExampleGenerator3D:
    """An example generator for generating 2-D training points. Here the training
        points are fixed and predefined.

        :param xs: The x-dimension of the training points
        :type xs: `torch.tensor`
        :param ys: The y-dimension of the training points
        :type ys: `torch.tensor`
    """

    def __init__(self, xs, ys, ts):
        self.size = len(xs)
        x = torch.tensor(xs, requires_grad=True, dtype=FLOAT_DTYPE)
        y = torch.tensor(ys, requires_grad=True, dtype=FLOAT_DTYPE)
        t = torch.tensor(ts, requires_grad=True, dtype=FLOAT_DTYPE)
        self.x, self.y, self.t = x.flatten(), y.flatten(), t.flatten()

    def get_examples(self):
        """Returns the training points
            points are fixed and predifined.

            :returns: The x and y dimension of the training points
            :rtype: tuple[`torch.tensor`, `torch.tensor`]
        """
        return self.x, self.y, self.t

def solve3D(
        pde, condition, xyt_min=None, xyt_max=None,
        net=None, train_generator=None, shuffle=True, valid_generator=None, optimizer=None, criterion=None,
        additional_loss_term=None, metrics=None,
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
    return solve3D_system(
        pde_system=lambda u, x, y, t: [pde(u, x, y, t)], conditions=[condition],
        xyt_min=xyt_min, xyt_max=xyt_max, nets=nets,
        train_generator=train_generator, shuffle=shuffle, valid_generator=valid_generator,
        optimizer=optimizer, criterion=criterion, additional_loss_term=additional_loss_term, metrics=metrics, batch_size=batch_size,
        max_epochs=max_epochs, monitor=monitor, return_internal=return_internal, return_best=return_best
    )


def solve3D_system(
        pde_system, conditions, xyt_min=None, xyt_max=None,
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
    # Train the neural network for 1 epoch, return the training loss and training metrics
    def train(train_generator, net, nets, pde_system, conditions, criterion, additional_loss_term, metrics, shuffle, optimizer):
        train_examples_x, train_examples_y, train_examples_t  = train_generator.get_examples()
        train_examples_x, train_examples_y, train_examples_t = train_examples_x.reshape((-1, 1)), train_examples_y.reshape((-1, 1)), train_examples_t.reshape((-1, 1))
        n_examples_train = train_generator.size
        idx = np.random.permutation(n_examples_train) if shuffle else np.arange(n_examples_train)

        batch_start, batch_end = 0, batch_size
        while batch_start < n_examples_train:
            if batch_end > n_examples_train:
                batch_end = n_examples_train
            batch_idx = idx[batch_start:batch_end]
            xs, ys, ts = train_examples_x[batch_idx], train_examples_y[batch_idx], train_examples_t[batch_idx]

            train_loss_batch = calculate_loss(xs, ys, ts, net, nets, pde_system, conditions, criterion, additional_loss_term)

            optimizer.zero_grad()
            train_loss_batch.backward()
            optimizer.step()

            batch_start += batch_size
            batch_end += batch_size

        train_loss_epoch = calculate_loss(train_examples_x, train_examples_y, train_examples_t, net, nets, pde_system, conditions, criterion, additional_loss_term)
        train_loss_epoch = train_loss_epoch.item()
        print(train_loss_epoch)

        train_metrics_epoch = calculate_metrics(train_examples_x, train_examples_y, train_examples_t, net, nets, conditions, metrics)
        return train_loss_epoch, train_metrics_epoch

    # Validate the neural network, return the validation loss and validation metrics
    def valid(valid_generator, net, nets, pde_system, conditions, criterion, additional_loss_term, metrics):
        valid_examples_x, valid_examples_y, valid_examples_t = valid_generator.get_examples()
        valid_examples_x, valid_examples_y, valid_examples_t = valid_examples_x.reshape((-1, 1)), valid_examples_y.reshape((-1, 1)), valid_examples_t.reshape((-1, 1))
        valid_loss_epoch = calculate_loss(valid_examples_x, valid_examples_y, valid_examples_t, net, nets, pde_system, conditions, criterion, additional_loss_term)
        valid_loss_epoch = valid_loss_epoch.item()

        valid_metrics_epoch = calculate_metrics(valid_examples_x, valid_examples_y, valid_examples_t, net, nets, conditions, metrics)
        return valid_loss_epoch, valid_metrics_epoch

    # calculate the loss function
    def calculate_loss(xs, ys, ts, net, nets, pde_system, conditions, criterion, additional_loss_term):
        us = _trial_solution_2input(net, nets, xs, ys, ts, conditions)

        Fuxyts = pde_system(*us, xs, ys, ts)

        loss = sum(
            criterion(Fuxy, torch.zeros_like(xs))
            for Fuxy in Fuxyts
        )
        if additional_loss_term is not None:
            loss += additional_loss_term(*us, xs, ys, ts)
        return loss

    # caclulate the metrics
    def calculate_metrics(xs, ys, ts, net, nets, conditions, metrics):
        us = _trial_solution_2input(net, nets, xs, ys, ts, conditions)
        metrics_ = {
            metric_name: metric_function(*us, xs, ys, ts).item()
            for metric_name, metric_function in metrics.items()
        }
        return metrics_
    ###################################################################################################

    if single_net and nets:
        raise RuntimeError('Only one of net and nets should be specified')
    # defaults to use a single neural network
    if (not single_net) and (not nets):
        net = FCNN(n_input_units=3, n_output_units=len(conditions), n_hidden_units=32, n_hidden_layers=1, actv=nn.Tanh)
    if single_net:
        # mark the Conditions so that we know which condition correspond to which output unit
        for ith, con in enumerate(conditions):
            con.set_impose_on(ith)
    if not train_generator:
        if (xyt_min is None) or (xyt_max is None):
            raise RuntimeError('Please specify xy_min and xy_max when train_generator is not specified')
        train_generator = ExampleGenerator3D((32, 32, 32), xyt_min, xyt_max, method='equally-spaced-noisy')
    if not valid_generator:
        if (xyt_min is None) or (xyt_max is None):
            raise RuntimeError('Please specify xy_min and xy_max when valid_generator is not specified')
        valid_generator = ExampleGenerator3D((32, 32, 32), xyt_min, xyt_max, method='equally-spaced')
    if (not optimizer) and single_net:  # using a single net
        optimizer = optim.Adam(single_net.parameters(), lr=0.001)
    if (not optimizer) and nets:  # using multiple nets
        all_parameters = []
        for net in nets:
            all_parameters += list(net.parameters())
        optimizer = optim.Adam(all_parameters, lr=0.001)
    if not criterion:
        criterion = nn.MSELoss()
    if metrics is None:
        metrics = {}

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

        print(epoch)

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

    def __call__(self, xs, ys, ts, as_type='tf'):
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
            xs = torch.tensor(xs, dtype=FLOAT_DTYPE)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys, dtype=FLOAT_DTYPE)
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=FLOAT_DTYPE)
        original_shape = xs.shape
        xs, ys, ts = xs.reshape(-1, 1), ys.reshape(-1, 1), ts.reshape(-1, 1)
        if as_type not in ('tf', 'np'):
            raise ValueError("The valid return types are 'tf' and 'np'.")

        us = _trial_solution_2input(self.single_net, self.nets, xs, ys, ts, self.conditions)
        us = [u.reshape(original_shape) for u in us]
        if as_type == 'np':
            us = [u.detach().cpu().numpy() for u in us]

        return us if len(us) > 1 else us[0]



