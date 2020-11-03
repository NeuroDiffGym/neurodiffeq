import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as tri

from .networks import FCNN
from .neurodiffeq import safe_diff as diff
from .generators import Generator2D, PredefinedGenerator
from ._version_utils import warn_deprecate_class
from .conditions import IrregularBoundaryCondition
from .conditions import NoCondition, DirichletBVP2D, IBVP1D
from copy import deepcopy

ExampleGenerator2D = warn_deprecate_class(Generator2D)
PredefinedExampleGenerator2D = warn_deprecate_class(PredefinedGenerator)


# Calculate the output of a neural network with 2 input.
# In the case where the neural network has multiple output unit, 
# `ith_unit` specifies which unit do we want.
def _network_output_2input(net, xs, ys, ith_unit):
    xys = torch.cat((xs, ys), 1)
    nn_output = net(xys)
    if ith_unit is not None:
        return nn_output[:, ith_unit].reshape(-1, 1)
    else:
        return nn_output

# Adjust the output of the neural network with trial solutions
# coded into `conditions`.
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


class Monitor2D:
    r"""A monitor for checking the status of the neural network during training.

    :param xy_min: The lower bound of 2 dimensions, if we only care about :math:`x \geq x_0` and :math:`y \geq y_0`, then `xy_min` is `(x_0, y_0)`.
    :type xy_min: tuple[float, float], optional
    :param xy_max: The upper boound of 2 dimensions, if we only care about :math:`x \leq x_1` and :math:`y \leq y_1`, then `xy_min` is `(x_1, y_1)`.
    :type xy_max: tuple[float, float], optional
    :param check_every: The frequency of checking the neural network represented by the number of epochs between two checks, defaults to 100.
    :type check_every: int, optional
    """

    def __init__(self, xy_min, xy_max, check_every=100, valid_generator=None):
        """Initializer method
        """
        self.using_non_gui_backend = (matplotlib.get_backend() == 'agg')
        self.check_every = check_every
        self.fig = None
        self.axs = []  # subplots
        # self.caxs = []  # colorbars
        self.cbs = []  # color bars
        # input for neural network
        if valid_generator is None:
            valid_generator = Generator2D([32, 32], xy_min, xy_max, method='equally-spaced')
        xs_ann, ys_ann = valid_generator.get_examples()
        self.xs_ann, self.ys_ann = xs_ann.reshape(-1, 1), ys_ann.reshape(-1, 1)
        self.xs_plot = self.xs_ann.detach().cpu().numpy().flatten()
        self.ys_plot = self.ys_ann.detach().cpu().numpy().flatten()

    # draw a contour plot of the surface (xs, ys) -> zs
    @staticmethod
    def _create_contour(ax, xs, ys, zs, condition):
        triang = tri.Triangulation(xs, ys)
        xs = xs[triang.triangles].mean(axis=1)
        ys = ys[triang.triangles].mean(axis=1)
        if condition:
            xs, ys = torch.tensor(xs), torch.tensor(ys)
            if isinstance(condition, IrregularBoundaryCondition):
                in_domain = condition.in_domain(xs, ys)
                triang.set_mask(~in_domain)

        contour = ax.tricontourf(triang, zs, cmap='coolwarm')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        return contour


    def check(self, single_net, nets, conditions, history):
        r"""Draw 2 plots: One shows the shape of the current solution (with heat map). The other shows the history training loss and validation loss.

        :param single_net: The neural network that approximates the PDE.
        :type single_net: `torch.nn.Module`
        :param nets: The neural networks that approximates the PDE.
        :type nets: list [`torch.nn.Module`]
        :param conditions: The initial/boundary condition of the PDE.
        :type conditions: list [`neurodiffeq.conditions.BaseCondition`]
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
            for i in range(n_axs-2):
                self.cbs.append(None)

        us = _trial_solution_2input(single_net, nets, self.xs_ann, self.ys_ann, conditions)

        for i, ax_u_con in enumerate( zip(self.axs[:-2], us, conditions) ):
            ax, u, con = ax_u_con
            ax.clear()
            u = u.detach().cpu().numpy().flatten()
            cs = self._create_contour(ax, self.xs_plot, self.ys_plot, u, con)
            if self.cbs[i] is None:
                self.cbs[i] = self.fig.colorbar(cs, format='%.0e', ax=ax)
            else:
                self.cbs[i].mappable.set_clim(vmin=u.min(), vmax=u.max())
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
        # if there's not custom metrics, then there won't be any labels in this axis
        if len(history) > 2:
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
    r"""Train a neural network to solve a PDE with 2 independent variables.

    :param pde: The PDE to solve. If the PDE is :math:`F(u, x, y) = 0` where :math:`u` is the dependent variable and :math:`x` and :math:`y` are the independent variables,
        then `pde` should be a function that maps :math:`(u, x, y)` to :math:`F(u, x, y)`.
    :type pde: function
    :param condition: The initial/boundary condition.
    :type condition: `neurodiffeq.conditions.BaseCondition`
    :param xy_min: The lower bound of 2 dimensions, if we only care about :math:`x \geq x_0` and :math:`y \geq y_0`, then `xy_min` is `(x_0, y_0)`, only needed when train_generator and valid_generator are not specified, defaults to None
    :type xy_min: tuple[float, float], optional
    :param xy_max: The upper bound of 2 dimensions, if we only care about :math:`x \leq x_1` and :math:`y \leq y_1`, then `xy_min` is `(x_1, y_1)`, only needed when train_generator and valid_generator are not specified, defaults to None
    :type xy_max: tuple[float, float], optional
    :param net: The neural network used to approximate the solution, defaults to None.
    :type net: `torch.nn.Module`, optional
    :param train_generator: The example generator to generate 1-D training points, default to None.
    :type train_generator: `neurodiffeq.generator.Generator2D`, optional
    :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
    :type shuffle: bool, optional
    :param valid_generator: The example generator to generate 1-D validation points, default to None.
    :type valid_generator: `neurodiffeq.generator.Generator2D`, optional
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
    r"""Train a neural network to solve a PDE with 2 independent variables.

        :param pde_system: The PDEsystem to solve. If the PDE is :math:`F_i(u_1, u_2, ..., u_n, x, y) = 0` where :math:`u_i` is the i-th dependent variable and :math:`x` and :math:`y` are the independent variables,
            then `pde_system` should be a function that maps :math:`(u_1, u_2, ..., u_n, x, y)` to a list where the i-th entry is :math:`F_i(u_1, u_2, ..., u_n, x, y)`.
        :type pde_system: function
        :param conditions: The initial/boundary conditions. The ith entry of the conditions is the condition that :math:`x_i` should satisfy.
        :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
        :param xy_min: The lower bound of 2 dimensions, if we only care about :math:`x \geq x_0` and :math:`y \geq y_0`, then `xy_min` is `(x_0, y_0)`, only needed when train_generator or valid_generator are not specified, defaults to None
        :type xy_min: tuple[float, float], optional
        :param xy_max: The upper bound of 2 dimensions, if we only care about :math:`x \leq x_1` and :math:`y \leq y_1`, then `xy_min` is `(x_1, y_1)`, only needed when train_generator or valid_generator are not specified, defaults to None
        :type xy_max: tuple[float, float], optional
        :param single_net: The single neural network used to approximate the solution. Only one of `single_net` and `nets` should be specified, defaults to None
        :param single_net: `torch.nn.Module`, optional
        :param nets: The neural networks used to approximate the solution, defaults to None.
        :type nets: list[`torch.nn.Module`], optional
        :param train_generator: The example generator to generate 1-D training points, default to None.
        :type train_generator: `neurodiffeq.generator.Generator2D`, optional
        :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
        :type shuffle: bool, optional
        :param valid_generator: The example generator to generate 1-D validation points, default to None.
        :type valid_generator: `neurodiffeq.generator.Generator2D`, optional
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
        train_loss_epoch = train_loss_epoch.item()

        train_metrics_epoch = calculate_metrics(train_examples_x, train_examples_y, net, nets, conditions, metrics)
        return train_loss_epoch, train_metrics_epoch

    # Vliadate the neural network, return the validation loss and validation metrics
    def valid(valid_generator, net, nets, pde_system, conditions, criterion, additional_loss_term, metrics):
        valid_examples_x, valid_examples_y = valid_generator.get_examples()
        valid_examples_x, valid_examples_y = valid_examples_x.reshape((-1, 1)), valid_examples_y.reshape((-1, 1))
        valid_loss_epoch = calculate_loss(valid_examples_x, valid_examples_y, net, nets, pde_system, conditions, criterion, additional_loss_term)
        valid_loss_epoch = valid_loss_epoch.item()

        valid_metrics_epoch = calculate_metrics(valid_examples_x, valid_examples_y, net, nets, conditions, metrics)
        return valid_loss_epoch, valid_metrics_epoch

    # calculate the loss function
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

    # caclulate the metrics
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
        train_generator = Generator2D((32, 32), xy_min, xy_max, method='equally-spaced-noisy')
    if not valid_generator:
        if (xy_min is None) or (xy_max is None):
            raise RuntimeError('Please specify xy_min and xy_max when valid_generator is not specified')
        valid_generator = Generator2D((32, 32), xy_min, xy_max, method='equally-spaced')
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
    r"""A solution to an PDE (system)

    :param single_net: The neural networks that approximates the PDE.
    :type single_net: `torch.nn.Module`
    :param nets: The neural networks that approximates the PDE.
    :type nets: list[`torch.nn.Module`]
    :param conditions: The initial/boundary conditions of the ODE (system).
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
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
        :type xs: `torch.Tensor` or sequence of number
        :param ys: the y-coordinates of points on which the dependent variables are evaluated.
        :type ys: `torch.Tensor` or sequence of number
        :param as_type: Whether the returned value is a `torch.Tensor` ('tf') or `numpy.array` ('np').
        :type as_type: str
        :return: dependent variables are evaluated at given points.
        :rtype: list[`torch.Tensor` or `numpy.array` (when there is more than one dependent variables)
            `torch.Tensor` or `numpy.array` (when there is only one dependent variable).
        """
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys)
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
    r"""Create animation of 1-D time-dependent problems.

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

############################# arbitraty boundary conditions #############################

# CONSTANTS
ROUND_TO_ZERO = 1e-7  # in the code below, values lower than ROUND_TO_ZERO are considered zero
K = 5.0
ALPHA = 5.0


class Point:
    r"""A 2D point.

    :param loc: the location of the point in the form of :math:`(x, y)`
    :type loc: tuple[float, float]
    """

    def __repr__(self):
        return f'Point({self.loc})'

    def __init__(self, loc):
        self.loc = tuple(float(d) for d in loc)
        self.dim = len(loc)


class DirichletControlPoint(Point):
    r"""A 2D point on the Dirichlet boundary.

    :param loc: the location of the point in the form of :math:`(x, y)`
    :type loc: tuple[float, float]
    :param val: the expected value of :math:`u` at this location (:math:`u(x, y)` is the function we are solving for)
    :type val: float
    """

    def __repr__(self):
        return f'DirichletControlPoint({self.loc}, val={self.val})'

    def __init__(self, loc, val):
        super().__init__(loc)
        self.val = float(val)


# TODO: Irregular Neumann boundary conditions are not working yet.
class NeumannControlPoint(Point):
    r"""A 2D point on the Neumann boundary.

    :param loc: the location of the point in the form of :math:`(x, y)`
    :type loc: tuple[float, float]
    :param val: the expected normal derivative of :math:`u` at this location (:math:`u(x, y)` is the function we are solving for)
    :type val: float
    """

    def __repr__(self):
        return f'NeumannControlPoint({self.loc}, val={self.val}, ' + \
               f'normal_vector={self.normal_vector})'

    def __init__(self, loc, val, normal_vector):
        super().__init__(loc)
        self.val = float(val)
        scale = sum(d ** 2 for d in normal_vector) ** 0.5
        self.normal_vector = tuple(d / scale for d in normal_vector)


class CustomBoundaryCondition(IrregularBoundaryCondition):
    r"""A boundary condition with irregular shape.

    :param center_point: A point that roughly locate at the center of the domain. It will be used to sort the control points 'clockwise'.
    :type center_point: `pde.Point`
    :param dirichlet_control_points: a list of points on the Dirichlet boundary
    :type dirichlet_control_points: list[pde.DirichletControlPoint]
    """

    def __init__(self, center_point, dirichlet_control_points, neumann_control_points=None):
        super().__init__()

        # for Dirichlet control points, drop deplicates and sort 'clockwise'
        self.dirichlet_control_points = self._clean_control_points(dirichlet_control_points, center_point)
        # fit Neumann boundary condition (A_D(x) in MacFall's paper)
        self.a_d_interp = InterpolatorCreator.fit_surface(self.dirichlet_control_points)
        # fit Dirichlet length factor (L_D(x) in MacFall's paper)
        self.l_d_interp = InterpolatorCreator.fit_length_factor(self.dirichlet_control_points)

        if neumann_control_points is None:
            neumann_control_points = []
        if len(neumann_control_points) > 0:
            # for Naumann control points, drop deplicates and sort 'clockwise'
            self.neumann_control_points = self._clean_control_points(neumann_control_points, center_point)
            # fit Neumann boundary condition (g(x) in MacFall's paper)
            self.g_interp = InterpolatorCreator.fit_surface(self.neumann_control_points)
            # fit Naumann length factor (L_M(x) in MacFall's paper)
            self.l_m_interp = InterpolatorCreator.fit_length_factor(self.neumann_control_points)
            # fit normal vector (n_hat(x) in MacFall's paper)
            self.n_hat_interp = InterpolatorCreator.fit_normal_vector(self.neumann_control_points)
        else:
            # the following fields are not needed when we don't have a Neumann boundary condition
            self.neumann_control_points = None
            self.g_interp = None
            self.l_m_interp = None
            self.n_hat_interp = None

    # A_D(x) in MacFall's paper
    def a_d(self, *dimensions):
        return self.a_d_interp.interpolate(dimensions)

    # L_D(x) in MacFall's paper
    def l_d(self, *dimensions):
        return self.l_d_interp.interpolate(dimensions)

    # g(x) in MacFall's paper
    def g(self, *dimensions):
        return self.g_interp.interpolate(dimensions)

    # L_M(x) in MacFall's paper
    def l_m(self, *dimensions):
        return self.l_m_interp.interpolate(dimensions)

    # F(x) in MacFall's paper
    def f(self, net, *dimensions):
        # return self.l_d(*dimensions) * self._nn_output(net, *dimensions)
        return self.l_d(*dimensions) * _network_output_2input(net, *dimensions, self.ith_unit)

    # n^hat(x) in MacFall's paper
    def n_hat(self, *dimensions):
        return self.n_hat_interp.interpolate(dimensions)

    # A_M(x) in MacFall's paper
    def a_m(self, net, *dimensions):
        # when we don't have a Neumann boundary condition
        if self.neumann_control_points is None:
            return 0.0

        fs = self.f(net, *dimensions)
        a_ds = self.a_d(*dimensions)
        l_ds = self.l_d(*dimensions)
        l_ms = self.l_m(*dimensions)
        n_hats = self.n_hat(*dimensions)

        numer = self.g(*dimensions) - sum(
            nk * (diff(a_ds, d) + diff(fs, d))
            for nk, d in zip(n_hats, dimensions)
        )
        denom = l_ds * sum(
            nk * diff(l_ms, d)
            for nk, d in zip(n_hats, dimensions)
        ) + K * (1 - torch.exp(-ALPHA * l_ms))

        return l_ds * l_ms * numer / denom

    # This method is called by Monitor2D when creating a contour. It returns a 
    # mask indicating whether a point are inside the problem. The mask is used
    # to crop the contour plot. Here we assume if a point (x, y) have positive
    # length factor L_D(x, y) then it's inside the domain.
    def in_domain(self, *dimensions):
        # when we don't have a Neumann boundary condition
        if self.neumann_control_points is None:
            return self.l_d(*dimensions) > 0.0
        return (self.l_d(*dimensions) > 0.0) & (self.l_m(*dimensions) > 0.0)

    def enforce(self, net, *dimensions):
        # enforce Dirichlet and Neumann boundary condition, equation[10] in MAcfall's paper
        return self.a_d(*dimensions) + self.a_m(net, *dimensions) + self.f(net, *dimensions)

    # This method removes the control points that are too close to each other 
    # and sort the control points 'clockwise' (the center is the center_point).
    # We sort the points because we want to map them to equally spaced points on
    # a circle, and the points need to be ordered so that we can assign corresponding
    # target points.
    @staticmethod
    def _clean_control_points(control_points, center_point):

        def gt_zero(number):
            return number >= ROUND_TO_ZERO

        def lt_zero(number):
            return number <= -ROUND_TO_ZERO

        def eq_zero(number):
            return abs(number) < ROUND_TO_ZERO

        def clockwise(cp):
            px, py = cp.loc
            cx, cy = center_point.loc
            dx, dy = px - cx, py - cy
            if gt_zero(dx) and eq_zero(dy):
                tier = 0
            elif gt_zero(dx) and lt_zero(dy):
                tier = 1
            elif eq_zero(dx) and lt_zero(dy):
                tier = 2
            elif lt_zero(dx) and lt_zero(dy):
                tier = 3
            elif lt_zero(dx) and eq_zero(dy):
                tier = 4
            elif lt_zero(dx) and gt_zero(dy):
                tier = 5
            elif eq_zero(dx) and gt_zero(dy):
                tier = 6
            elif gt_zero(dx) and gt_zero(dy):
                tier = 7
            # assume that the second key won't be used
            # - i.e. on the same side of center point (left or right)
            # there won't be multiple control points that
            # has the same y-coordinate as the center point
            return (tier, dx / dy if not eq_zero(dy) else 0)

        control_points.sort(key=clockwise)

        def same_point(p1, p2):
            return eq_zero(p1.loc[0] - p2.loc[0]) and eq_zero(p1.loc[1] - p2.loc[1])

        # remove the control points that are defined more than once
        unique_control_points = [control_points[0]]
        for cp in control_points[1:]:
            if not same_point(cp, unique_control_points[-1]):
                unique_control_points.append(cp)
        return unique_control_points


class InterpolatorCreator:

    # Create an interpolator to map (x, y) -> A_D(x, y)
    @staticmethod
    def fit_surface(dirichlet_or_neumann_control_points):
        # specify input and output of thin plate spline
        from_points = dirichlet_or_neumann_control_points
        to_values = [dncp.val for dncp in dirichlet_or_neumann_control_points]
        # fit thin plate spline and save coefficients
        coefs = InterpolatorCreator._solve_thin_plate_spline(from_points, to_values)
        return SurfaceInterpolator(coefs, dirichlet_or_neumann_control_points)

    # Create an interpolator to map (x, y) -> L_D(x, y)
    @staticmethod
    def fit_length_factor(control_points, radius=0.5):
        # specify input and output of thin plate spline
        from_points = control_points
        to_points = InterpolatorCreator._create_circular_targets(control_points, radius)
        n_dim = to_points[0].dim
        to_values_each_dim = [[tp.loc[i] for tp in to_points] for i in range(n_dim)]
        # fit thin plate spline and save coefficients
        coefs_each_dim = [
            InterpolatorCreator._solve_thin_plate_spline(from_points, to_values)
            for to_values in to_values_each_dim
        ]

        return LengthFactorInterpolator(coefs_each_dim, control_points, radius)

    # Create an interpolator to map (x, y) -> \hat{n}(x, y) (The normal vector
    # on the boundary). It's only used in Neumann boundary conditions, which is
    # not working yet.
    @staticmethod
    def fit_normal_vector(neumann_control_points):
        # specify input and output of thin plate spline
        from_points = neumann_control_points
        to_points = [
            Point(loc=(ncp.normal_vector[0], ncp.normal_vector[1]))
            for ncp in neumann_control_points
        ]
        n_dim = to_points[0].dim
        to_values_each_dim = [[tp.loc[i] for tp in to_points] for i in range(n_dim)]
        # fit thin plate spline and save coefficients
        coefs_each_dim = [
            InterpolatorCreator._solve_thin_plate_spline(from_points, to_values)
            for to_values in to_values_each_dim
        ]
        return NormalVectorInterpolator(coefs_each_dim, neumann_control_points)

    # All the interpolators are based on thin plate splines. This method fits the
    # coefficients of the thin plate spline.
    @staticmethod
    def _solve_thin_plate_spline(from_points, to_values):
        assert len(from_points) == len(to_values)
        n_dims = from_points[0].dim
        n_pnts = len(from_points)
        n_eqs = n_dims + n_pnts + 1

        # weights of the eq_no'th equation
        def equation_weights(eq_no):

            weights = np.zeros(n_eqs)

            # the first M equations (M is the number of control points)
            if eq_no < n_pnts:
                p = from_points[eq_no]
                # the first M weights
                for i, fp in enumerate(from_points):
                    ri_sq = Interpolator._ri_sq_thin_plate_spline_pretrain(p, fp)
                    weights[i] = ri_sq * np.log(ri_sq)
                # the M+1'th weight
                weights[n_pnts] = 1.0
                # the rest #dimension weights
                for j in range(n_dims):
                    weights[n_pnts + 1 + j] = p.loc[j]
            # the M+1'th equation
            elif eq_no < n_pnts + n_dims:
                j = eq_no - n_pnts
                for i in range(n_pnts):
                    weights[i] = from_points[i].loc[j]
            # the rest #dimension equations
            elif eq_no == n_pnts + n_dims:
                weights[:n_pnts] = 1.0
            else:
                raise ValueError(f'Invalid equation number: {eq_no}')

            return weights

        # create linear system
        W = np.zeros((n_eqs, n_eqs))
        for eq_no in range(n_eqs):
            W[eq_no] = equation_weights(eq_no)
        b = np.zeros(n_eqs)
        b[:n_pnts] = to_values

        # solve linear system and return coefficients
        return np.linalg.solve(W, b)

    @staticmethod
    def _create_circular_targets(control_points, radius):
        # create equally spaced target points, this is for 2-d control points
        # TODO 3-d control points
        return [
            Point((radius * np.cos(theta), radius * np.sin(theta)))
            for theta in -np.linspace(0, 2 * np.pi, len(control_points), endpoint=False)
        ]


# Interpolatror are used for calculating L_D, A_D and \hat{n}
class Interpolator:

    def interpolate(self, dimensions):
        raise NotImplementedError

    # calculate the output of the thin plate spline using the fitted coefficients
    @staticmethod
    def _interpolate_by_thin_plate_spline(coefs, control_points, dimensions):
        n_pnts = len(control_points)
        to_value_unfinished = torch.zeros_like(dimensions[0])
        # the first M basis functions (M is the number of control points)
        for coef, cp in zip(coefs, control_points):
            ri_sq = Interpolator._ri_sq_thin_plate_spline_trainval(cp, dimensions)
            to_value_unfinished += coef * ri_sq * torch.log(ri_sq)
        # the M+1'th basis function
        to_value_unfinished += coefs[n_pnts]
        # the rest #dimension basis functions
        for j, d in enumerate(dimensions):
            to_value_unfinished += coefs[n_pnts + 1 + j] * d
        to_value = to_value_unfinished
        return to_value

    # to be used in fitting coefficients of thin plate spline
    @staticmethod
    def _ri_sq_thin_plate_spline_pretrain(point_i, point_j, stiffness=0.01):
        return sum((di - dj) ** 2 for di, dj in zip(point_i.loc, point_j.loc)) + stiffness ** 2

    # to be used in transforming output of neural networks
    @staticmethod
    def _ri_sq_thin_plate_spline_trainval(point_i, dimensions, stiffness=0.01):
        return sum((d - di) ** 2 for di, d in zip(point_i.loc, dimensions)) + stiffness ** 2


# calculating A_D
class SurfaceInterpolator(Interpolator):

    def __init__(self, coefs, control_points):
        self.coefs = coefs
        self.control_points = control_points

    # Return A_D(x, y), dimensions are a tuple: (x-tensor, y-tensor)
    def interpolate(self, dimensions):
        return Interpolator._interpolate_by_thin_plate_spline(
            self.coefs, self.control_points, dimensions
        )


# calculating L_D
class LengthFactorInterpolator(Interpolator):

    def __init__(self, coefs_each_dim, control_points, radius):
        self.coefs_each_dim = coefs_each_dim
        self.control_points = control_points
        self.radius = radius

    # Return L_D(x, y), dimensions are a tuple: (x-tensor, y-tensor)
    def interpolate(self, dimensions):
        dimensions_mapped = tuple(
            Interpolator._interpolate_by_thin_plate_spline(
                coefs_dim, self.control_points, dimensions
            )
            for coefs_dim in self.coefs_each_dim
        )
        return self.radius ** 2 - sum(d ** 2 for d in dimensions_mapped)


# calculating \hat{n}. It's only used in Neumann boundary conditions, which is
# not working yet.
class NormalVectorInterpolator(Interpolator):

    def __init__(self, coefs_each_dim, neumann_control_points):
        self.coefs_each_dim = coefs_each_dim
        self.neumann_control_points = neumann_control_points

    # Return \hat{n}(x, y), dimensions are a tuple: (x-tensor, y-tensor)
    def interpolate(self, dimensions):
        dimensions_mapped = tuple(
            Interpolator._interpolate_by_thin_plate_spline(
                coefs_dim, self.neumann_control_points, dimensions
            )
            for coefs_dim in self.coefs_each_dim
        )
        return dimensions_mapped
