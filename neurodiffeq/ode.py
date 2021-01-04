import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import FCNN
from .generators import Generator1D
from ._version_utils import warn_deprecate_class
from .monitors import Monitor1D
from .conditions import NoCondition, IVP, DirichletBVP
from .solvers import Solver1D
from copy import deepcopy
import warnings

ExampleGenerator = warn_deprecate_class(Generator1D)
Monitor = warn_deprecate_class(Monitor1D)


def _trial_solution(single_net, nets, ts, conditions):
    if single_net:  # using a single net
        us = [
            con.enforce(single_net, ts)
            for con in conditions
        ]
    else:  # using multiple nets
        us = [
            con.enforce(net, ts)
            for con, net in zip(conditions, nets)
        ]
    return us


def solve(
        ode, condition, t_min=None, t_max=None,
        net=None, train_generator=None, valid_generator=None,
        optimizer=None, criterion=None, n_batches_train=1, n_batches_valid=4,
        additional_loss_term=None, metrics=None, max_epochs=1000,
        monitor=None, return_internal=False, return_best=False, batch_size=None, shuffle=None,
):
    r"""Train a neural network to solve an ODE.

    :param ode:
        The ODE to solve.
        If the ODE is :math:`F(x, t) = 0`
        where :math:`x` is the dependent variable and :math:`t` is the independent variable,
        then `ode` should be a function that maps :math:`(x, t)` to :math:`F(x, t)`.
    :type ode: callable
    :param condition:
        The initial/boundary condition.
    :type condition: `neurodiffeq.conditions.BaseCondition`
    :param net:
        The neural network used to approximate the solution.
        Defaults to None.
    :type net: `torch.nn.Module`, optional
    :param t_min:
        The lower bound of the domain (t) on which the ODE is solved,
        only needed when train_generator or valid_generator are not specified.
        Defaults to None
    :type t_min: float
    :param t_max:
        The upper bound of the domain (t) on which the ODE is solved,
        only needed when train_generator or valid_generator are not specified.
        Defaults to None
    :type t_max: float
    :param train_generator:
        The example generator to generate 1-D training points.
        Default to None.
    :type train_generator: `neurodiffeq.generators.Generator1D`, optional
    :param valid_generator:
        The example generator to generate 1-D validation points.
        Default to None.
    :type valid_generator: `neurodiffeq.generators.Generator1D`, optional
    :param optimizer:
        The optimization method to use for training.
        Defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion:
        The loss function to use for training.
        Defaults to None.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param additional_loss_term:
        Extra terms to add to the loss function besides the part specified by `criterion`.
        The input of `additional_loss_term` should be the same as `ode`.
    :type additional_loss_term: callable
    :param metrics:
        Metrics to keep track of during training.
        The metrics should be passed as a dictionary where the keys are the names of the metrics,
        and the values are the corresponding function.
        The input functions should be the same as `ode` and the output should be a numeric value.
        The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, callable]
    :param max_epochs:
        The maximum number of epochs to train.
        Defaults to 1000.
    :type max_epochs: int, optional
    :param monitor:
        The monitor to check the status of neural network during training.
        Defaults to None.
    :type monitor: `neurodiffeq.ode.Monitor`, optional
    :param return_internal:
        Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function.
        Defaults to False.
    :type return_internal: bool, optional
    :param return_best:
        Whether to return the nets that achieved the lowest validation loss.
        Defaults to False.
    :type return_best: bool, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    :return:
        The solution of the ODE.
        The history of training loss and validation loss.
        Optionally, the nets, conditions, training generator, validation generator, optimizer and loss function.
    :rtype: tuple[`neurodiffeq.ode.Solution`, dict] or tuple[`neurodiffeq.ode.Solution`, dict, dict]


    .. note::
        This function is deprecated, use a ``neurodiffeq.solvers.Solver1D`` instead.
    """
    nets = None if not net else [net]
    return solve_system(
        ode_system=lambda x, t: [ode(x, t)], conditions=[condition],
        t_min=t_min, t_max=t_max, nets=nets,
        train_generator=train_generator, shuffle=shuffle, valid_generator=valid_generator,
        optimizer=optimizer, criterion=criterion, n_batches_train=n_batches_train, n_batches_valid=n_batches_valid,
        additional_loss_term=additional_loss_term, metrics=metrics,
        batch_size=batch_size, max_epochs=max_epochs, monitor=monitor, return_internal=return_internal,
        return_best=return_best
    )


def solve_system(
        ode_system, conditions, t_min, t_max,
        single_net=None, nets=None, train_generator=None, valid_generator=None,
        optimizer=None, criterion=None, n_batches_train=1, n_batches_valid=4,
        additional_loss_term=None, metrics=None, max_epochs=1000, monitor=None,
        return_internal=False, return_best=False, batch_size=None, shuffle=None,
):
    r"""Train a neural network to solve an ODE.

    :param ode_system:
        The ODE system to solve.
        If the ODE system consists of equations :math:`F_i(x_1, x_2, ..., x_n, t) = 0`
        where :math:`x_i` is the dependent i-th variable and :math:`t` is the independent variable,
        then `ode_system` should be a function that maps :math:`(x_1, x_2, ..., x_n, t)` to a list
        where the i-th entry is :math:`F_i(x_1, x_2, ..., x_n, t)`.
    :type ode_system: callable
    :param conditions:
        The initial/boundary conditions.
        The ith entry of the conditions is the condition that :math:`x_i` should satisfy.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param t_min:
        The lower bound of the domain (t) on which the ODE is solved,
        only needed when train_generator or valid_generator are not specified.
        Defaults to None
    :type t_min: float.
    :param t_max:
        The upper bound of the domain (t) on which the ODE is solved,
        only needed when train_generator or valid_generator are not specified.
        Defaults to None.
    :type t_max: float
    :param single_net:
        The single neural network used to approximate the solution.
        Only one of `single_net` and `nets` should be specified.
        Defaults to None
    :param single_net: `torch.nn.Module`, optional
    :param nets:
        The neural networks used to approximate the solution.
        Defaults to None.
    :type nets: list[`torch.nn.Module`], optional
    :param train_generator:
        The example generator to generate 1-D training points.
        Default to None.
    :type train_generator: `neurodiffeq.generators.Generator1D`, optional
    :param valid_generator:
        The example generator to generate 1-D validation points.
        Default to None.
    :type valid_generator: `neurodiffeq.generators.Generator1D`, optional
    :param optimizer:
        The optimization method to use for training.
        Defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion:
        The loss function to use for training.
        Defaults to None and sum of square of the output of `ode_system` will be used.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param additional_loss_term:
        Extra terms to add to the loss function besides the part specified by `criterion`.
        The input of `additional_loss_term` should be the same as `ode_system`.
    :type additional_loss_term: callable
    :param metrics:
        Additional metrics to be logged (besides loss). ``metrics`` should be a dict where

        - Keys are metric names (e.g. 'analytic_mse');
        - Values are functions (callables) that computes the metric value.
          These functions must accept the same input as the differential equation ``ode_system``.

    :type metrics: dict[str, callable], optional
    :param max_epochs:
        The maximum number of epochs to train.
        Defaults to 1000.
    :type max_epochs: int, optional
    :param monitor:
        The monitor to check the status of nerual network during training.
        Defaults to None.
    :type monitor: `neurodiffeq.ode.Monitor`, optional
    :param return_internal:
        Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function.
        Defaults to False.
    :type return_internal: bool, optional
    :param return_best:
        Whether to return the nets that achieved the lowest validation loss.
        Defaults to False.
    :type return_best: bool, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    :return:
        The solution of the ODE. The history of training loss and validation loss.
        Optionally, the nets, conditions, training generator, validation generator, optimizer and loss function.
    :rtype: tuple[`neurodiffeq.ode.Solution`, dict] or tuple[`neurodiffeq.ode.Solution`, dict, dict]


    .. note::
        This function is deprecated, use a ``neurodiffeq.solvers.Solver1D`` instead.
    """

    warnings.warn(
        "The `solve_system` function is deprecated, use a `neurodiffeq.solvers.Solver1D` instance instead",
        FutureWarning,
    )
    if single_net and nets:
        raise ValueError('Only one of net and nets should be specified')

    # For backward compatibility defaults to use a single neural network
    if (not single_net) and (not nets):
        single_net = FCNN(
            n_input_units=1,
            n_output_units=len(conditions),
            hidden_units=(32, 32),
            actv=nn.Tanh,
        )

    if single_net:
        # mark the Conditions so that we know which condition correspond to which output unit
        for ith, con in enumerate(conditions):
            con.set_impose_on(ith)
        nets = [single_net] * len(conditions)

    if additional_loss_term:
        class CustomSolver1D(Solver1D):
            def additional_loss(self, funcs, key):
                return additional_loss_term(*funcs, *self._batch_examples[key])
    else:
        class CustomSolver1D(Solver1D):
            pass

    solver = CustomSolver1D(
        ode_system=ode_system,
        conditions=conditions,
        t_min=t_min,
        t_max=t_max,
        nets=nets,
        train_generator=train_generator,
        valid_generator=valid_generator,
        optimizer=optimizer,
        criterion=criterion,
        n_batches_train=n_batches_train,
        n_batches_valid=n_batches_valid,
        metrics=metrics,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    solver.fit(max_epochs=max_epochs, monitor=monitor)
    solution = solver.get_solution(copy=True, best=return_best)
    ret = (solution, solver.metrics_history)
    if return_internal:
        params = ['nets', 'conditions', 'train_generator', 'valid_generator', 'optimizer', 'criterion']
        internals = solver.get_internals(params, return_type="dict")
        ret = ret + (internals,)
    return ret
