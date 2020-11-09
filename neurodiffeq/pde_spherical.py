import os
import sys
import dill
import warnings
import logging
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from .function_basis import RealSphericalHarmonics

from .networks import FCNN
from ._version_utils import warn_deprecate_class
from .generators import Generator3D, GeneratorSpherical
from .conditions import NoCondition, DirichletBVPSpherical, InfDirichletBVPSpherical
from .conditions import DirichletBVPSphericalBasis, InfDirichletBVPSphericalBasis
from inspect import signature
from copy import deepcopy
from datetime import datetime

# generators defined in this module have been move to generators.py (and renamed)
ExampleGenerator3D = warn_deprecate_class(Generator3D)
ExampleGeneratorSpherical = warn_deprecate_class(GeneratorSpherical)

# conditions defined in this module have been moved to conditions.py (and renamed)
NoConditionSpherical = warn_deprecate_class(NoCondition)
NoConditionSphericalHarmonics = warn_deprecate_class(NoCondition)
DirichletBVPSpherical = warn_deprecate_class(DirichletBVPSpherical)
DirichletBVPSphericalHarmonics = warn_deprecate_class(DirichletBVPSphericalBasis)
InfDirichletBVPSpherical = warn_deprecate_class(InfDirichletBVPSpherical)
InfDirichletBVPSphericalHarmonics = warn_deprecate_class(InfDirichletBVPSphericalBasis)


def _nn_output_spherical_input(net, rs, thetas, phis):
    points = torch.cat((rs, thetas, phis), 1)
    return net(points)


class SolutionSpherical:
    """A solution to a PDE (system) in spherical coordinates

    :param nets: The neural networks that approximate the PDE.
    :type nets: list[`torch.nn.Module`]
    :param conditions: The conditions of the PDE (system).
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    """

    def __init__(self, nets, conditions):
        """Initializer method
        """
        self.nets = deepcopy(nets)
        self.conditions = deepcopy(conditions)

    def _compute_u(self, net, condition, rs, thetas, phis):
        return condition.enforce(net, rs, thetas, phis)

    def __call__(self, rs, thetas, phis, as_type='tf'):
        """Evaluate the solution at certain points.

        :param rs: The radii of points where the neural network output is evaluated.
        :type rs: `torch.Tensor`
        :param thetas: The latitudes of points where the neural network output is evaluated. `theta` ranges [0, pi]
        :type thetas: `torch.Tensor`
        :param phis: The longitudes of points where the neural network output is evaluated. `phi` ranges [0, 2*pi)
        :type phis: `torch.Tensor`
        :param as_type: Whether the returned value is a `torch.Tensor` ('tf') or `numpy.array` ('np').
        :type as_type: str
        :return: dependent variables are evaluated at given points.
        :rtype: list[`torch.Tensor` or `numpy.array` (when there is more than one dependent variables)
            `torch.Tensor` or `numpy.array` (when there is only one dependent variable)
        """
        if not isinstance(rs, torch.Tensor):
            rs = torch.tensor(rs)
        if not isinstance(thetas, torch.Tensor):
            thetas = torch.tensor(thetas)
        if not isinstance(phis, torch.Tensor):
            phis = torch.tensor(phis)
        original_shape = rs.shape
        rs = rs.reshape(-1, 1)
        thetas = thetas.reshape(-1, 1)
        phis = phis.reshape(-1, 1)
        if as_type not in ('tf', 'np'):
            raise ValueError("The valid return types are 'tf' and 'np'.")

        vs = [
            self._compute_u(net, con, rs, thetas, phis).reshape(original_shape)
            for con, net in zip(self.conditions, self.nets)
        ]
        if as_type == 'np':
            vs = [v.detach().cpu().numpy().flatten() for v in vs]

        return vs if len(self.nets) > 1 else vs[0]


def solve_spherical(
        pde, condition, r_min=None, r_max=None,
        net=None, train_generator=None, shuffle=True, valid_generator=None, analytic_solution=None,
        optimizer=None, criterion=None, batch_size=16, max_epochs=1000,
        monitor=None, return_internal=False, return_best=False, harmonics_fn=None,
):
    """[DEPRECATED, use SphericalSolver class instead] Train a neural network to solve one PDE with spherical inputs in 3D space

        :param pde: The PDE to solve. If the PDE is :math:`F(u, r,\\theta, \\phi) = 0` where :math:`u` is the dependent variable and :math:`r`, :math:`\\theta` and :math:`\\phi` are the independent variables,
            then `pde` should be a function that maps :math:`(u, r, \\theta, \\phi)` to :math:`F(u, r,\\theta, \\phi)`
        :type pde: callable
        :param condition: The initial/boundary condition that :math:`u` should satisfy.
        :type condition: `neurodiffeq.conditions.BaseCondition`
        :param r_min: radius for inner boundary; ignored if both generators are provided; optional
        :type r_min: float
        :param r_max: radius for outer boundary; ignored if both generators are provided; optional
        :type r_max: float
        :param net: The neural network used to approximate the solution, defaults to None.
        :type net: `torch.nn.Module`, optional
        :param train_generator: The example generator to generate 3-D training points, default to None.
        :type train_generator: `neurodiffeq.pde_spherical.BaseGenerator`, optional
        :param valid_generator: The example generator to generate 3-D validation points, default to None.
        :type valid_generator: `neurodiffeq.pde_spherical.BaseGenerator`, optional
        :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
        :type shuffle: bool, optional
        :param analytic_solution: analytic solution to the pde system, used for testing purposes; should map (rs, thetas, phis) to u
        :type analytic_solution: callable
        :param optimizer: The optimization method to use for training, defaults to None.
        :type optimizer: `torch.optim.Optimizer`, optional
        :param criterion: The loss function to use for training, defaults to None.
        :type criterion: `torch.nn.modules.loss._Loss`, optional
        :param batch_size: The size of the mini-batch to use, defaults to 16.
        :type batch_size: int, optional
        :param max_epochs: The maximum number of epochs to train, defaults to 1000.
        :type max_epochs: int, optional
        :param monitor: The monitor to check the status of neural network during training, defaults to None.
        :type monitor: `neurodiffeq.pde_spherical.MonitorSpherical`, optional
        :param return_internal: Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function, defaults to False.
        :type return_internal: bool, optional
        :param return_best: Whether to return the nets that achieved the lowest validation loss, defaults to False.
        :type return_best: bool, optional
        :param harmonics_fn: function basis (spherical harmonics for example) if solving coefficients of a function basis; used when returning solution
        :type harmonics_fn: callable
        :return: The solution of the PDE. The history of training loss and validation loss.
            Optionally, MSE against analytic solution, the nets, conditions, training generator, validation generator, optimizer and loss function.
            The solution is a function that has the signature `solution(xs, ys, as_type)`.
        :rtype: tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict]; or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict]; or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict, dict]


        .. note::
            This function is deprecated, use a `SphericalSolver` instead
        """

    warnings.warn("solve_spherical is deprecated, consider using SphericalSolver instead")
    pde_sytem = lambda u, r, theta, phi: [pde(u, r, theta, phi)]
    conditions = [condition]
    nets = [net] if net is not None else None
    if analytic_solution is None:
        analytic_solutions = None
    else:
        analytic_solutions = lambda r, theta, phi: [analytic_solution(r, theta, phi)]

    return solve_spherical_system(
        pde_system=pde_sytem, conditions=conditions, r_min=r_min, r_max=r_max,
        nets=nets, train_generator=train_generator, shuffle=shuffle, valid_generator=valid_generator,
        analytic_solutions=analytic_solutions, optimizer=optimizer, criterion=criterion, batch_size=batch_size,
        max_epochs=max_epochs, monitor=monitor, return_internal=return_internal, return_best=return_best,
        harmonics_fn=harmonics_fn,
    )


def solve_spherical_system(
        pde_system, conditions, r_min=None, r_max=None,
        nets=None, train_generator=None, shuffle=True, valid_generator=None, analytic_solutions=None,
        optimizer=None, criterion=None, batch_size=None,
        max_epochs=1000, monitor=None, return_internal=False, return_best=False, harmonics_fn=None
):
    """[DEPRECATED, use SphericalSolver class instead] Train a neural network to solve a PDE system with spherical inputs in 3D space

        :param pde_system: The PDEs ystem to solve. If the PDE is :math:`F_i(u_1, u_2, ..., u_n, r,\\theta, \\phi) = 0` where :math:`u_i` is the i-th dependent variable and :math:`r`, :math:`\\theta` and :math:`\\phi` are the independent variables,
            then `pde_system` should be a function that maps :math:`(u_1, u_2, ..., u_n, r, \\theta, \\phi)` to a list where the i-th entry is :math:`F_i(u_1, u_2, ..., u_n, r, \\theta, \\phi)`.
        :type pde_system: callable
        :param conditions: The initial/boundary conditions. The ith entry of the conditions is the condition that :math:`u_i` should satisfy.
        :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
        :param r_min: radius for inner boundary; ignored if both generators are provided; optional
        :type r_min: float
        :param r_max: radius for outer boundary; ignored if both generators are provided; optional
        :type r_max: float
        :param nets: The neural networks used to approximate the solution, defaults to None.
        :type nets: list[`torch.nn.Module`], optionalnerate 3-D training points, default to None.
        :param train_generator: The example generator to generate 3-D training points, default to None.
        :type train_generator: `neurodiffeq.pde_spherical.BaseGenerator`, optional
        :param valid_generator: The example generator to generate 3-D validation points, default to None.
        :type valid_generator: `neurodiffeq.pde_spherical.BaseGenerator`, optional
        :param shuffle: deprecated and ignored; shuffling should be implemented in genrators
        :type shuffle: bool, optional
        :param analytic_solutions: analytic solution to the pde system, used for testing purposes; should map (rs, thetas, phis) to a list of [u_1, u_2, ..., u_n]
        :type analytic_solutions: callable
        :param optimizer: The optimization method to use for training, defaults to None.
        :type optimizer: `torch.optim.Optimizer`, optional
        :param criterion: The loss function to use for training, defaults to None.
        :type criterion: `torch.nn.modules.loss._Loss`, optional
        :param batch_size: The size of the mini-batch to use, defaults to 16.
        :type batch_size: int, optional
        :param max_epochs: The maximum number of epochs to train, defaults to 1000.
        :type max_epochs: int, optional
        :param monitor: The monitor to check the status of neural network during training, defaults to None.
        :type monitor: `neurodiffeq.pde_spherical.MonitorSpherical`, optional
        :param return_internal: Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function, defaults to False.
        :type return_internal: bool, optional
        :param return_best: Whether to return the nets that achieved the lowest validation loss, defaults to False.
        :type return_best: bool, optional
        :param harmonics_fn: function basis (spherical harmonics for example) if solving coefficients of a function basis; used when returning solution
        :type harmonics_fn: callable
        :return: The solution of the PDE. The history of training loss and validation loss.
            Optionally, MSE against analytic solutions, the nets, conditions, training generator, validation generator, optimizer and loss function.
            The solution is a function that has the signature `solution(xs, ys, as_type)`.
        :rtype: tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict]; or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict]; or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict, dict]

        .. note::
            This function is deprecated, use a `SphericalSolver` instead
        """
    warnings.warn("solve_spherical_system is deprecated, consider using SphericalSolver instead")

    if harmonics_fn is None:
        def enforcer(net, cond, points):
            return cond.enforce(net, *points)
    else:
        def enforcer(net, cond, points):
            return (cond.enforce(net, points[0]) * harmonics_fn(*points[1:])).sum(dim=1, keepdims=True)

    solver = SphericalSolver(
        pde_system=pde_system,
        conditions=conditions,
        r_min=r_min,
        r_max=r_max,
        nets=nets,
        train_generator=train_generator,
        valid_generator=valid_generator,
        analytic_solutions=analytic_solutions,
        optimizer=optimizer,
        criterion=criterion,
        n_batches_train=1,
        n_batches_valid=1,
        enforcer=enforcer,
        # deprecated arguments
        batch_size=batch_size,
        shuffle=shuffle,
    )

    solver.fit(max_epochs=max_epochs, monitor=monitor)
    solution = solver.get_solution(copy=True, best=return_best, harmonics_fn=harmonics_fn)
    ret = (solution, solver.loss)
    if analytic_solutions is not None:
        ret = ret + (solver.analytic_mse,)
    if return_internal:
        params = ['nets', 'conditions', 'train_generator', 'valid_generator', 'optimizer', 'criterion']
        internals = solver.get_internals(params, return_type="dict")
        ret = ret + (internals,)
    return ret


class SphericalSolver:
    """A solver class for solving PDEs in spherical coordinates
    
    :param pde_system: the PDE system to solve; maps a tuple of three coordinates to a tuple of PDE residuals, both the coordinates and PDE residuals must have shape (-1, 1)
    :type pde_system: callable
    :param conditions: list of boundary conditions for each target function
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param r_min: radius for inner boundary; ignored if train_generator and valid_generator are both set; r_min > 0; optional
    :type r_min: float
    :param r_max: radius for outer boundary; ignored if train_generator and valid_generator are both set; r_max > r_min; optional
    :type r_max: float
    :param nets: list of neural networks for parameterized solution; if provided, length must equal that of conditions; optional
    :type nets: list[torch.nn.Module]
    :param train_generator: generator for sampling training points, must provide a .get_examples() method and a .size field; optional
    :type train_generator: `neurodiffeq.pde_spherical.BaseGenerator`
    :param valid_generator: generator for sampling validation points, must provide a .get_examples() method and a .size field; optional
    :type valid_generator: `neurodiffeq.pde_spherical.BaseGenerator`
    :param analytic_solutions: analytical solutions to be compared with neural net solutions; maps a tuple of three coordinates to a tuple of function values; output shape shoule match that of networks; optional
    :type analytic_solutions: callable
    :param optimizer: optimizer to be used for training; optional
    :type optimizer: torch.nn.optim.Optimizer
    :param criterion: function that maps a PDE residual vector (torch tensor with shape (-1, 1)) to a scalar loss; optional
    :type criterion: callable
    :param n_batches_train: number of batches to train in every epoch; where batch-size equals train_generator.size
    :type n_batches_train: int
    :param n_batches_valid: number of batches to valid in every epoch; where batch-size equals valid_generator.size
    :type n_batches_valid: int
    :param enforcer: a function mapping a network, a condition, and a batch (returned by a generator) to the function values evaluated on the batch
    :type enforcer: callable
    :param batch_size: DEPRECATED and IGNORED; each batch will use all samples generated, specify n_batches_train and n_batches_valid instead
    :type batch_size: int
    :param shuffle: deprecated; shuffling should be performed by generators
    :type shuffle: bool
    """

    def __init__(self, pde_system, conditions, r_min=None, r_max=None,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None,
                 optimizer=None, criterion=None, n_batches_train=1, n_batches_valid=4, enforcer=None,
                 # deprecated arguments are listed below
                 shuffle=False, batch_size=None):

        if shuffle:
            warnings.warn("param `shuffle` is deprecated and ignored; shuffling should be performed by generators")

        if batch_size is not None:
            warnings.warn("param `batch_size` is deprecated and ignored; "
                          "specify n_batches_train and n_batches_valid instead")

        if train_generator is None or valid_generator is None:
            if r_min is None or r_max is None:
                raise ValueError(f"Either generator is not provided, r_min and r_max should be both provided: "
                                 f"got r_min={r_min}, r_max={r_max}, train_generator={train_generator}, "
                                 f"valid_generator={valid_generator}")

        self.pdes = pde_system
        self.conditions = conditions
        self.n_funcs = len(conditions)
        self.r_min = r_min
        self.r_max = r_max
        if nets is None:
            self.nets = [
                FCNN(n_input_units=3, n_hidden_units=32, n_hidden_layers=1, actv=nn.Tanh)
                for _ in range(self.n_funcs)
            ]
        else:
            self.nets = nets

        if train_generator is None:
            train_generator = GeneratorSpherical(512, r_min, r_max, method='equally-spaced-noisy')

        if valid_generator is None:
            valid_generator = GeneratorSpherical(512, r_min, r_max, method='equally-spaced-noisy')

        self.analytic_solutions = analytic_solutions
        self.enforcer = enforcer

        if optimizer is None:
            all_params = []
            for n in self.nets:
                all_params += n.parameters()
            self.optimizer = optim.Adam(all_params, lr=0.001)
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = lambda residual_tensor: (residual_tensor ** 2).mean()
        else:
            self.criterion = criterion

        def make_pair_dict(train=None, valid=None):
            return {'train': train, 'valid': valid}

        self.generator = make_pair_dict(train=train_generator, valid=valid_generator)
        # loss history
        self.loss = make_pair_dict(train=[], valid=[])
        # analytic MSE history
        self.analytic_mse = make_pair_dict(train=[], valid=[])
        # number of batches for training / validation;
        self.n_batches = make_pair_dict(train=n_batches_train, valid=n_batches_valid)
        # current batch of samples, kept for additional_loss term to use
        self._batch_examples = make_pair_dict()
        # current network with lowest loss
        self.best_nets = None
        # current lowest loss
        self.lowest_loss = None
        # local epoch in a `.fit` call, should only be modified inside self.fit()
        self.local_epoch = 0
        # maximum local epochs to run in a `.fit()` call, should only set by inside self.fit()
        self._max_local_epoch = 0
        # controls early stopping, should be set to False at the beginning of a `.fit()` call
        # and optionally set to False by `callbacks` in `.fit()` to support early stopping
        self._stop_training = False
        # the _phase variable is registered for callback functions to access
        self._phase = None

    @property
    def global_epoch(self):
        """Global epoch count, always equal to the length of train loss history

        :return: number of training epochs that have been run
        :rtype: int
        """
        return len(self.loss['train'])

    def _auto_enforce(self, net, cond, *points):
        """Enforce condition on network with inputs.
            If self.enforcer is set, use it;
            otherwise, fill cond.enforce() with as many arguments as needed

        :param net: network for parameterized solution
        :type net: torch.nn.Module
        :param cond: condition (a.k.a. parameterization) for the network
        :type cond: `neurodiffeq.conditions.BaseCondition`
        :param points: a tuple of vectors, each with shape = (-1, 1)
        :type points: tuple[torch.Tensor]
        :return: function values at sampled points
        :rtype: torch.Tensor
        """
        if self.enforcer:
            return self.enforcer(net, cond, points)

        n_params = len(signature(cond.enforce).parameters)
        points = points[:n_params - 1]
        return cond.enforce(net, *points)

    def _update_history(self, value, metric_type, key):
        """Append a value to corresponding history list

        :param value: value to be appended
        :type value: float
        :param metric_type: {'loss', 'analytic_mse'}; type of history metrics
        :type metric_type: str
        :param key: {'train', 'valid'}; dict key in self.loss / self.analytic_mse
        :type key: str
        """
        self._phase = key
        if metric_type == 'loss':
            self.loss[key].append(value)
        elif metric_type == 'analytic_mse':
            self.analytic_mse[key].append(value)
        else:
            raise KeyError(f'history type = {metric_type} not understood')

    def _update_train_history(self, value, metric_type):
        """Append a value to corresponding training history list"""
        self._update_history(value, metric_type, key='train')

    def _update_valid_history(self, value, metric_type):
        """Append a value to corresponding validation history list"""
        self._update_history(value, metric_type, key='valid')

    def _generate_batch(self, key):
        """Generate the next batch, register in self._batch_examples and return the batch

        :param key: {'train', 'valid'}; dict key in self._examples / self._batch_examples / self._batch_start
        :type key: str
        """
        # the following side effects are helpful for future extension,
        # especially for additional loss term that depends on the coordinates
        self._phase = key
        self._batch_examples[key] = [v.reshape(-1, 1) for v in self.generator[key].get_examples()]
        return self._batch_examples[key]

    def _generate_train_batch(self):
        """Generate the next training batch, register in self._batch_examples and return"""
        return self._generate_batch('train')

    def _generate_valid_batch(self):
        """Generate the next validation batch, register in self._batch_examples and return"""
        return self._generate_batch('valid')

    def _do_optimizer_step(self):
        r"""Optimization procedures after gradients have been computed. Usually, self.optimizer.step() is sufficient.
            At times, user can overwrite this method to perform gradient clipping, etc. Here is an example:
        >>> import itertools
        >>> class MySolver(SphericalSolver)
        >>>     def _do_optimizer_step(self):
        >>>         nn.utils.clip_grad_norm_(itertools.chain([net.parameters() for net in self.nets]), 1.0, 'inf')
        >>>         self.optimizer.step()
        """
        self.optimizer.step()

    def _run_epoch(self, key):
        """Run an epoch on train/valid points, update history, and perform an optimization step if key=='train'
        Note that the optimization step is only performed after all batches are run
        This method doesn't resample points, which shall be handled in the `.fit()` call.

        :param key: {'train', 'valid'}; phase of the epoch
        :type key: str
        """
        self._phase = key
        epoch_loss = 0.0
        epoch_analytic_mse = 0

        # perform forward pass for all batches: a single graph is created and release in every iteration
        # see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/17
        for batch_id in range(self.n_batches[key]):
            batch = self._generate_batch(key)
            funcs = [
                self._auto_enforce(n, c, *batch) for n, c in zip(self.nets, self.conditions)
            ]

            if self.analytic_solutions is not None:
                funcs_true = self.analytic_solutions(*batch)
                for f_pred, f_true in zip(funcs, funcs_true):
                    epoch_analytic_mse += ((f_pred - f_true) ** 2).mean().item()

            residuals = self.pdes(*funcs, *batch)
            residuals = torch.cat(residuals, dim=1)
            loss = self.criterion(residuals) + self.additional_loss(funcs, key)

            # normalize loss across batches
            loss /= self.n_batches[key]

            # accumulate gradients before the current graph is collected as garbage
            if key == 'train':
                loss.backward()
            epoch_loss += loss.item()

        # calculate mean loss of all batches and register to history
        self._update_history(epoch_loss, 'loss', key)

        # perform optimization step when training
        if key == 'train':
            self._do_optimizer_step()
            self.optimizer.zero_grad()
        # update lowest_loss and best_net when validating
        else:
            self._update_best()

        # calculate mean analytic mse of all batches and register to history
        if self.analytic_solutions is not None:
            epoch_analytic_mse /= self.n_batches[key]
            epoch_analytic_mse /= self.n_funcs
            self._update_history(epoch_analytic_mse, 'analytic_mse', key)

    def run_train_epoch(self):
        """Run a training epoch, update history, and perform gradient descent"""
        self._run_epoch('train')

    def run_valid_epoch(self):
        """Run a validation epoch and update history"""
        self._run_epoch('valid')

    def _update_best(self):
        """Update self.lowest_loss and self.best_nets if current validation loss is lower than self.lowest_loss"""
        current_loss = self.loss['valid'][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.best_nets = deepcopy(self.nets)

    def fit(self, max_epochs, callbacks=None, monitor=None):
        r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.
            This method does not return solution, which is done in the `.get_solution` method.
            If `callbacks` is passed, callbacks are run one at a time, after training, validating, updaing best model and before monitor checking
            A callback function `cb(solver)` can set `solver._stop_training` to True to perform early stopping,

        :param max_epochs: number of epochs to run
        :type max_epochs: int
        :param monitor: DEPRECATED; use a MonitorCallback instance instead; Monitor for visualizing solution and metrics
        :rtype monitor: `neurodiffeq.pde_spherical.MonitorSpherical`
        :param callbacks: a list of callback functions, each accepting the solver instance itself as its only argument
        :rtype callbacks: list[callable]
        """
        self._stop_training = False
        self._max_local_epoch = max_epochs

        if monitor:
            warnings.warn("Monitor is deprecated, use a MonitorCallback instead")

        for local_epoch in range(max_epochs):
            # stops training if self._stop_training is set to True by a callback
            if self._stop_training:
                break

            # register local epoch so it can be accessed by callbacks
            self.local_epoch = local_epoch
            self.run_train_epoch()
            self.run_valid_epoch()

            if callbacks:
                for cb in callbacks:
                    cb(self)

            if monitor:
                if (local_epoch + 1) % monitor.check_every == 0 or local_epoch == max_epochs - 1:
                    monitor.check(
                        self.nets,
                        self.conditions,
                        loss_history=self.loss,
                        analytic_mse_history=self.analytic_mse,
                    )

    def get_solution(self, copy=True, best=True, harmonics_fn=None):
        """Return a solution class

        :param copy: if True, use a deep copy of internal nets and conditions
        :type copy: bool
        :param best: if True, return the solution with lowest loss instead of the solution after the last epoch
        :type best: bool
        :param harmonics_fn: if set, use it as function basis for returned solution
        :type harmonics_fn: callable
        :return: trained solution
        :rtype: `neurodiffeq.pde_spherical.SolutionSpherical`
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        if harmonics_fn:
            return SolutionSphericalHarmonics(nets, conditions, harmonics_fn=harmonics_fn)
        else:
            return SolutionSpherical(nets, conditions)

    def get_internals(self, param_names, return_type='list'):
        """Return internal variable(s) of the solver
        If param_names == 'all', return all internal variables as a dict;
        If param_names is single str, return the corresponding variables
        If param_names is a list and return_type == 'list', return corresponding internal variables as a list
        If param_names is a list and return_type == 'dict', return a dict with keys in param_names

        :param param_names: a parameter name or a list of parameter names
        :type param_names: str or list[str]
        :param return_type: {'list', 'dict'}; ignored if `param_names` is a str
        :type return_type: str
        :return: a single parameter, or a list/dict of parameters as indicated above
        :rtype: list or dict or any
        """
        # get internals according to provided param_names
        available_params = {
            "analytic_mse": self.analytic_mse,
            "analytic_solutions": self.analytic_solutions,
            "n_batches": self.n_batches,
            "best_nets": self.best_nets,
            "criterion": self.criterion,
            "conditions": self.conditions,
            "global_epoch": self.global_epoch,
            "loss": self.loss,
            "lowest_loss": self.lowest_loss,
            "n_funcs": self.n_funcs,
            "nets": self.nets,
            "optimizer": self.optimizer,
            "pdes": self.pdes,
            "r_max": self.r_max,
            "r_min": self.r_min,
            "generator": self.generator,
        }

        if param_names == "all":
            return available_params

        if isinstance(param_names, str):
            return available_params[param_names]

        if return_type == 'list':
            return [available_params[name] for name in param_names]
        elif return_type == "dict":
            return {name: available_params[name] for name in param_names}
        else:
            raise ValueError(f"unrecognized return_type = {return_type}")

    def additional_loss(self, funcs, key):
        r"""Return additional loss; this method is to be overridden by subclasses
            This method can use any of the internal variables: the current batch, the nets, the conditions, etc.

        :param funcs: outputs of the networks after enforced by conditions
        :type funcs: list[torch.Tensor]
        :param key: {'train', 'valid'}; phase of the epoch; used to access the sample batch, etc.
        :type key: str
        :return: additional scalar loss
        :rtype: torch.Tensor
        """
        return 0.0


class MonitorSpherical:
    """A monitor for checking the status of the neural network during training.

    :param r_min: The lower bound of radius, i.e., radius of interior boundary
    :type r_min: float
    :param r_max: The upper bound of radius, i.e., radius of exterior boundary
    :type r_max: float
    :param check_every: The frequency of checking the neural network represented by the number of epochs between two checks, defaults to 100.
    :type check_every: int, optional
    :param var_names: names of dependent variables; if provided, shall be used for plot titles; defaults to None
    :type var_names: list[str]
    :param shape: shape of mesh for visualizing the solution; defaults to (10, 10, 10)
    :type shape: tuple[int]
    :param r_scale: 'linear' or 'log'; controls the grid point in the :math:`r` direction; defaults to 'linear'
    :type r_scale: str
    :param theta_min: The lower bound of polar angle, defaults to :math:`0`
    :type theta_min: float
    :param theta_max: The upper bound of polar angle, defaults to :math:`\\pi`
    :type theta_max: float
    :param phi_min: The lower bound of azimuthal angle, defaults to :math:`0`
    :type phi_min: float
    :param phi_max: The upper bound of azimuthal angle, defaults to :math:`2\\pi`
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
        self.ax_analytic = None
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

    def check(self, nets, conditions, loss_history, analytic_mse_history=None):
        r"""Draw (3n + 2) plots:
             1) For each function u(r, phi, theta), there are 3 axes:
                a) one ax for u-r curves grouped by phi
                b) one ax for u-r curves grouped by theta
                c) one ax for u-theta-phi contour heat map
             2) Additionally, one ax for MSE against analytic solution, another for training and validation loss

        :param nets: The neural networks that approximates the PDE.
        :type nets: list [`torch.nn.Module`]
        :param conditions: The initial/boundary condition of the PDE.
        :type conditions: list [`neurodiffeq.conditions.BaseCondition`]
        :param loss_history: The history of training loss and validation loss. The 'train' entry is a list of training loss and 'valid' entry is a list of validation loss.
        :type loss_history: dict['train': list[float], 'valid': list[float]]
        :param analytic_mse_history: The history of training and validation MSE against analytic solution. The 'train' entry is a list of training analytic MSE and 'valid' entry is a list of validation analytic MSE.
        :type analytic_mse_history: dict['train': list[float], 'valid': list[float]]

        .. note::
            `check` is meant to be called by the function `solve2D`.
        """

        # initialize the figure and axes here so that the Monitor knows the number of dependent variables and
        # shape of the figure, number of the subplots, etc.
        # Draw (3n + 2) plots:
        #     1) For each function u(r, phi, theta), there are 3 axes:
        #         a) one ax for u-r curves grouped by phi
        #         b) one ax for u-r curves grouped by theta
        #         c) one ax for u-theta-phi contour heat map
        #     2) Additionally, one ax for MSE against analytic solution, another for training and validation loss
        n_vars = len(nets) if self.n_vars is None else self.n_vars
        n_row = n_vars + 1
        n_col = 3
        if not self.fig:
            self.fig = plt.figure(figsize=(24, 6 * n_row))
            self.fig.tight_layout()
            self.axs = self.fig.subplots(nrows=n_row, ncols=n_col, gridspec_kw={'width_ratios': [1, 1, 2]})
            for ax in self.axs[n_row - 1]:
                ax.remove()
            self.cbs = [None] * n_vars
            if analytic_mse_history is not None:
                self.ax_analytic = self.fig.add_subplot(n_row, 2, n_row * 2 - 1)
                self.ax_loss = self.fig.add_subplot(n_row, 2, n_row * 2)
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

        if analytic_mse_history is not None:
            self._update_history(self.ax_analytic, analytic_mse_history, y_label='MSE',
                                 title='MSE against analytic solution')
            self._update_history(self.ax_loss, loss_history, y_label='Loss', title='Loss')
        else:
            self._update_history(self.ax_loss, loss_history)

        self.customization()
        self.fig.canvas.draw()
        # for command-line, interactive plots, not pausing can lead to graphs not being displayed at all
        # see https://stackoverflow.com/questions/19105388/python-2-7-mac-osx-interactive-plotting-with-matplotlib-not-working
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
    def _update_history(ax, history, x_label='Epochs', y_label=None, title=None):
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.plot(history['train'], label='train')
        ax.plot(history['valid'], label='valid')
        ax.set_yscale('log')
        ax.legend()

    def new(self):
        self.fig = None
        self.axs = []
        self.cbs = []
        self.ax_analytic = None
        self.ax_loss = None
        return self

    def set_variable_count(self, n):
        r"""Manually set the number of scalar fields to be visualized;
            If not set, defaults to length of `nets` passed to `self.check()` every time `self.check()` is called

        :param n: number of scalar fields to overwrite default
        :type n: int
        :return: self
        """
        self.n_vars = n
        return self

    def unset_variable_count(self):
        r"""Manually unset the number of scalar fields to be visualized;
            Once unset, the number defaults to length of `nets` passed to `self.check()` every time `self.check()` is called

        :return: self
        """
        self.n_vars = None
        return self


class SolutionSphericalHarmonics(SolutionSpherical):
    """
    A solution to a PDE (system) in spherical coordinates

    :param nets: list of networks that takes in radius tensor and outputs the coefficients of spherical harmonics
    :type nets: list[`torch.nn.Module`]
    :param conditions: list of conditions to be enforced on each nets; must be of the same length as nets
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param harmonics_fn: mapping from :math:`\\theta` and :math:`\\phi` to basis functions, e.g., spherical harmonics
    :type harmonics_fn: callable
    :param max_degree: DEPRECATED and SUPERSEDED by harmonics_fn; highest used for the harmonic basis
    :type max_degree: int
    """

    def __init__(self, nets, conditions, max_degree=None, harmonics_fn=None):
        super(SolutionSphericalHarmonics, self).__init__(nets, conditions)
        if (harmonics_fn is None) and (max_degree is None):
            raise ValueError("harmonics_fn should be specified")

        if max_degree is not None:
            warnings.warn("`max_degree` is DEPRECATED; pass `harmonics_fn` instead, which takes precedence")
            self.harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)

        if harmonics_fn is not None:
            self.harmonics_fn = harmonics_fn

    def _compute_u(self, net, condition, rs, thetas, phis):
        products = condition.enforce(net, rs) * self.harmonics_fn(thetas, phis)
        return torch.sum(products, dim=1)


class MonitorSphericalHarmonics(MonitorSpherical):
    """A monitor for checking the status of the neural network during training.

    :param r_min: The lower bound of radius, i.e., radius of interior boundary
    :type r_min: float
    :param r_max: The upper bound of radius, i.e., radius of exterior boundary
    :type r_max: float
    :param check_every: The frequency of checking the neural network represented by the number of epochs between two checks, defaults to 100.
    :type check_every: int, optional
    :param var_names: names of dependent variables; if provided, shall be used for plot titles; defaults to None
    :type var_names: list[str]
    :param shape: shape of mesh for visualizing the solution; defaults to (10, 10, 10)
    :type shape: tuple[int]
    :param r_scale: 'linear' or 'log'; controls the grid point in the :math:`r` direction; defaults to 'linear'
    :type r_scale: str
    :param harmonics_fn: mapping from :math:`\\theta` and :math:`\\phi` to basis functions, e.g., spherical harmonics
    :type harmonics_fn: callable
    :param theta_min: The lower bound of polar angle, defaults to :math:`0`
    :type theta_min: float
    :param theta_max: The upper bound of polar angle, defaults to :math:`\\pi`
    :type theta_max: float
    :param phi_min: The lower bound of azimuthal angle, defaults to :math:`0`
    :type phi_min: float
    :param phi_max: The upper bound of azimuthal angle, defaults to :math:`2\\pi`
    :type phi_max: float
    :param max_degree: DEPRECATED and SUPERSEDED by harmonics_fn; highest used for the harmonic basis
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


# callbacks to be passed to SphericalSolver.fit()


class MonitorCallback:
    """A callback for updating the monitor plots (and optionally saving the fig to disk)

    :param monitor: the underlying monitor responsible for plotting solutions
    :type monitor: MonitorSpherical
    :param fig_dir: directory for saving monitor figs; if not specified, figs will not be saved
    :type fig_dir: str
    :param check_against: which epoch count to check against; either 'local' (default) or 'global'
    :type check_against: str
    :param repaint_last: whether to update the plot on the last local epoch, defaults to True
    :type repaint_last: bool
    """

    def __init__(self, monitor, fig_dir=None, check_against='local', repaint_last=True):
        self.monitor = monitor
        self.fig_dir = fig_dir
        self.repaint_last = repaint_last
        if check_against not in ['local', 'global']:
            raise ValueError(f'unknown check_against type = {check_against}')
        self.check_against = check_against

    def to_repaint(self, solver):
        if self.check_against == 'local':
            epoch_now = solver.local_epoch + 1
        elif self.check_against == 'global':
            epoch_now = solver.global_epoch + 1
        else:
            raise ValueError(f'unknown check_against type = {self.check_against}')

        if epoch_now % self.monitor.check_every == 0:
            return True
        if self.repaint_last and solver.local_epoch == solver._max_local_epoch - 1:
            return True

        return False

    def __call__(self, solver):
        if not self.to_repaint(solver):
            return

        self.monitor.check(
            solver.nets,
            solver.conditions,
            loss_history=solver.loss,
            analytic_mse_history=solver.analytic_solutions
        )
        if self.fig_dir:
            pic_path = os.path.join(self.fig_dir, f"epoch-{solver.global_epoch}.png")
            self.monitor.fig.savefig(pic_path)


class CheckpointCallback:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def __call__(self, solver):
        if solver.local_epoch == solver._max_local_epoch - 1:
            now = datetime.now()
            timestr = now.strftime("%Y-%m-%d_%H-%M-%S")
            fname = os.path.join(self.ckpt_dir, timestr + ".internals")
            with open(fname, 'wb') as f:
                dill.dump(solver.get_internals("all"), f)
                logging.info(f"Saved checkpoint to {fname} at local epoch = {solver.local_epoch} "
                             f"(global epoch = {solver.global_epoch})")


class ReportOnFitCallback:
    def __call__(self, solver):
        if solver.local_epoch == 0:
            logging.info(
                f"Starting from global epoch {solver.global_epoch - 1}, training on {(solver.r_min, solver.r_max)}")
            tb = solver.generator['train'].size
            ntb = solver.n_batches['train']
            t = tb * ntb
            vb = solver.generator['valid'].size
            nvb = solver.n_batches['valid']
            v = vb * nvb
            logging.info(f"train size = {tb} x {ntb} = {t}, valid_size = {vb} x {nvb} = {v}")
