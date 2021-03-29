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
from ._version_utils import warn_deprecate_class, deprecated_alias
from .generators import Generator3D, GeneratorSpherical
from .conditions import NoCondition, DirichletBVPSpherical, InfDirichletBVPSpherical
from .conditions import DirichletBVPSphericalBasis, InfDirichletBVPSphericalBasis
from .solvers import SolverSpherical
from .monitors import MonitorSpherical, MonitorSphericalHarmonics
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

# old solver name is deprecated
SphericalSolver = warn_deprecate_class(SolverSpherical)


def solve_spherical(
        pde, condition, r_min=None, r_max=None,
        net=None, train_generator=None, valid_generator=None, analytic_solution=None,
        optimizer=None, criterion=None, max_epochs=1000,
        monitor=None, return_internal=False, return_best=False, harmonics_fn=None, batch_size=None, shuffle=None,
):
    r"""[**DEPRECATED**, use SphericalSolver class instead]
    Train a neural network to solve one PDE with spherical inputs in 3D space.

    :param pde:
        The PDE to solve.
        If the PDE is :math:`F(u, r,\theta, \phi) = 0`, where :math:`u` is the dependent variable
        and :math:`r`, :math:`\theta` and :math:`\phi` are the independent variables,
        then `pde` should be a function that maps :math:`(u, r, \theta, \phi)` to :math:`F(u, r,\theta, \phi)`.
    :type pde: callable
    :param condition:
        The initial/boundary condition that :math:`u` should satisfy.
    :type condition: `neurodiffeq.conditions.BaseCondition`
    :param r_min:
        Radius for inner boundary; ignored if both generators are provided.
    :type r_min: float, optional
    :param r_max:
        Radius for outer boundary; ignored if both generators are provided.
    :type r_max: float, optional
    :param net:
        The neural network used to approximate the solution.
        Defaults to None.
    :type net: `torch.nn.Module`, optional
    :param train_generator:
        The example generator to generate 3-D training points.
        Default to None.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param valid_generator:
        The example generator to generate 3-D validation points.
        Default to None.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param analytic_solution:
        Analytic solution to the pde system, used for testing purposes.
        It should map (``rs``, ``thetas``, ``phis``) to u.
    :type analytic_solution: callable
    :param optimizer:
        The optimization method to use for training.
        Defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion:
        The loss function to use for training.
        Defaults to None.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param max_epochs:
        The maximum number of epochs to train.
        Defaults to 1000.
    :type max_epochs: int, optional
    :param monitor:
        The monitor to check the status of neural network during training.
        Defaults to None.
    :type monitor: `neurodiffeq.pde_spherical.MonitorSpherical`, optional
    :param return_internal:
        Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function.
        Defaults to False.
    :type return_internal: bool, optional
    :param return_best:
        Whether to return the nets that achieved the lowest validation loss.
        Defaults to False.
    :type return_best: bool, optional
    :param harmonics_fn:
        Function basis (spherical harmonics for example) if solving coefficients of a function basis.
        Used when returning the solution.
    :type harmonics_fn: callable
    :return:
        The solution of the PDE. The history of training loss and validation loss.
        Optionally, MSE against analytic solution, the nets, conditions, training generator,
        validation generator, optimizer and loss function.
        The solution is a function that has the signature `solution(xs, ys, as_type)`.
    :rtype:
        tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict]
        or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict]
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool

    .. note::
        This function is deprecated, use a ``neurodiffeq.solvers.SphericalSolver`` instead
    """

    warnings.warn("solve_spherical is deprecated, consider using SphericalSolver instead", FutureWarning)
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
        nets=None, train_generator=None, valid_generator=None, analytic_solutions=None,
        optimizer=None, criterion=None, max_epochs=1000, monitor=None, return_internal=False,
        return_best=False, harmonics_fn=None, batch_size=None, shuffle=None,
):
    r"""[**DEPRECATED**, use SphericalSolver class instead]
    Train a neural network to solve a PDE system with spherical inputs in 3D space

    :param pde_system:
        The PDEs ystem to solve.
        If the PDE is :math:`F_i(u_1, u_2, ..., u_n, r,\theta, \phi) = 0`
        where :math:`u_i` is the i-th dependent variable
        and :math:`r`, :math:`\theta` and :math:`\phi` are the independent variables,
        then `pde_system` should be a function that maps :math:`(u_1, u_2, ..., u_n, r, \theta, \phi)`
        to a list where the i-th entry is :math:`F_i(u_1, u_2, ..., u_n, r, \theta, \phi)`.
    :type pde_system: callable
    :param conditions:
        The initial/boundary conditions.
        The ith entry of the conditions is the condition that :math:`u_i` should satisfy.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param r_min:
        Radius for inner boundary.
        Ignored if both generators are provided.
    :type r_min: float, optional
    :param r_max:
        Radius for outer boundary.
        Ignored if both generators are provided.
    :type r_max: float, optional
    :param nets:
        The neural networks used to approximate the solution.
        Defaults to None.
    :type nets: list[`torch.nn.Module`], optional
    :param train_generator:
        The example generator to generate 3-D training points.
        Default to None.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param valid_generator:
        The example generator to generate 3-D validation points.
        Default to None.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param analytic_solutions:
        Analytic solution to the pde system, used for testing purposes.
        It should map (rs, thetas, phis) to a list of [u_1, u_2, ..., u_n].
    :type analytic_solutions: callable
    :param optimizer:
        The optimization method to use for training.
        Defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion:
        The loss function to use for training.
        Defaults to None.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param max_epochs:
        The maximum number of epochs to train.
        Defaults to 1000.
    :type max_epochs: int, optional
    :param monitor:
        The monitor to check the status of neural network during training.
        Defaults to None.
    :type monitor: `neurodiffeq.pde_spherical.MonitorSpherical`, optional
    :param return_internal:
        Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function.
        Defaults to False.
    :type return_internal: bool, optional
    :param return_best:
        Whether to return the nets that achieved the lowest validation loss.
        Defaults to False.
    :type return_best: bool, optional
    :param harmonics_fn:
        Function basis (spherical harmonics for example) if solving coefficients of a function basis.
        Used when returning solution.
    :type harmonics_fn: callable
    :return:
        The solution of the PDE. The history of training loss and validation loss.
        Optionally, MSE against analytic solutions, the nets, conditions,
        training generator, validation generator, optimizer and loss function.
        The solution is a function that has the signature `solution(xs, ys, as_type)`.
    :rtype:
        tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict]
        or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict]
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify n_batches_train and n_batches_valid instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool


    .. note::
        This function is deprecated, use a ``neurodiffeq.solvers.SphericalSolver`` instead
    """
    warnings.warn("solve_spherical_system is deprecated, consider using SphericalSolver instead", FutureWarning)

    if harmonics_fn is None:
        def enforcer(net, cond, points):
            return cond.enforce(net, *points)
    else:
        def enforcer(net, cond, points):
            return (cond.enforce(net, points[0]) * harmonics_fn(*points[1:])).sum(dim=1, keepdims=True)

    solver = SolverSpherical(
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
    ret = (solution, solver.metrics_history)
    if return_internal:
        params = ['nets', 'conditions', 'train_generator', 'valid_generator', 'optimizer', 'criterion']
        internals = solver.get_internals(params, return_type="dict")
        ret = ret + (internals,)
    return ret

