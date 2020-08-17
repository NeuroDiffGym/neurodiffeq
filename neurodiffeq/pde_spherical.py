import sys
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .spherical_harmonics import RealSphericalHarmonics

from .networks import FCNN
from .neurodiffeq import diff
from copy import deepcopy


def _nn_output_spherical_input(net, rs, thetas, phis):
    points = torch.cat((rs, thetas, phis), 1)
    return net(points)


class BaseBVPSpherical:
    def enforce(self, net, r, theta, phi):
        raise NotImplementedError(f"Abstract {self.__class__.__name__} cannot be enforced")


class NoConditionSpherical(BaseBVPSpherical):
    def __init__(self):
        pass

    def enforce(self, net, r, theta, phi):
        return _nn_output_spherical_input(net, r, theta, phi)


class ExampleGenerator3D:
    """An example generator for generating 3-D training points. NOT TO BE CONFUSED with `ExampleGeneratorSpherical`
        :param grid: The discretization of the 3 dimensions, if we want to generate points on a :math:`m \\times n \\times k` grid, then `grid` is `(m, n, k)`, defaults to `(10, 10, 10)`.
        :type grid: tuple[int, int, int], optional
        :param xyz_min: The lower bound of 3 dimensions, if we only care about :math:`x \\geq x_0`, :math:`y \\geq y_0`, and :math:`z \\geq z_0` then `xyz_min` is `(x_0, y_0, z_0)`, defaults to `(0.0, 0.0, 0.0)`.
        :type xyz_min: tuple[float, float, float], optional
        :param xyz_max: The upper bound of 3 dimensions, if we only care about :math:`x \\leq x_1`, :math:`y \\leq y_1`, and :math:`z \\leq z_1` then `xyz_max` is `(x_1, y_1, z_1)`, defaults to `(1.0, 1.0, 1.0)`.
        :type xyz_max: tuple[float, float, float], optional
        :param method: The distribution of the 3-D points generated. If set to 'equally-spaced', the points will be fixed to the grid specified. If set to 'equally-spaced-noisy', a normal noise will be added to the previously mentioned set of points, defaults to 'equally-spaced-noisy'.
        :type method: str, optional
        :raises ValueError: When provided with an unknown method.
    """

    def __init__(self, grid=(10, 10, 10), xyz_min=(0.0, 0.0, 0.0), xyz_max=(1.0, 1.0, 1.0),
                 method='equally-spaced-noisy'):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 2-D training points. It will be called by the function `solve2D`.
        """
        self.size = grid[0] * grid[1] * grid[2]

        x = torch.linspace(xyz_min[0], xyz_max[0], grid[0], requires_grad=True)
        y = torch.linspace(xyz_min[1], xyz_max[1], grid[1], requires_grad=True)
        z = torch.linspace(xyz_min[2], xyz_max[2], grid[2], requires_grad=True)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        self.grid_x, self.grid_y, self.grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

        def trunc(tensor, min, max):
            tensor[tensor < min] = min
            tensor[tensor > max] = max

        if method == 'equally-spaced':
            self.get_examples = lambda: (self.grid_x, self.grid_y, self.grid_z)
        elif method == 'equally-spaced-noisy':
            self.noise_xmean = torch.zeros(self.size)
            self.noise_ymean = torch.zeros(self.size)
            self.noise_zmean = torch.zeros(self.size)
            self.noise_xstd = torch.ones(self.size) * ((xyz_max[0] - xyz_min[0]) / grid[0]) / 4.0
            self.noise_ystd = torch.ones(self.size) * ((xyz_max[1] - xyz_min[1]) / grid[1]) / 4.0
            self.noise_zstd = torch.ones(self.size) * ((xyz_max[2] - xyz_min[2]) / grid[2]) / 4.0
            self.get_examples = lambda: (
                trunc(self.grid_x + torch.normal(mean=self.noise_xmean, std=self.noise_xstd), xyz_min[0], xyz_max[0]),
                trunc(self.grid_y + torch.normal(mean=self.noise_ymean, std=self.noise_ystd), xyz_min[1], xyz_max[1]),
                trunc(self.grid_z + torch.normal(mean=self.noise_zmean, std=self.noise_zstd), xyz_min[2], xyz_max[2]),
            )
        else:
            raise ValueError(f'Unknown method: {method}')


class ExampleGeneratorSpherical:
    """An example generator for generating points in spherical coordinates. NOT TO BE CONFUSED with `ExampleGenerator3D`
    :param size: number of points in 3-D sphere
    :type size: int
    :param r_min: radius of the interior boundary
    :type r_min: float, optional
    :param r_max: radius of the exterior boundary
    :type r_max: float, optional
    :param method: The distribution of the 3-D points generated. If set to 'equally-radius-noisy', radius of the points will be drawn from a uniform distribution :math:`r \\sim U[r_{min}, r_{max}]`. If set to 'equally-spaced-noisy', squared radius of the points will be drawn from a uniform distribution :math:`r^2 \\sim U[r_{min}^2, r_{max}^2]`
    :type method: str, optional
    """

    def __init__(self, size, r_min=0., r_max=1., method='equally-spaced-noisy'):
        if r_min < 0 or r_max < r_min:
            raise ValueError(f"Illegal range [f{r_min}, {r_max}]")

        if method == 'equally-spaced-noisy':
            lower = r_min ** 2
            upper = r_max ** 2
            rng = upper - lower
            self.get_r = lambda: torch.sqrt(rng * torch.rand(self.shape) + lower)
        elif method == "equally-radius-noisy":
            lower = r_min
            upper = r_max
            rng = upper - lower
            self.get_r = lambda: rng * torch.rand(self.shape) + lower
        else:
            raise ValueError(f'Unknown method: {method}')

        self.size = size  # stored for `solve_spherical_system` to access
        self.shape = (size,)  # used for `self.get_example()`

    def get_examples(self):
        a = torch.rand(self.shape)
        b = torch.rand(self.shape)
        c = torch.rand(self.shape)
        denom = a + b + c
        # `x`, `y`, `z` here are just for computation of `theta` and `phi`
        epsilon = 1e-6
        x = torch.sqrt(a / denom) + epsilon
        y = torch.sqrt(b / denom) + epsilon
        z = torch.sqrt(c / denom) + epsilon
        # `sign_x`, `sign_y`, `sign_z` are either -1 or +1
        sign_x = torch.randint(0, 2, self.shape, dtype=x.dtype) * 2 - 1
        sign_y = torch.randint(0, 2, self.shape, dtype=y.dtype) * 2 - 1
        sign_z = torch.randint(0, 2, self.shape, dtype=z.dtype) * 2 - 1

        x = x * sign_x
        y = y * sign_y
        z = z * sign_z

        theta = torch.acos(z).requires_grad_(True)
        phi = -torch.atan2(y, x) + np.pi  # atan2 ranges (-pi, pi] instead of [0, 2pi)
        phi.requires_grad_(True)
        r = self.get_r().requires_grad_(True)

        return r, theta, phi


class EnsembleExampleGenerator:
    r"""
    An ensemble generator for sampling points, whose `get_example` returns all the samples of its sub-generators
    :param \*generators: a sequence of sub-generators, must have a .size field and a .get_examples() method
    """

    def __init__(self, *generators):
        self.generators = generators
        self.size = sum(gen.size for gen in generators)

    def get_examples(self):
        all_examples = [gen.get_examples() for gen in self.generators]
        # zip(*sequence) is just `unzip`ping a sequence into sub-sequences, refer to this post for more
        # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip
        segmented = zip(*all_examples)
        return [torch.cat(seg) for seg in segmented]


class DirichletBVPSpherical(BaseBVPSpherical):
    """Dirichlet boundary condition for the interior and exterior boundary of the sphere, where the interior boundary is not necessarily a point
        We are solving :math:`u(t)` given :math:`u(r, \\theta, \\phi)\\bigg|_{r = r_0} = f(\\theta, \\phi)` and :math:`u(r, \\theta, \\phi)\\bigg|_{r = r_1} = g(\\theta, \\phi)`

    :param r_0: The radius of the interior boundary. When r_0 = 0, the interior boundary is collapsed to a single point (center of the ball)
    :type r_0: float
    :param f: The value of :math:u on the interior boundary. :math:`u(r, \\theta, \\phi)\\bigg|_{r = r_0} = f(\\theta, \\phi)`.
    :type f: function
    :param r_1: The radius of the exterior boundary; if set to None, `g` must also be None
    :type r_1: float or None
    :param g: The value of :math:u on the exterior boundary. :math:`u(r, \\theta, \\phi)\\bigg|_{r = r_1} = g(\\theta, \\phi)`. If set to None, `r_1` must also be set to None
    :type g: function or None
    """

    def __init__(self, r_0, f, r_1=None, g=None):
        """Initializer method
        """
        if (r_1 is None) ^ (g is None):
            raise ValueError(f'r_1 and g must be both/neither set to None; got r_1={r_1}, g={g}')
        self.r_0, self.r_1 = r_0, r_1
        self.f, self.g = f, g

    def enforce(self, net, r, theta, phi):
        r"""Enforce the output of a neural network to satisfy the boundary condition.

        :param net: The neural network that approximates the ODE.
        :type net: `torch.nn.Module`
        :param r: The radii of points where the neural network output is evaluated.
        :type r: `torch.tensor`
        :param theta: The latitudes of points where the neural network output is evaluated. `theta` ranges [0, pi]
        :type theta: `torch.tensor`
        :param phi: The longitudes of points where the neural network output is evaluated. `phi` ranges [0, 2*pi)
        :type phi: `torch.tensor`
        :return: The modified output which now satisfies the boundary condition.
        :rtype: `torch.tensor`


        .. note::
            `enforce` is meant to be called by the function `solve_spherical` and `solve_spherical_system`.
        """
        u = _nn_output_spherical_input(net, r, theta, phi)
        if self.r_1 is None:
            return (1 - torch.exp(-r + self.r_0)) * u + self.f(theta, phi)
        else:
            r_tilde = (r - self.r_0) / (self.r_1 - self.r_0)
            # noinspection PyTypeChecker
            return self.f(theta, phi) * (1 - r_tilde) + \
                   self.g(theta, phi) * r_tilde + \
                   (1. - torch.exp((1 - r_tilde) * r_tilde)) * u


class InfDirichletBVPSpherical(BaseBVPSpherical):
    """Similar to `DirichletBVPSpherical`; only difference is we are considering :math:`g(\\theta, \\phi)` as :math:`r_1 \\to \\infty`, so `r_1` doesn't need to be specified
        We are solving :math:`u(t)` given :math:`u(r, \\theta, \\phi)\\bigg|_{r = r_0} = f(\\theta, \\phi)` and :math:`\\lim_{r \\to \\infty} u(r, \\theta, \\phi) = g(\\theta, \\phi)`

    :param r_0: The radius of the interior boundary. When r_0 = 0, the interior boundary is collapsed to a single point (center of the ball)
    :type r_0: float
    :param f: The value of :math:u on the interior boundary. :math:`u(r, \\theta, \\phi)\\bigg|_{r = r_0} = f(\\theta, \\phi)`.
    :type f: function
    :param g: The value of :math:u on the exterior boundary. :math:`u(r, \\theta, \\phi)\\bigg|_{r = r_1} = g(\\theta, \\phi)`.
    :type g: function
    :param order: The smallest :math:k that guarantees :math:`\\lim_{r \\to +\\infty} u(r, \\theta, \\phi) e^{-k r} = 0`, defaults to 1
    :type order: int or float, optional
    """

    def __init__(self, r_0, f, g, order=1):
        self.r_0 = r_0
        self.f = f
        self.g = g
        self.order = order

    def enforce(self, net, r, theta, phi):
        r"""Enforce the output of a neural network to satisfy the boundary condition.

        :param net: The neural network that approximates the PDE.
        :type net: `torch.nn.Module`
        :param r: The radii of points where the neural network output is evaluated.
        :type r: `torch.tensor`
        :param theta: The latitudes of points where the neural network output is evaluated. `theta` ranges [0, pi]
        :type theta: `torch.tensor`
        :param phi: The longitudes of points where the neural network output is evaluated. `phi` ranges [0, 2*pi)
        :type phi: `torch.tensor`
        :return: The modified output which now satisfies the boundary condition.
        :rtype: `torch.tensor`


        .. note::
            `enforce` is meant to be called by the function `solve_spherical` and `solve_spherical_system`.
        """
        u = _nn_output_spherical_input(net, r, theta, phi)
        dr = r - self.r_0
        return self.f(theta, phi) * torch.exp(-self.order * dr) + \
               self.g(theta, phi) * torch.tanh(dr) + \
               torch.exp(-self.order * dr) * torch.tanh(dr) * u


class SolutionSpherical:
    """A solution to a PDE (system) in spherical coordinates

    :param nets: The neural networks that approximate the PDE.
    :type nets: list[`torch.nn.Module`]
    :param conditions: The conditions of the PDE (system).
    :type conditions: list[`neurodiffeq.pde_spherical.BaseBVPSpherical`]
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
        :type rs: `torch.tensor`
        :param thetas: The latitudes of points where the neural network output is evaluated. `theta` ranges [0, pi]
        :type thetas: `torch.tensor`
        :param phis: The longitudes of points where the neural network output is evaluated. `phi` ranges [0, 2*pi)
        :type phis: `torch.tensor`
        :param as_type: Whether the returned value is a `torch.tensor` ('tf') or `numpy.array` ('np').
        :type as_type: str
        :return: dependent variables are evaluated at given points.
        :rtype: list[`torch.tensor` or `numpy.array` (when there is more than one dependent variables)
            `torch.tensor` or `numpy.array` (when there is only one dependent variable)
        """
        if not isinstance(rs, torch.Tensor):
            rs = torch.tensor(rs, dtype=torch.float32)
        if not isinstance(thetas, torch.Tensor):
            thetas = torch.tensor(thetas, dtype=torch.float32)
        if not isinstance(phis, torch.Tensor):
            phis = torch.tensor(phis, dtype=torch.float32)
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
        pde, condition, r_min, r_max,
        net=None, train_generator=None, shuffle=True, valid_generator=None, analytic_solution=None,
        optimizer=None, criterion=None, batch_size=16, max_epochs=1000,
        monitor=None, return_internal=False, return_best=False
):
    """Train a neural network to solve one PDE with spherical inputs in 3D space

        :param pde: The PDE to solve. If the PDE is :math:`F(u, r,\\theta, \\phi) = 0` where :math:`u` is the dependent variable and :math:`r`, :math:`\\theta` and :math:`\\phi` are the independent variables,
            then `pde` should be a function that maps :math:`(u, r, \\theta, \\phi)` to :math:`F(u, r,\\theta, \\phi)`
        :type pde: function
        :param condition: The initial/boundary condition that :math:`u` should satisfy.
        :type condition: `neurodiffeq.pde_spherical.BaseBVPSpherical` or `neurodiffeq.pde_spherical.BaseBVPSphericalHarmonics`
        :param r_min: The lower bound of radius, if we only care about :math:`r \\geq r_0` , then `r_min` is `r_0`.
        :type r_min: float
        :param r_max: The upper bound of radius, if we only care about :math:`r \\leq r_1` , then `r_max` is `r_1`.
        :type r_max: float
        :param net: The neural network used to approximate the solution, defaults to None.
        :type net: `torch.nn.Module`, optional
        :param train_generator: The example generator to generate 3-D training points, default to None.
        :type train_generator: `neurodiffeq.pde_spherical.ExampleGeneratorSpherical`, optional
        :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
        :type shuffle: bool, optional
        :param valid_generator: The example generator to generate 3-D validation points, default to None.
        :type valid_generator: `neurodiffeq.pde_spherical.ExampleGeneratorSpherical`, optional
        :param analytic_solution: analytic solution to the pde system, used for testing purposes; should map (rs, thetas, phis) to u
        :type analytic_solution: function
        :param optimizer: The optimization method to use for training, defaults to None.
        :type optimizer: `torch.optim.Optimizer`, optional
        :param criterion: The loss function to use for training, defaults to None.
        :type criterion: `torch.nn.modules.loss._Loss`, optional
        :param batch_size: The shape of the mini-batch to use, defaults to 16.
        :type batch_size: int, optional
        :param max_epochs: The maximum number of epochs to train, defaults to 1000.
        :type max_epochs: int, optional
        :param monitor: The monitor to check the status of neural network during training, defaults to None.
        :type monitor: `neurodiffeq.pde_spherical.MonitorSpherical`, optional
        :param return_internal: Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function, defaults to False.
        :type return_internal: bool, optional
        :param return_best: Whether to return the nets that achieved the lowest validation loss, defaults to False.
        :type return_best: bool, optional
        :return: The solution of the PDE. The history of training loss and validation loss.
            Optionally, MSE against analytic solution, the nets, conditions, training generator, validation generator, optimizer and loss function.
            The solution is a function that has the signature `solution(xs, ys, as_type)`.
        :rtype: tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict]; or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict]; or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict, dict]
        """

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
    )


def solve_spherical_system(
        pde_system, conditions, r_min, r_max,
        nets=None, train_generator=None, shuffle=True, valid_generator=None, analytic_solutions=None,
        optimizer=None, criterion=None, batch_size=16,
        max_epochs=1000, monitor=None, return_internal=False, return_best=False
):
    """Train a neural network to solve a PDE system with spherical inputs in 3D space

        :param pde_system: The PDEsystem to solve. If the PDE is :math:`F_i(u_1, u_2, ..., u_n, r,\\theta, \\phi) = 0` where :math:`u_i` is the i-th dependent variable and :math:`r`, :math:`\\theta` and :math:`\\phi` are the independent variables,
            then `pde_system` should be a function that maps :math:`(u_1, u_2, ..., u_n, r, \\theta, \\phi)` to a list where the i-th entry is :math:`F_i(u_1, u_2, ..., u_n, r, \\theta, \\phi)`.
        :type pde_system: function
        :param conditions: The initial/boundary conditions. The ith entry of the conditions is the condition that :math:`u_i` should satisfy.
        :type conditions: list[`neurodiffeq.pde_spherical.BaseBVPSpherical`] or list[`neurodiffeq.pde_spherical.BaseBVPSphericalHarmonics`]
        :param r_min: The lower bound of radius, if we only care about :math:`r \\geq r_0` , then `r_min` is `r_0`.
        :type r_min: float
        :param r_max: The upper bound of radius, if we only care about :math:`r \\leq r_1` , then `r_max` is `r_1`.
        :type r_max: float
        :param nets: The neural networks used to approximate the solution, defaults to None.
        :type nets: list[`torch.nn.Module`], optionalnerate 3-D training points, default to None.
        :type train_generator: `neurodiffeq.pde_spherical.E
        :param train_generator: The example generator to gexampleGeneratorSpherical`, optional
        :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
        :type shuffle: bool, optional
        :param valid_generator: The example generator to generate 3-D validation points, default to None.
        :type valid_generator: `neurodiffeq.pde_spherical.ExampleGeneratorSpherical`, optional
        :param analytic_solutions: analytic solution to the pde system, used for testing purposes; should map (rs, thetas, phis) to a list of [u_1, u_2, ..., u_n]
        :type analytic_solutions: function
        :param optimizer: The optimization method to use for training, defaults to None.
        :type optimizer: `torch.optim.Optimizer`, optional
        :param criterion: The loss function to use for training, defaults to None.
        :type criterion: `torch.nn.modules.loss._Loss`, optional
        :param batch_size: The shape of the mini-batch to use, defaults to 16.
        :type batch_size: int, optional
        :param max_epochs: The maximum number of epochs to train, defaults to 1000.
        :type max_epochs: int, optional
        :param monitor: The monitor to check the status of neural network during training, defaults to None.
        :type monitor: `neurodiffeq.pde_spherical.MonitorSpherical`, optional
        :param return_internal: Whether to return the nets, conditions, training generator, validation generator, optimizer and loss function, defaults to False.
        :type return_internal: bool, optional
        :param return_best: Whether to return the nets that achieved the lowest validation loss, defaults to False.
        :type return_best: bool, optional
        :return: The solution of the PDE. The history of training loss and validation loss.
            Optionally, MSE against analytic solutions, the nets, conditions, training generator, validation generator, optimizer and loss function.
            The solution is a function that has the signature `solution(xs, ys, as_type)`.
        :rtype: tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict]; or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict]; or tuple[`neurodiffeq.pde_spherical.SolutionSpherical`, dict, dict, dict]
        """
    # default values
    n_dependent_vars = len(conditions)
    if not nets:
        nets = [
            FCNN(n_input_units=3, n_hidden_units=32, n_hidden_layers=1, actv=nn.Tanh)
            for _ in range(n_dependent_vars)
        ]
    if not train_generator:
        train_generator = ExampleGeneratorSpherical(512, r_min, r_max, method='equally-spaced-noisy')
    if not valid_generator:
        valid_generator = ExampleGeneratorSpherical(512, r_min, r_max, method='equally-spaced-noisy')
    if not optimizer:
        all_parameters = []
        for net in nets:
            all_parameters += list(net.parameters())
        optimizer = optim.Adam(all_parameters, lr=0.001)
    if not criterion:
        criterion = nn.MSELoss()

    if return_internal:
        internal = {
            'nets': nets,
            'conditions': conditions,
            'train_generator': train_generator,
            'valid_generator': valid_generator,
            'optimizer': optimizer,
            'criterion': criterion
        }

    n_examples_train = train_generator.size
    n_examples_valid = valid_generator.size
    # R.H.S. for the PDE system
    train_zeros = torch.zeros(batch_size).reshape((-1, 1))
    valid_zeros = torch.zeros(batch_size).reshape((-1, 1))

    loss_history = {'train': [], 'valid': []}
    analytic_mse = {'train': [], 'valid': []} if analytic_solutions else None
    mse_fn = nn.MSELoss()
    valid_loss_epoch_min = np.inf
    solution_min = None

    for epoch in range(max_epochs):
        train_loss_epoch = 0.0
        train_analytic_loss_epoch = 0.0

        train_examples_r, train_examples_theta, train_examples_phi = train_generator.get_examples()
        train_examples_r = train_examples_r.reshape((-1, 1))
        train_examples_theta = train_examples_theta.reshape((-1, 1))
        train_examples_phi = train_examples_phi.reshape((-1, 1))
        idx = np.random.permutation(n_examples_train) if shuffle else np.arange(n_examples_train)
        batch_start, batch_end = 0, batch_size

        while batch_start < n_examples_train:
            if batch_end > n_examples_train:
                batch_end = n_examples_train
            batch_idx = idx[batch_start:batch_end]
            rs = train_examples_r[batch_idx]
            thetas = train_examples_theta[batch_idx]
            phis = train_examples_phi[batch_idx]

            # the dependent variables
            us = [
                _auto_enforce(con, net, rs, thetas, phis)
                for con, net in zip(conditions, nets)
            ]

            if analytic_solutions:
                vs = analytic_solutions(rs, thetas, phis)
                with torch.no_grad():
                    train_analytic_loss_epoch += \
                        mse_fn(torch.stack(us), torch.stack(vs)).item() * (batch_end - batch_start)

            Fs = pde_system(*us, rs, thetas, phis)
            loss = 0.0
            for F in Fs:
                if F.shape[0] < train_zeros.shape[0]:
                    print("WARNING: batch size doesn't divide training size, which could lead to unstable behaviour",
                          file=sys.stderr)
                    loss += criterion(F, torch.zeros_like(F)) * F.shape[0] / train_zeros.shape[0]
                else:
                    loss += criterion(F, train_zeros)  # type: torch.Tensor
            train_loss_epoch += loss.item() * (batch_end - batch_start)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_start += batch_size
            batch_end += batch_size

        loss_history['train'].append(train_loss_epoch / n_examples_train)
        if analytic_solutions:
            analytic_mse['train'].append(train_analytic_loss_epoch / n_examples_train)

        # calculate the validation loss
        valid_analytic_loss_epoch = 0.0
        valid_examples_r, valid_examples_theta, valid_examples_phi = valid_generator.get_examples()
        valid_examples_r = valid_examples_r.reshape(-1, 1)
        valid_examples_theta = valid_examples_theta.reshape(-1, 1)
        valid_examples_phi = valid_examples_phi.reshape(-1, 1)
        idx = np.random.permutation(n_examples_valid) if shuffle else np.arange(n_examples_valid)
        batch_start, batch_end = 0, batch_size

        valid_loss_epoch = 0.0
        while batch_start < n_examples_valid:
            if batch_end > n_examples_valid:
                batch_end = n_examples_valid
            batch_idx = idx[batch_start:batch_end]
            rs = valid_examples_r[batch_idx]
            thetas = valid_examples_theta[batch_idx]
            phis = valid_examples_phi[batch_idx]
            us = [_auto_enforce(con, net, rs, thetas, phis) for con, net in zip(conditions, nets)]
            if analytic_solutions:
                vs = analytic_solutions(rs, thetas, phis)
                with torch.no_grad():
                    valid_analytic_loss_epoch += \
                        mse_fn(torch.stack(us), torch.stack(vs)).item() * (batch_end - batch_start)

            Fs = pde_system(*us, rs, thetas, phis)
            for F in Fs:
                valid_loss_epoch += criterion(F, valid_zeros).item() * (batch_end - batch_start)
            batch_start += batch_size
            batch_end += batch_size

        loss_history['valid'].append(valid_loss_epoch / n_examples_valid)
        if analytic_solutions:
            analytic_mse['valid'].append(valid_analytic_loss_epoch / n_examples_valid)

        if monitor and (epoch % monitor.check_every == 0 or epoch == max_epochs - 1):  # update plots on finish
            monitor.check(nets, conditions, loss_history, analytic_mse_history=analytic_mse)

        if return_best and valid_loss_epoch < valid_loss_epoch_min:
            valid_loss_epoch_min = valid_loss_epoch
            solution_min = get_solution(nets, conditions)

    if return_best:
        solution = solution_min
    else:
        solution = get_solution(nets, conditions)

    ret = (solution, loss_history)
    if analytic_solutions is not None:
        ret = ret + (analytic_mse,)
    if return_internal:
        ret = ret + (internal,)
    return ret


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
    """

    def __init__(self, r_min, r_max, check_every=100, var_names=None, shape=(10, 10, 10)):
        """Initializer method
        """
        self.using_non_gui_backend = matplotlib.get_backend() is 'agg'
        self.check_every = check_every
        self.fig = None
        self.axs = []  # subplots
        self.cbs = []  # color bars
        self.names = var_names
        self.shape = shape
        # input for neural network
        gen = ExampleGenerator3D(
            grid=shape,
            xyz_min=(r_min, 0., 0.),
            xyz_max=(r_max, np.pi, 2 * np.pi),
            method='equally-spaced'
        )
        rs, thetas, phis = gen.get_examples()

        self.r_tensor = rs.reshape(-1, 1)
        self.theta_tensor = thetas.reshape(-1, 1)
        self.phi_tensor = phis.reshape(-1, 1)

        self.r_label = rs.reshape(-1).detach().cpu().numpy()
        self.theta_label = thetas.reshape(-1).detach().cpu().numpy()
        self.phi_label = phis.reshape(-1).detach().cpu().numpy()

    def _compute_u(self, net, condition):
        return condition.enforce(net, self.r_tensor, self.theta_tensor, self.phi_tensor)

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
        :type conditions: list [`neurodiffeq.pde_spherical.BaseBVPSpherical`]
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
        n_axs = len(nets) * 3 + 2
        n_row = len(nets) + 1
        n_col = 3
        if not self.fig:
            self.fig = plt.figure(figsize=(20, 6 * n_row))
            for i in range(n_axs):
                self.axs.append(self.fig.add_subplot(n_row, n_col, i + 1))
            for i in range(len(nets)):
                self.cbs.append(None)

        us = [
            self._compute_u(net, cond).detach().cpu().numpy()
            # cond.enforce(net, self.rs, self.thetas, self.phis).detach().cpu().numpy()
            for net, cond in zip(nets, conditions)
        ]

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

            # ax for u-r curve grouped by phi
            ax = self.axs[3 * i]
            ax.clear()
            sns.lineplot(x='$r$', y='u', hue='$\\phi$', data=df, ax=ax)
            ax.set_title(f'{var_name}($r$) grouped by $\\phi$')
            ax.set_ylabel(var_name)

            # ax for u-r curve grouped by theta
            ax = self.axs[3 * i + 1]
            ax.clear()
            sns.lineplot(x='$r$', y='u', hue='$\\theta$', data=df, ax=ax)
            ax.set_title(f'{var_name}($r$) grouped by $\\theta$')
            ax.set_ylabel(var_name)

            # u-theta-phi heat map
            ax = self.axs[3 * i + 2]
            ax.clear()
            ax.set_xlabel('$\\phi$')
            ax.set_ylabel('$\\theta$')
            ax.set_title(f'{var_name} averaged across $r$')
            cax = ax.matshow(u_across_r, cmap='magma', interpolation='nearest')
            if self.cbs[i]:
                self.cbs[i].remove()
            self.cbs[i] = self.fig.colorbar(cax, ax=ax)

        self.axs[-2].clear()
        self.axs[-2].set_title('MSE against analytic solution')
        self.axs[-2].set_ylabel('MSE')
        self.axs[-2].set_xlabel('epochs')
        if analytic_mse_history:
            self.axs[-2].plot(analytic_mse_history['train'], label='training')
            self.axs[-2].plot(analytic_mse_history['valid'], label='validation')
            self.axs[-2].set_yscale('log')
            self.axs[-2].legend()

        self.axs[-1].clear()
        self.axs[-1].plot(loss_history['train'], label='training loss')
        self.axs[-1].plot(loss_history['valid'], label='validation loss')
        self.axs[-1].set_title('loss during training')
        self.axs[-1].set_ylabel('loss')
        self.axs[-1].set_xlabel('epochs')
        self.axs[-1].set_yscale('log')
        self.axs[-1].legend()

        self.fig.canvas.draw()
        # for command-line, interactive plots, not pausing can lead to graphs not being displayed at all
        # see https://stackoverflow.com/questions/19105388/python-2-7-mac-osx-interactive-plotting-with-matplotlib-not-working
        if not self.using_non_gui_backend:
            plt.pause(0.05)

    def new(self):
        self.fig = None
        self.axs = []
        self.cbs = []
        return self


class BaseBVPSphericalHarmonics:
    """
    :param max_degree: highest degree for spherical harmonics
    :type max_degree: int
    """

    def __init__(self, max_degree=4):
        self.max_degree = max_degree

    def enforce(self, net, r):
        raise NotImplementedError(f'Abstract BVP {self.__class__.__name__} cannot be enforced')


def _coefficients_at_radius(net, r):
    # return net(torch.stack([r], dim=1))
    return net(r)


class NoConditionSphericalHarmonics(BaseBVPSphericalHarmonics):
    def enforce(self, net, r):
        return _coefficients_at_radius(net, r)


class DirichletBVPSphericalHarmonics(BaseBVPSphericalHarmonics):
    """Similar to `DirichletBVPSpherical`; only difference is this condition is enforced on a neural net that takes in :math:r and returns the spherical harmonic coefficients R(r)
        i.e., we constrain the coefficients :math:`R(r)` of spherical harmonics instead of the inner product :math:`R(r) \\cdot Y(\\theta, \\phi)`
        We are solving :math:`R(r)` given :math:`R(r)\\bigg|_{r = r_0} = R_0` and :math:`R(r)\\bigg|_{r = r_1} = R_1`.

    :param r_0: The radius of the interior boundary. When r_0 = 0, the interior boundary is collapsed to a single point (center of the ball)
    :type r_0: float
    :param R_0: The value of harmonic coefficients :math:R on the interior boundary. :math:`R(r)\\bigg|_{r = r_0} = R_0`.
    :type R_0: torch.tensor
    :param r_1: The radius of the exterior boundary; if set to None, `R_1` must also be None
    :type r_1: float or None
    :param R_1: The value of harmonic coefficients :math:R on the exterior bounadry. :math:`R(r)\\bigg|_{r = r_1} = R_1`.
    :type R_1: torch.tensor
    :param max_degree: highest degree for spherical harmonics
    :type max_degree: int
    """

    def __init__(self, r_0, R_0, r_1=None, R_1=None, max_degree=4):
        """Initializer method
        """
        super(DirichletBVPSphericalHarmonics, self).__init__(max_degree=max_degree)
        if (r_1 is None) ^ (R_1 is None):
            raise ValueError(f'r_1 and R_1 must be both/neither set to None; got r_1={r_1}, R_1={R_1}')
        self.r_0, self.r_1 = r_0, r_1
        self.R_0, self.R_1 = R_0, R_1

    def enforce(self, net, r):
        r"""Enforce the output of a neural network to satisfy the boundary condition.

        :param net: The neural network that approximates the coefficients for spherical harmonics.
        :type net: `torch.nn.Module`
        :param r: The radii of points where the neural network output is evaluated.
        :type r: `torch.tensor`
        :return: The modified output which now satisfies the boundary condition.
        :rtype: `torch.tensor`


        .. note::
            `enforce` is meant to be called by the function `solve_spherical` and `solve_spherical_system`.
        """
        R_raw = _coefficients_at_radius(net, r)
        if self.r_1 is None:
            # noinspection PyTypeChecker
            ret = (1 - torch.exp(-r + self.r_0)) * R_raw + self.R_0
        else:
            r_tilde = (r - self.r_0) / (self.r_1 - self.r_0)
            # noinspection PyTypeChecker
            ret = self.R_0 * (1 - r_tilde) + self.R_1 * r_tilde + (1. - torch.exp((1 - r_tilde) * r_tilde)) * R_raw
        return ret


class InfDirichletBVPSphericalHarmonics(BaseBVPSphericalHarmonics):
    """Similar to `InfDirichletBVPSpherical`; only difference is this condition is enforced on a neural net that takes in :math:r and returns the spherical harmonic coefficients R(r)
        i.e., we constrain the coefficients :math:`R(r)` of spherical harmonics instead of the inner product :math:`R(r) \\cdot Y(\\theta, \\phi)`
        We are solving :math:`R(r)` given :math:`R(r)\\bigg|_{r = r_0} = R_0` and :math:`\\lim_{r \\to \\infty} R(r) = R_\\infty`

    :param r_0: The radius of the interior boundary. When r_0 = 0, the interior boundary is collapsed to a single point (center of the ball)
    :type r_0: float
    :param R_0: The value of harmonic coefficients :math:R on the interior boundary. :math:`R(r)\\bigg|_{r = r_0} = R_0`.
    :type R_0: torch.tensor
    :param R_inf: The value of harmonic coefficients :math:R at infinity. :math:`\\lim_{r \\to \\infty} R(r) = R_\\infty`.
    :type R_inf: torch.tensor
    :param order: The smallest :math:k that guarantees :math:`\\lim_{r \\to +\\infty} R(r) e^{-k r} = \\bf 0`, defaults to 1
    :type order: int or float, optional
    :param max_degree: highest degree for spherical harmonics
    :type max_degree: int
    """

    def __init__(self, r_0, R_0, R_inf, order=1, max_degree=4):
        super(InfDirichletBVPSphericalHarmonics, self).__init__(max_degree=max_degree)
        self.r_0 = r_0
        self.R_0 = R_0
        self.R_inf = R_inf
        self.order = order

    def enforce(self, net, r):
        r"""Enforce the output of a neural network to satisfy the boundary condition.

        :param net: The neural network that approximates the coefficients for the spherical harmonics.
        :type net: `torch.nn.Module`
        :param r: The radii of points where the neural network output is evaluated.
        :type r: `torch.tensor`
        :return: The modified output which now satisfies the boundary condition.
        :rtype: `torch.tensor`

        .. note::
            `enforce` is meant to be called by the function `solve_spherical` and `solve_spherical_system`.
        """
        R_raw = _coefficients_at_radius(net, r)
        dr = r - self.r_0
        return self.R_0 * torch.exp(-self.order * dr) + \
               self.R_inf * torch.tanh(dr) + \
               torch.exp(-self.order * dr) * torch.tanh(dr) * R_raw


class SolutionSphericalHarmonics(SolutionSpherical):
    """
    A solution to a PDE (system) in spherical coordinates

    :param nets: list of networks that takes in radius tensor and outputs the coefficients of spherical harmonics
    :type nets: list[`torch.nn.Module`]
    :param conditions: list of conditions to be enforced on each nets; must be of the same length as nets
    :type conditions: list[BaseBVPSphericalHarmonics]
    :param max_degree: max_degree for spherical harmonics; defaults to 4
    :type max_degree: int
    """

    def __init__(self, nets, conditions, max_degree=4):
        super(SolutionSphericalHarmonics, self).__init__(nets, conditions)
        self.max_degree = max_degree
        self.harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)

    def _compute_u(self, net, condition, rs, thetas, phis):
        products = condition.enforce(net, rs) * self.harmonics_fn(thetas, phis)
        return torch.sum(products, dim=1)


class SolutionCylindricalFourier(SolutionSpherical):
    def __init__(self, nets, conditions, max_degree=4):
        from .cylindrical_fourier_series import RealFourierSeries
        super(SolutionCylindricalFourier, self).__init__(nets, conditions)
        self.max_degree = max_degree
        self.harmonics_fn = RealFourierSeries(max_degree=max_degree)

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
    :param max_degree: highest degree for spherical harmonics; defaults to None
    :type var_names: list[str]
    :param shape: shape of mesh for visualizing the solution; defaults to (10, 10, 10)
    :type shape: tuple[int]
    """

    def __init__(self, r_min, r_max, check_every=100, var_names=None, shape=(10, 10, 10), max_degree=4):
        super(MonitorSphericalHarmonics, self).__init__(
            r_min,
            r_max,
            check_every=check_every,
            var_names=var_names,
            shape=shape
        )

        self.max_degree = max_degree
        self.harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)

    def _compute_u(self, net, condition):
        products = condition.enforce(net, self.r_tensor) * self.harmonics_fn(self.theta_tensor, self.phi_tensor)
        return torch.sum(products, dim=1)


def _auto_enforce(cond, net, r, theta, phi):
    """
    This function automatically decides to return either of the two
        1. cond.enforce(net, r, theta, phi) for BaseBVPSpherical
        1. cond.enforce(net, r) for BaseBVPSphericalHarmonics
    """
    if isinstance(cond, BaseBVPSpherical):
        return cond.enforce(net, r, theta, phi)
    elif isinstance(cond, BaseBVPSphericalHarmonics):
        return cond.enforce(net, r)
    else:
        return cond.enforce(net, r)
        # raise TypeError(f'{cond} of class {cond.__class__.__name__} cannot be enforced')


def get_solution(nets, conditions):
    """
    automatically choose between SolutionSpherical and SolutionSphericalHarmonics based on class of conditions
    :param nets: list of networks that either return the output or coefficients of spherical harmonics
    :type nets: list[`torch.nn.Module`]
    :param conditions: list of conditions that are compatible with the output of networks
    :type conditions: list[`neurodiffeq.pde_spherical.BaseBVPSpherical`] or list[`neurodiffeq.pde_spherical.BaseBVPSphericalHarmonics`]
    :return: appropriate solution class with `nets` and `conditions`
    :rtype: `neurodiffeq.pde_spherical.SolutionSpherical`
    """
    if isinstance(conditions[0], BaseBVPSpherical):
        return SolutionSpherical(nets, conditions)
    elif isinstance(conditions[0], BaseBVPSphericalHarmonics):
        max_degree = conditions[0].max_degree
        return SolutionSphericalHarmonics(nets, conditions, max_degree=max_degree)
    else:
        max_degree = conditions[0].max_degree
        return SolutionCylindricalFourier(nets, conditions, max_degree=max_degree)
