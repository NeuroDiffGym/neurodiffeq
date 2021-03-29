import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use('Agg')  # use a non-GUI backend, so plots are not shown during testing
from math import erf, sqrt
import pytest
from neurodiffeq.neurodiffeq import safe_diff as diff
from neurodiffeq.generators import GeneratorSpherical, Generator3D
from neurodiffeq.conditions import NoCondition
from neurodiffeq.conditions import DirichletBVPSpherical
from neurodiffeq.conditions import InfDirichletBVPSpherical
from neurodiffeq.conditions import DirichletBVPSphericalBasis
from neurodiffeq.pde_spherical import solve_spherical, solve_spherical_system
from neurodiffeq.pde_spherical import MonitorSpherical
from neurodiffeq.pde_spherical import MonitorSphericalHarmonics
from neurodiffeq.solvers import SolutionSpherical, SolutionSphericalHarmonics
from neurodiffeq.function_basis import RealSphericalHarmonics, HarmonicsLaplacian
from neurodiffeq.networks import FCNN


@pytest.fixture(autouse=True)
def magic():
    MAGIC = 42
    torch.manual_seed(MAGIC)
    np.random.seed(MAGIC)


def laplacian_spherical(u, r, theta, phi):
    """a helper function that computes the Laplacian in spherical coordinates
    """
    r_component = diff(u * r, r, order=2) / r
    theta_component = diff(torch.sin(theta) * diff(u, theta), theta) / (r ** 2 * torch.sin(theta))
    phi_component = diff(u, phi, order=2) / (r ** 2 * torch.sin(theta) ** 2)
    return r_component + theta_component + phi_component


def test_solve_spherical():
    pde = laplacian_spherical
    generator = GeneratorSpherical(512)

    # 0-boundary condition; solution should be u(r, theta, phi) = 0 identically
    f = lambda th, ph: 0.
    g = lambda th, ph: 0.
    condition = DirichletBVPSpherical(r_0=0., f=f, r_1=1., g=g)
    with pytest.warns(FutureWarning):
        solution, loss_history = solve_spherical(pde, condition, 0.0, 1.0, max_epochs=2, return_best=True)
    assert isinstance(solution, SolutionSpherical)


def test_monitor_spherical():
    f = lambda th, ph: 0.
    g = lambda th, ph: 0.
    conditions = [DirichletBVPSpherical(r_0=0., f=f, r_1=1., g=g)]
    nets = [FCNN(3, 1)]
    monitor = MonitorSpherical(0.0, 1.0, check_every=1)
    loss_history = {
        'train': list(np.random.rand(10)),
        'valid': list(np.random.rand(10)),
    }
    analytic_mse_history = {
        'train': list(np.random.rand(10)),
        'valid': list(np.random.rand(10)),
    }
    history = {
        'train_loss': list(np.random.rand(10)),
        'valid_loss': list(np.random.rand(10)),
        'train_foo': list(np.random.rand(10)),
        'valid_foo': list(np.random.rand(10)),
        'train_bar': list(np.random.rand(10)),
        'valid_bar': list(np.random.rand(10)),
    }
    with pytest.warns(FutureWarning):
        monitor.check(nets, conditions, history=history, analytic_mse_history=analytic_mse_history)
    with pytest.warns(FutureWarning):
        monitor.check(nets, conditions, history=loss_history)
    with pytest.warns(FutureWarning):
        monitor.check(nets, conditions, history=loss_history, analytic_mse_history=analytic_mse_history)
    with pytest.raises(ValueError):
        monitor.check(nets, conditions, history={'train_foo': [], 'valid_foo': []})
    monitor.check(nets, conditions, history=history)


def test_solve_spherical_system():
    # a PDE system that can be decoupled into 2 Laplacian equations :math:`\\nabla^2 u = 0` and :math:`\\nabla^2 v = 0`
    pde_system = lambda u, v, r, theta, phi: [
        laplacian_spherical(u, r, theta, phi) + laplacian_spherical(v, r, theta, phi),
        laplacian_spherical(u, r, theta, phi) - laplacian_spherical(v, r, theta, phi),
    ]

    # constant boundary conditions for u and v; solution should be u = 0 identically and v = 1 identically
    conditions = [
        DirichletBVPSpherical(r_0=0., f=lambda phi, theta: 0., r_1=1., g=lambda phi, theta: 0.),
        DirichletBVPSpherical(r_0=0., f=lambda phi, theta: 1., r_1=1., g=lambda phi, theta: 1.),
    ]

    with pytest.warns(FutureWarning):
        solution, _ = solve_spherical_system(pde_system, conditions, 0.0, 1.0, max_epochs=2, return_best=True)
    assert isinstance(solution, SolutionSpherical)


def test_electric_potential_gaussian_charged_density():
    # total charge
    Q = 1.
    # standard deviation of gaussian
    sigma = 1.
    # medium permittivity
    epsilon = 1.
    # Coulomb constant
    k = 1 / (4 * np.pi * epsilon)
    # coefficient of gaussian term
    gaussian_coeff = Q / (sigma ** 3) / np.power(2 * np.pi, 1.5)
    # distribution of charge
    rho_f = lambda r: gaussian_coeff * torch.exp(- r.pow(2) / (2 * sigma ** 2))
    # analytic solution, refer to https://en.wikipedia.org/wiki/Poisson%27s_equation
    analytic_solution = lambda r, th, ph: (k * Q / r) * torch.erf(r / (np.sqrt(2) * sigma))

    # interior and exterior radius
    r_0, r_1 = 0.1, 3.
    # values at interior and exterior boundary
    v_0 = (k * Q / r_0) * erf(r_0 / (np.sqrt(2) * sigma))
    v_1 = (k * Q / r_1) * erf(r_1 / (np.sqrt(2) * sigma))

    def validate(solution):
        generator = GeneratorSpherical(512, r_min=r_0, r_max=r_1)
        rs, thetas, phis = generator.get_examples()
        us = solution(rs, thetas, phis, to_numpy=True)
        vs = analytic_solution(rs, thetas, phis).detach().cpu().numpy()
        assert us.shape == vs.shape

    # solving the problem using normal network (subject to the influence of polar singularity of laplacian operator)

    pde1 = lambda u, r, th, ph: laplacian_spherical(u, r, th, ph) + rho_f(r) / epsilon
    condition1 = DirichletBVPSpherical(r_0, lambda th, ph: v_0, r_1, lambda th, ph: v_1)
    monitor1 = MonitorSpherical(r_0, r_1, check_every=50)
    with pytest.warns(FutureWarning):
        solution1, metrics_history = solve_spherical(
            pde1, condition1, r_0, r_1,
            max_epochs=2,
            return_best=True,
            analytic_solution=analytic_solution,
            monitor=monitor1,
        )
    validate(solution1)

    # solving the problem using spherical harmonics (laplcian computation is optimized)
    max_degree = 2
    harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)
    harmonic_laplacian = HarmonicsLaplacian(max_degree=max_degree)
    pde2 = lambda R, r, th, ph: harmonic_laplacian(R, r, th, ph) + rho_f(r) / epsilon
    R_0 = torch.tensor([v_0 * 2] + [0 for _ in range((max_degree + 1) ** 2 - 1)])
    R_1 = torch.tensor([v_1 * 2] + [0 for _ in range((max_degree + 1) ** 2 - 1)])

    def analytic_solution2(r, th, ph):
        sol = torch.zeros(r.shape[0], (max_degree + 1) ** 2)
        sol[:, 0:1] = 2 * analytic_solution(r, th, ph)
        return sol

    condition2 = DirichletBVPSphericalBasis(r_0=r_0, R_0=R_0, r_1=r_1, R_1=R_1, max_degree=max_degree)
    monitor2 = MonitorSphericalHarmonics(r_0, r_1, check_every=50, harmonics_fn=harmonics_fn)
    net2 = FCNN(n_input_units=1, n_output_units=(max_degree + 1) ** 2)
    with pytest.warns(FutureWarning):
        solution2, metrics_history = solve_spherical(
            pde2, condition2, r_0, r_1,
            net=net2,
            max_epochs=2,
            return_best=True,
            analytic_solution=analytic_solution2,
            monitor=monitor2,
            harmonics_fn=harmonics_fn,
        )
    validate(solution2)


def test_spherical_harmonics():
    # number of training points
    N_SAMPLES = 100
    # highest degree for spherical harmonics
    MAX_DEGREE = 4
    # expected output shape
    OUTPUT_SHAPE = (N_SAMPLES, (MAX_DEGREE + 1) ** 2)

    # real spherical harmonics written in cartesian coordinates (radius is assumed to be 1)
    # l = 0
    Y0_0 = lambda x, y, z: torch.ones_like(x)
    # l = 1
    Y1n1 = lambda x, y, z: y
    Y1_0 = lambda x, y, z: z
    Y1p1 = lambda x, y, z: x
    # l = 2
    Y2n2 = lambda x, y, z: x * y
    Y2n1 = lambda x, y, z: y * z
    Y2_0 = lambda x, y, z: -x ** 2 - y ** 2 + 2 * z ** 2
    Y2p1 = lambda x, y, z: z * x
    Y2p2 = lambda x, y, z: x ** 2 - y ** 2
    # l = 3
    Y3n3 = lambda x, y, z: (3 * x ** 2 - y ** 2) * y
    Y3n2 = lambda x, y, z: x * y * z
    Y3n1 = lambda x, y, z: y * (4 * z ** 2 - x ** 2 - y ** 2)
    Y3_0 = lambda x, y, z: z * (2 * z ** 2 - 3 * x ** 2 - 3 * y ** 2)
    Y3p1 = lambda x, y, z: x * (4 * z ** 2 - x ** 2 - y ** 2)
    Y3p2 = lambda x, y, z: (x ** 2 - y ** 2) * z
    Y3p3 = lambda x, y, z: (x ** 2 - 3 * y ** 2) * x
    # l = 4
    Y4n4 = lambda x, y, z: x * y * (x ** 2 - y ** 2)
    Y4n3 = lambda x, y, z: (3 * x ** 2 - y ** 2) * y * z
    Y4n2 = lambda x, y, z: x * y * (7 * z ** 2 - 1)
    Y4n1 = lambda x, y, z: y * z * (7 * z ** 2 - 3)
    Y4_0 = lambda x, y, z: 35 * z ** 4 - 30 * z ** 2 + 3
    Y4p1 = lambda x, y, z: x * z * (7 * z ** 2 - 3)
    Y4p2 = lambda x, y, z: (x ** 2 - y ** 2) * (7 * z ** 2 - 1)
    Y4p3 = lambda x, y, z: (x ** 2 - 3 * y ** 2) * x * z
    Y4p4 = lambda x, y, z: x ** 2 * (x ** 2 - 3 * y ** 2) - y ** 2 * (3 * x ** 2 - y ** 2)

    harmonics_fn_cartesian = [
        Y0_0,
        Y1n1, Y1_0, Y1p1,
        Y2n2, Y2n1, Y2_0, Y2p1, Y2p2,
        Y3n3, Y3n2, Y3n1, Y3_0, Y3p1, Y3p2, Y3p3,
        Y4n4, Y4n3, Y4n2, Y4n1, Y4_0, Y4p1, Y4p2, Y4p3, Y4p4,
    ]

    # coefficients of spherical harmonics components
    # l = 0
    C0_0 = 1 / 2
    # l = 1
    C1n1 = sqrt(3 / 4)
    C1_0 = sqrt(3 / 4)
    C1p1 = sqrt(3 / 4)
    # l = 2
    C2n2 = sqrt(15) / 2
    C2n1 = sqrt(15) / 2
    C2_0 = sqrt(5) / 4
    C2p1 = sqrt(15) / 2
    C2p2 = sqrt(15) / 4
    # l = 3
    C3n3 = sqrt(35 / 2) / 4
    C3n2 = sqrt(105) / 2
    C3n1 = sqrt(21 / 2) / 4
    C3_0 = sqrt(7) / 4
    C3p1 = sqrt(21 / 2) / 4
    C3p2 = sqrt(105) / 4
    C3p3 = sqrt(35 / 2) / 4
    # l = 4
    C4n4 = sqrt(35) * 3 / 4
    C4n3 = sqrt(35 / 2) * 3 / 4
    C4n2 = sqrt(5) * 3 / 4
    C4n1 = sqrt(5 / 2) * 3 / 4
    C4_0 = 3 / 16
    C4p1 = sqrt(5 / 2) * 3 / 4
    C4p2 = sqrt(5) * 3 / 8
    C4p3 = sqrt(35 / 2) * 3 / 4
    C4p4 = sqrt(35) * 3 / 16

    normalizer = [
        C0_0,
        C1n1, C1_0, C1p1,
        C2n2, C2n1, C2_0, C2p1, C2p2,
        C3n3, C3n2, C3n1, C3_0, C3p1, C3p2, C3p3,
        C4n4, C4n3, C4n2, C4n1, C4_0, C4p1, C4p2, C4p3, C4p4,
    ]

    harmonics_fn_cartesian = harmonics_fn_cartesian[:(MAX_DEGREE + 1) ** 2]
    normalizer = normalizer[:(MAX_DEGREE + 1) ** 2]
    harmonics_fn = RealSphericalHarmonics(max_degree=MAX_DEGREE)
    theta = torch.rand(N_SAMPLES, 1) * np.pi
    phi = torch.rand(N_SAMPLES, 1) * 2 * np.pi
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    harmonics = harmonics_fn(theta, phi)
    # test the shape of output
    assert harmonics.shape == OUTPUT_SHAPE, f"got shape={harmonics.shape}; expected shape={OUTPUT_SHAPE}"
    harmonics_cartesian = torch.cat(
        [f(x, y, z) * c for f, c in zip(harmonics_fn_cartesian, normalizer)],
        dim=1
    )
    abs_diff = abs(harmonics - harmonics_cartesian)

    # test the correctness of spherical harmonics written in terms of theta and phi
    assert torch.max(abs_diff) <= 1e-5, f"difference too large, check again:\n {abs_diff.max()}"


def test_spherical_laplacian():
    N_FLOAT = np.float64
    T_FLOAT = torch.float64
    n_samples = 10
    r_value = np.random.rand(n_samples, 1).astype(N_FLOAT)
    theta_value = np.random.rand(n_samples, 1).astype(N_FLOAT)
    phi_value = np.random.rand(n_samples, 1).astype(N_FLOAT)
    r_net = FCNN(n_input_units=1, n_output_units=25).to(T_FLOAT)

    # compute laplacians using spherical harmonics property
    r1 = torch.tensor(r_value, dtype=T_FLOAT, requires_grad=True)
    theta1 = torch.tensor(theta_value, dtype=T_FLOAT, requires_grad=True)
    phi1 = torch.tensor(phi_value, dtype=T_FLOAT, requires_grad=True)
    R1 = r_net(r1)
    harmonics_laplacian = HarmonicsLaplacian(max_degree=4)
    lap1 = harmonics_laplacian(R1, r1, theta1, phi1)

    # compute laplacians using brute force
    r2 = torch.tensor(r_value, dtype=T_FLOAT, requires_grad=True)
    theta2 = torch.tensor(theta_value, dtype=T_FLOAT, requires_grad=True)
    phi2 = torch.tensor(phi_value, dtype=T_FLOAT, requires_grad=True)
    R2 = r_net(r2)
    spherical_fn = RealSphericalHarmonics(max_degree=4)
    harmonics = spherical_fn(theta2, phi2)
    u = torch.sum(R2 * harmonics, dim=1, keepdim=True)
    lap2 = laplacian_spherical(u, r2, theta2, phi2)

    lap1 = lap1.detach().cpu().numpy()
    lap2 = lap2.detach().cpu().numpy()
    assert np.isclose(lap2, lap1).all(), \
        f'Laplcians computed using spherical harmonics trick differ from brute force solution, {lap1} != {lap2}'
