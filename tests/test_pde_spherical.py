import torch
import torch.nn as nn
import numpy as np
from pytest import raises
from neurodiffeq import diff
from neurodiffeq.pde_spherical import ExampleGeneratorSpherical, ExampleGenerator3D
from neurodiffeq.pde_spherical import NoConditionSpherical, DirichletBVPSpherical, InfDirichletBVPSpherical
from neurodiffeq.pde_spherical import solve_spherical, solve_spherical_system
from neurodiffeq.pde_spherical import SolutionSpherical
from neurodiffeq.pde_spherical import MonitorSpherical


def laplacian_spherical(u, r, theta, phi):
    """a helper function that computes the Laplacian in spherical coordinates
    """
    r_component = diff(u * r, r, order=2) / r
    theta_component = diff(torch.sin(theta) * diff(u, theta), theta) / (r ** 2 * torch.sin(theta))
    phi_component = diff(u, phi, order=2) / (r ** 2 * torch.sin(theta) ** 2)
    return r_component + theta_component + phi_component


def test_dirichlet_bvp_spherical():
    # B.C. for the interior boundary (r_min)
    interior = nn.Linear(in_features=2, out_features=1, bias=True)
    f = lambda theta, phi: interior(torch.cat([theta, phi], dim=1))

    # B.C. for the exterior boundary (r_max)
    exterior = nn.Linear(in_features=2, out_features=1, bias=True)
    g = lambda theta, phi: exterior(torch.cat([theta, phi], dim=1))

    bvp = DirichletBVPSpherical(r_0=0., f=f, r_1=1.0, g=g)

    net = nn.Linear(in_features=3, out_features=1, bias=True)
    theta = torch.rand(10, 1) * np.pi
    phi = torch.rand(10, 1) * 2 * np.pi

    r = torch.zeros_like(theta)
    v0 = f(theta, phi).detach().numpy()
    u0 = bvp.enforce(net, r, theta, phi).detach().numpy()
    assert np.isclose(v0, u0, atol=1.e-5).all(), f"Unmatched boundary {v0} != {u0}"

    r = torch.ones_like(theta)
    v1 = g(theta, phi).detach().numpy()
    u1 = bvp.enforce(net, r, theta, phi).detach().numpy()
    assert np.isclose(v1, u1, atol=1.e-5).all(), f"Unmatched boundary {v1} != {u1}"

    print("DirichletBVPSpherical test passed")


def test_inf_dirichlet_bvp_spherical():
    # B.C. for the interior boundary (r_min)
    interior = nn.Linear(in_features=2, out_features=1, bias=True)
    f = lambda theta, phi: interior(torch.cat([theta, phi], dim=1))

    # B.C. for the exterior boundary (r_max)
    exterior = nn.Linear(in_features=2, out_features=1, bias=True)
    g = lambda theta, phi: exterior(torch.cat([theta, phi], dim=1))

    inf_bvp = InfDirichletBVPSpherical(r_0=0., f=f, g=g, order=1)

    net = nn.Linear(in_features=3, out_features=1, bias=True)
    theta = torch.rand(10, 1) * np.pi
    phi = torch.rand(10, 1) * (2 * np.pi)

    r = torch.zeros_like(theta)
    v0 = f(theta, phi).detach().numpy()
    u0 = inf_bvp.enforce(net, r, theta, phi).detach().numpy()
    assert np.isclose(v0, u0, atol=1.e-5).all(), f"Unmatched boundary {v0} != {u0}"

    r = torch.ones_like(theta) * 1e10  # using the real inf results in error because (inf * 0) returns nan in torch
    v_inf = g(theta, phi).detach().numpy()
    u_inf = inf_bvp.enforce(net, r, theta, phi).detach().numpy()
    assert np.isclose(v_inf, u_inf, atol=1.e-5).all(), f"Unmatched boundary {v_inf} != {u_inf}"

    print("InfDirichletBVPSpherical test passed")


def test_train_generator_spherical():
    pde = laplacian_spherical
    condition = NoConditionSpherical()
    train_generator = ExampleGeneratorSpherical(size=64, r_min=0., r_max=1., method='equally-spaced-noisy')
    r, th, ph = train_generator.get_examples()
    assert (0. < r.min()) and (r.max() < 1.)
    assert (0. <= th.min()) and (th.max() <= np.pi)
    assert (0. <= ph.min()) and (ph.max() <= 2 * np.pi)

    valid_generator = ExampleGeneratorSpherical(size=64, r_min=1., r_max=1., method='equally-radius-noisy')
    r, th, ph = valid_generator.get_examples()
    assert (r == 1).all()
    assert (0. <= th.min()) and (th.max() <= np.pi)
    assert (0. <= ph.min()) and (ph.max() <= 2 * np.pi)

    solve_spherical(pde, condition, 0.0, 1.0,
                    train_generator=train_generator,
                    valid_generator=valid_generator,
                    max_epochs=5)
    with raises(ValueError):
        _ = ExampleGeneratorSpherical(64, method='bad_generator')

    with raises(ValueError):
        _ = ExampleGeneratorSpherical(64, r_min=-1.0)

    with raises(ValueError):
        _ = ExampleGeneratorSpherical(64, r_min=1.0, r_max=0.0)

    print("GeneratorSpherical tests passed")


def test_solve_spherical():
    pde = laplacian_spherical
    generator = ExampleGeneratorSpherical(512)

    # 0-boundary condition; solution should be u(r, theta, phi) = 0 identically
    f = lambda th, ph: 0.
    g = lambda th, ph: 0.
    condition = DirichletBVPSpherical(r_0=0., f=f, r_1=1., g=g)
    solution, loss_history = solve_spherical(pde, condition, 0.0, 1.0, max_epochs=500, return_best=True)
    rs, thetas, phis = generator.get_examples()
    us = solution(rs, thetas, phis, as_type='np')
    assert np.isclose(us, np.zeros_like(us), atol=0.005).all(), f"Solution is not straight 0s: {us}"

    # 1-boundary condition; solution should be u(r, theta, phi) = 1 identically
    f = lambda th, ph: 1.
    g = lambda th, ph: 1.
    condition = DirichletBVPSpherical(r_0=0., f=f, r_1=1., g=g)
    solution, loss_history = solve_spherical(pde, condition, 0.0, 1.0, max_epochs=500, return_best=True)
    rs, thetas, phis = generator.get_examples()
    us = solution(rs, thetas, phis, as_type='np')
    assert np.isclose(us, np.ones_like(us), atol=0.005).all(), f"Solution is not straight 1s: {us}"

    print("solve_spherical tests passed")


def test_monitor_spherical():
    pde = laplacian_spherical

    f = lambda th, ph: 0.
    g = lambda th, ph: 0.
    condition = DirichletBVPSpherical(r_0=0., f=f, r_1=1., g=g)
    monitor = MonitorSpherical(0.0, 1.0, check_every=1)
    solve_spherical(pde, condition, 0.0, 1.0, max_epochs=50, monitor=monitor)

    print("MonitorSpherical test passed")


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

    solution, loss_history = solve_spherical_system(pde_system, conditions, 0.0, 1.0, max_epochs=500, return_best=True)
    generator = ExampleGeneratorSpherical(512, r_min=0., r_max=1.)
    rs, thetas, phis = generator.get_examples()
    us, vs = solution(rs, thetas, phis, as_type='np')

    assert np.isclose(us, np.zeros(512), atol=0.005).all(), f"Solution u is not straight 0s: {us}"
    assert np.isclose(vs, np.ones(512), atol=0.005).all(), f"Solution v is not straight 1s: {vs}"

    print("solve_spherical_system tests passed")


def test_electric_potential_uniformly_charged_ball():
    """
    electric potential on uniformly charged solid sphere
    refer to http://www.phys.uri.edu/~gerhard/PHY204/tsl94.pdf
    """
    # free charge volume density
    rho = 1.
    # medium permittivity
    epsilon = 1.
    # Coulomb constant
    k = 1. / (4 * np.pi * epsilon)
    # radius of the ball
    R = 1.
    # total electric charge on solid sphere
    Q = (4 / 3) * np.pi * (R ** 3) * rho
    # electric potential at sphere center
    v_0 = 1.5 * k * Q / R
    # electric potential on sphere boundary
    v_R = k * Q / R
    # analytic solution of electric potential
    analytic_solution = lambda r, th, ph: k * Q / (2 * R) * (3 - (r / R) ** 2)

    pde = lambda u, r, theta, phi: laplacian_spherical(u, r, theta, phi) + rho / epsilon
    condition = DirichletBVPSpherical(r_0=0., f=lambda th, ph: v_0, r_1=R, g=lambda th, ph: v_R)
    monitor = MonitorSpherical(0.0, R, check_every=50)

    solution, loss_history, analytic_mse = solve_spherical(pde, condition, 0., R, max_epochs=500, return_best=True,
                                                           analytic_solution=analytic_solution, monitor=monitor)
    generator = ExampleGeneratorSpherical(512)
    rs, thetas, phis = generator.get_examples()
    us = solution(rs, thetas, phis, as_type="np")
    vs = analytic_solution(rs, thetas, phis).detach().numpy()
    abs_diff = abs(us - vs)

    assert np.isclose(us, vs, atol=0.008).all(), \
        f"Solution doesn't match analytic expectation {us} != {vs}, abs_diff={abs_diff}"

    print("electric-potential-on-uniformly-charged-solid-sphere passed")
