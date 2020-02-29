import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use('Agg')  # use a non-GUI backend, so plots are not shown during testing
from math import erf
from pytest import raises
from neurodiffeq import diff
from neurodiffeq.pde_spherical import ExampleGeneratorSpherical, ExampleGenerator3D
from neurodiffeq.pde_spherical import NoConditionSpherical, DirichletBVPSpherical, InfDirichletBVPSpherical
from neurodiffeq.pde_spherical import solve_spherical, solve_spherical_system
from neurodiffeq.pde_spherical import SolutionSpherical
from neurodiffeq.pde_spherical import MonitorSpherical
from neurodiffeq.spherical_harmonics import RealSphericalHarmonics
from neurodiffeq.networks import SphericalHarmonicsNN

import torch
import torch.nn as nn

torch.manual_seed(43)
np.random.seed(43)


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
    v0 = f(theta, phi).detach().cpu().numpy()
    u0 = bvp.enforce(net, r, theta, phi).detach().cpu().numpy()
    assert np.isclose(v0, u0, atol=1.e-5).all(), f"Unmatched boundary {v0} != {u0}"

    r = torch.ones_like(theta)
    v1 = g(theta, phi).detach().cpu().numpy()
    u1 = bvp.enforce(net, r, theta, phi).detach().cpu().numpy()
    assert np.isclose(v1, u1, atol=1.e-5).all(), f"Unmatched boundary {v1} != {u1}"

    bvp_half = DirichletBVPSpherical(r_0=2., f=f)

    r = torch.ones_like(theta) * 2.
    v2 = f(theta, phi).detach().cpu().numpy()
    u2 = bvp_half.enforce(net, r, theta, phi).detach().cpu().numpy()
    assert np.isclose(v2, u2, atol=1.e-5).all(), f"Unmatched boundary {v2} != {u2}"

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
    v0 = f(theta, phi).detach().cpu().numpy()
    u0 = inf_bvp.enforce(net, r, theta, phi).detach().cpu().numpy()
    assert np.isclose(v0, u0, atol=1.e-5).all(), f"Unmatched boundary {v0} != {u0}"

    r = torch.ones_like(theta) * 1e10  # using the real inf results in error because (inf * 0) returns nan in torch
    v_inf = g(theta, phi).detach().cpu().numpy()
    u_inf = inf_bvp.enforce(net, r, theta, phi).detach().cpu().numpy()
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
    vs = analytic_solution(rs, thetas, phis).detach().cpu().numpy()
    abs_diff = abs(us - vs)

    assert np.isclose(us, vs, atol=0.008).all(), \
        f"Solution doesn't match analytic expectation {us} != {vs}, abs_diff={abs_diff}"

    print("electric-potential-on-uniformly-charged-solid-sphere passed")


def test_electric_potential_gaussian_charged_density():
    def subtest(net=None, max_epoch=500):
        print(f'subtest: network = {net}, max_epoch={max_epoch}')
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

        pde = lambda u, r, th, ph: laplacian_spherical(u, r, th, ph) + rho_f(r) / epsilon
        r_0, r_1 = 0.1, 3.
        v_0 = (k * Q / r_0) * erf(r_0 / (np.sqrt(2) * sigma))
        v_1 = (k * Q / r_1) * erf(r_1 / (np.sqrt(2) * sigma))
        condition = DirichletBVPSpherical(r_0, lambda th, ph: v_0, r_1, lambda th, ph: v_1)
        monitor = MonitorSpherical(r_0, r_1, check_every=50)

        solution, loss_history, analytic_mse = solve_spherical(pde, condition, r_0, r_1, max_epochs=max_epoch, net=net,
                                                               return_best=True, analytic_solution=analytic_solution,
                                                               monitor=monitor, batch_size=16)

        generator = ExampleGeneratorSpherical(512, r_min=r_0, r_max=r_1)
        rs, thetas, phis = generator.get_examples()
        us = solution(rs, thetas, phis, as_type="np")
        vs = analytic_solution(rs, thetas, phis).detach().cpu().numpy()
        rdiff = abs(us - vs) / vs
        assert np.isclose(us, vs, rtol=0.05).all(), \
            f"Solution doesn't match analytic expectattion {us} != {vs}, relative-diff={rdiff}"

        print("subtest electric-potential-on-gaussian-charged-density passed")

    subtest(SphericalHarmonicsNN(max_degree=1), max_epoch=200)
    subtest(None, max_epoch=500)


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

    harmonics_fn_cartesian = harmonics_fn_cartesian[:(MAX_DEGREE + 1) ** 2]
    harmonics_fn = RealSphericalHarmonics(max_degree=MAX_DEGREE)
    theta = torch.rand(N_SAMPLES) * np.pi
    phi = torch.rand(N_SAMPLES) * 2 * np.pi
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    harmonics = harmonics_fn(theta, phi)
    # test the shape of output
    assert harmonics.shape == OUTPUT_SHAPE, f"got shape={harmonics.shape}; expected shape={OUTPUT_SHAPE}"
    harmonics_cartesian = torch.stack([f(x, y, z) for f in harmonics_fn_cartesian], dim=1)
    abs_diff = abs(harmonics - harmonics_cartesian)

    # test the correctness of spherical harmonics written in terms of theta and phi
    assert torch.max(abs_diff) <= 1e-5, f"difference too large, check again:\n {abs_diff.max()}"


def test_spherical_harmonics_nn():
    # number of training points
    N_SAMPLES = 100
    # highest degree for spherical harmonics
    MAX_DEGREE = 4
    # expected output shape
    OUTPUT_SHAPE = (N_SAMPLES, 1)

    nn = SphericalHarmonicsNN(max_degree=MAX_DEGREE)
    inp = torch.rand(N_SAMPLES, 3)
    outp = nn(inp)
    assert outp.shape == OUTPUT_SHAPE, f"got shape={outp.shape}; expected shape={OUTPUT_SHAPE}"
