import torch
import numpy as np
import random
from neurodiffeq.conditions import NoCondition
from neurodiffeq.conditions import IVP
from neurodiffeq.conditions import EnsembleCondition
from neurodiffeq.conditions import DirichletBVP
from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.conditions import DirichletBVPSpherical
from neurodiffeq.conditions import InfDirichletBVPSpherical
from neurodiffeq.conditions import DirichletBVPSphericalBasis
from neurodiffeq.conditions import InfDirichletBVPSphericalBasis
from neurodiffeq.conditions import IBVP1D
from neurodiffeq.networks import FCNN
from neurodiffeq.neurodiffeq import safe_diff as diff
from pytest import raises

MAGIC = 42
torch.manual_seed(MAGIC)
np.random.seed(MAGIC)

N_SAMPLES = 10
ones = torch.ones(N_SAMPLES, 1, requires_grad=True)

x0, x1 = random.random(), random.random()
y0, y1 = random.random(), random.random()


def all_close(x_tensor, y_tensor, rtol=5e-4, atol=1e-6, equal_nan=False):
    if isinstance(y_tensor, (float, int)):
        y_tensor = torch.ones_like(x_tensor) * y_tensor
    return torch.isclose(x_tensor, y_tensor, rtol=rtol, atol=atol, equal_nan=equal_nan).all()


def test_no_condition():
    N_INPUTS = 5
    N_OUTPUTS = 5

    for n_in, n_out in zip(range(1, N_INPUTS), range(1, N_OUTPUTS)):
        xs = [torch.rand(N_SAMPLES, 1, requires_grad=True) for _ in range(n_in)]
        net = FCNN(n_in, n_out)

        cond = NoCondition()
        y_cond = cond.enforce(net, *xs)
        y_raw = net(torch.cat(xs, dim=1))
        assert (y_cond == y_raw).all()


def test_ivp():
    x = x0 * ones
    net = FCNN(1, 1)

    cond = IVP(x0, y0)
    y = cond.enforce(net, x)
    assert torch.isclose(y, y0 * ones).all(), "y(x_0) != y_0"

    cond = IVP(x0, y0, y1)
    y = cond.enforce(net, x)
    assert all_close(y, y0), "y(x_0) != y_0"
    assert all_close(diff(y, x), y1), "y'(x_0) != y'_0"


def test_ensemble_condition():
    net = FCNN(1, 2)
    cond = EnsembleCondition(
        IVP(x0, y0),
        IVP(x1, y0, y1),
    )

    x = x0 * ones
    y = cond.enforce(net, x)
    ya = y[:, 0:1]
    assert all_close(ya, y0), "y(x_0) != y_0"
    x = x1 * ones
    y = cond.enforce(net, x)
    yb = y[:, 1:2]
    assert all_close(yb, y0), "y(x_0) != y_0"
    assert all_close(diff(yb, x), y1), "y'(x_0) != y'_0"

    net = FCNN(1, 1)
    cond = EnsembleCondition(
        IVP(x0, y0),
    )
    x = x0 * ones
    y = cond.enforce(net, x)
    assert all_close(y, y0), "y(x_0) != y_0"


def test_dirichlet_bvp():
    cond = DirichletBVP(x0, y0, x1, y1)
    net = FCNN(1, 1)

    x = x0 * ones
    y = cond.enforce(net, x)
    assert all_close(y, y0), "y(x_0) != y_0"

    x = x1 * ones
    y = cond.enforce(net, x)
    assert all_close(y, y1), "y(x_1) != y_1"


def test_dirichlet_bvp_2d():
    # fix u(x, y) at the four corners (x0, y0), (x0, y1), (x1, y0), (x1, y1),
    u00, u01, u10, u11 = [random.random() for _ in range(4)]

    # set the boundary conditions on the four sides
    net_f0, net_f1, net_g0, net_g1 = [FCNN(1, 1) for _ in range(4)]
    cond_f0 = DirichletBVP(y0, u00, y1, u01)
    cond_f1 = DirichletBVP(y0, u10, y1, u11)
    cond_g0 = DirichletBVP(x0, u00, x1, u10)
    cond_g1 = DirichletBVP(x0, u01, x1, u11)
    f0 = lambda y: cond_f0.enforce(net_f0, y)
    f1 = lambda y: cond_f1.enforce(net_f1, y)
    g0 = lambda x: cond_g0.enforce(net_g0, x)
    g1 = lambda x: cond_g1.enforce(net_g1, x)

    # test whether condition is enforced
    condition = DirichletBVP2D(x0, f0, x1, f1, y0, g0, y1, g1)
    net = FCNN(2, 1)

    x = x0 * ones
    y = torch.linspace(y0, y1, ones.numel(), requires_grad=True).reshape(-1, 1)
    assert all_close(condition.enforce(net, x, y), f0(y)), "left boundary not satisfied"

    x = x1 * ones
    y = torch.linspace(y0, y1, ones.numel(), requires_grad=True).reshape(-1, 1)
    assert all_close(condition.enforce(net, x, y), f1(y)), "right boundary not satisfied"

    x = torch.linspace(x0, x1, ones.numel(), requires_grad=True).reshape(-1, 1)
    y = y0 * ones
    assert all_close(condition.enforce(net, x, y), g0(x)), "lower boundary not satisfied"

    x = torch.linspace(x0, x1, ones.numel(), requires_grad=True).reshape(-1, 1)
    y = y1 * ones
    assert all_close(condition.enforce(net, x, y), g1(x)), "upper boundary not satisfied"


def test_ibvp_1d():
    t0, t1 = random.random(), random.random()

    u00, u01, u10, u11 = [random.random() for _ in range(4)]

    # set the initial condition ut0(x) = u(x, t0)
    net_ut0 = FCNN(1, 1)
    cond_ut0 = DirichletBVP(x0, u00, x1, u10)
    ut0 = lambda x: cond_ut0.enforce(net_ut0, x)

    # set the Dirichlet boundary conditions g(t) = u(x0, t) and h(t) = u(x1, t)
    net_g, net_h = FCNN(1, 1), FCNN(1, 1)
    cond_g = IVP(t0, u00)
    cond_h = IVP(t0, u10)
    g = lambda t: cond_g.enforce(net_g, t)
    h = lambda t: cond_h.enforce(net_h, t)

    # set the Neumann boundary conditions p(t) = u'_x(x0, t) and q(t) = u'_x(x1, t)
    x = x0 * ones
    p0 = diff(ut0(x), x)[0, 0].item()
    x = x1 * ones
    q0 = diff(ut0(x), x)[0, 0].item()
    p1, q1 = random.random(), random.random()
    net_p, net_q = FCNN(1, 1), FCNN(1, 1)
    cond_p = DirichletBVP(t0, p0, t1, p1)
    cond_q = DirichletBVP(t0, q0, t1, q1)
    p = lambda t: cond_p.enforce(net_p, t)
    q = lambda t: cond_q.enforce(net_q, t)

    # initialize a random network
    net = FCNN(2, 1)

    # test Dirichlet-Dirichlet condition
    condition = IBVP1D(x0, x1, t0, ut0, x_min_val=g, x_max_val=h)
    x = torch.linspace(x0, x1, N_SAMPLES, requires_grad=True).view(-1, 1)
    t = t0 * ones
    assert all_close(condition.enforce(net, x, t), ut0(x)), "initial condition not satisfied"
    x = x0 * ones
    t = torch.linspace(t0, t1, N_SAMPLES, requires_grad=True).view(-1, 1)
    assert all_close(condition.enforce(net, x, t), g(t)), "left Dirichlet BC not satisfied"
    x = x1 * ones
    t = torch.linspace(t0, t1, N_SAMPLES, requires_grad=True).view(-1, 1)
    assert all_close(condition.enforce(net, x, t), h(t)), "right Dirichlet BC not satisfied"

    # test Dirichlet-Neumann condition
    condition = IBVP1D(x0, x1, t0, ut0, x_min_val=g, x_max_prime=q)
    x = torch.linspace(x0, x1, N_SAMPLES, requires_grad=True).view(-1, 1)
    t = t0 * ones
    assert all_close(condition.enforce(net, x, t), ut0(x)), "initial condition not satisfied"
    x = x0 * ones
    t = torch.linspace(t0, t1, N_SAMPLES, requires_grad=True).view(-1, 1)
    assert all_close(condition.enforce(net, x, t), g(t)), "left Dirichlet BC not satisfied"
    x = x1 * ones
    t = torch.linspace(t0, t1, N_SAMPLES, requires_grad=True).view(-1, 1)
    assert all_close(diff(condition.enforce(net, x, t), x), q(t)), "right Neumann BC not satisfied"

    # test Neumann-Dirichlet condition
    condition = IBVP1D(x0, x1, t0, ut0, x_min_prime=p, x_max_val=h)
    x = torch.linspace(x0, x1, N_SAMPLES, requires_grad=True).view(-1, 1)
    t = t0 * ones
    assert all_close(condition.enforce(net, x, t), ut0(x)), "initial condition not satisfied"
    x = x0 * ones
    t = torch.linspace(t0, t1, N_SAMPLES, requires_grad=True).view(-1, 1)
    assert all_close(diff(condition.enforce(net, x, t), x), p(t)), "left Neumann BC not satisfied"
    x = x1 * ones
    t = torch.linspace(t0, t1, N_SAMPLES, requires_grad=True).view(-1, 1)
    assert all_close(condition.enforce(net, x, t), h(t)), "right Dirichlet BC not satisfied"

    # test Neumann-Neumann condition
    condition = IBVP1D(x0, x1, t0, ut0, x_min_prime=p, x_max_prime=q)
    x = torch.linspace(x0, x1, N_SAMPLES, requires_grad=True).view(-1, 1)
    t = t0 * ones
    assert all_close(condition.enforce(net, x, t), ut0(x)), "initial condition not satisfied"
    x = x0 * ones
    t = torch.linspace(t0, t1, N_SAMPLES, requires_grad=True).view(-1, 1)
    assert all_close(diff(condition.enforce(net, x, t), x), p(t)), "left Neumann BC not satisfied"
    x = x1 * ones
    t = torch.linspace(t0, t1, N_SAMPLES, requires_grad=True).view(-1, 1)
    assert all_close(diff(condition.enforce(net, x, t), x), q(t)), "right Neumann BC not satisfied"

    # test unimplemented combination of conditions
    with raises(NotImplementedError):
        IBVP1D(
            t_min=0, t_min_val=lambda x: 0,
            x_min=0, x_min_val=None, x_min_prime=None,
            x_max=1, x_max_val=None, x_max_prime=None,
        )
    with raises(NotImplementedError):
        IBVP1D(
            t_min=0, t_min_val=lambda x: 0,
            x_min=0, x_min_val=lambda t: 0, x_min_prime=lambda t: 0,
            x_max=1, x_max_val=None, x_max_prime=None,
        )
    with raises(NotImplementedError):
        IBVP1D(
            t_min=0, t_min_val=lambda x: 0,
            x_min=0, x_min_val=None, x_min_prime=lambda t: 0,
            x_max=1, x_max_val=None, x_max_prime=None,
        )


def test_dirichlet_bvp_spherical():
    r0, r1 = x0, x1
    r2 = (r0 + r1) / 2

    no_condition = NoCondition()
    # B.C. for the interior boundary (r_min)
    net_f = FCNN(2, 1)
    f = lambda th, ph: no_condition.enforce(net_f, th, ph)

    # B.C. for the exterior boundary (r_max)
    net_g = FCNN(2, 1)
    g = lambda th, ph: no_condition.enforce(net_g, th, ph)

    condition = DirichletBVPSpherical(r_0=r0, f=f, r_1=r1, g=g)

    net = FCNN(3, 1)
    theta = torch.rand(N_SAMPLES, 1) * np.pi
    phi = torch.rand(N_SAMPLES, 1) * 2 * np.pi
    r = r0 * ones
    assert all_close(condition.enforce(net, r, theta, phi), f(theta, phi)), "inner Dirichlet BC not satisfied"
    r = r1 * ones
    assert all_close(condition.enforce(net, r, theta, phi), g(theta, phi)), "inner Dirichlet BC not satisfied"

    condition = DirichletBVPSpherical(r_0=r2, f=f)
    r = r2 * ones
    assert all_close(condition.enforce(net, r, theta, phi), f(theta, phi)), "single ended BC not satisfied"


def test_inf_dirichlet_bvp_spherical():
    r0 = random.random()
    r1 = 1e15
    no_condition = NoCondition()
    net_f, net_g = FCNN(2, 1), FCNN(2, 1)

    # B.C. for the interior boundary (r=r_min)
    f = lambda th, ph: no_condition.enforce(net_f, th, ph)
    # B.C. for the exterior boundary (r=infinity)
    g = lambda th, ph: no_condition.enforce(net_g, th, ph)

    net = FCNN(3, 1)
    condition = InfDirichletBVPSpherical(r_0=r0, f=f, g=g, order=1)
    theta = torch.rand(10, 1) * np.pi
    phi = torch.rand(10, 1) * (2 * np.pi)

    r = r0 * ones
    assert all_close(condition.enforce(net, r, theta, phi), f(theta, phi)), "inner DirichletBC not satisfied"
    r = r1 * ones
    assert all_close(condition.enforce(net, r, theta, phi), g(theta, phi)), "Infinity DirichletBC not satisfied"


def test_dirichlet_bvp_spherical_basis():
    N_COMPONENTS = 25
    r0, r1 = x0, x1
    r2 = (r0 + r1) / 2

    R0 = torch.rand(N_SAMPLES, N_COMPONENTS)
    R1 = torch.rand(N_SAMPLES, N_COMPONENTS)
    R2 = torch.rand(N_SAMPLES, N_COMPONENTS)

    condition = DirichletBVPSphericalBasis(r_0=r0, R_0=R0, r_1=r1, R_1=R1)
    net = FCNN(1, N_COMPONENTS)

    r = r0 * ones
    assert all_close(condition.enforce(net, r), R0), "inner Dirichlet BC not satisfied"
    r = r1 * ones
    assert all_close(condition.enforce(net, r), R1), "outer Dirichlet BC not satisfied"

    condition = DirichletBVPSphericalBasis(r_0=r2, R_0=R2)
    r = r2 * ones
    assert all_close(condition.enforce(net, r), R2), "single ended BC not satisfied"


def test_inf_dirichlet_bvp_spherical_basis():
    N_COMPONENTS = 25
    r0 = random.random()
    r_inf = 1e15

    R0 = torch.rand(N_SAMPLES, N_COMPONENTS)
    R_inf = torch.rand(N_SAMPLES, N_COMPONENTS)

    condition = InfDirichletBVPSphericalBasis(r_0=r0, R_0=R0, R_inf=R_inf)
    net = FCNN(1, N_COMPONENTS)

    r = r0 * ones
    assert all_close(condition.enforce(net, r), R0), "inner Dirichlet BC not satisfied"
    r = r_inf * ones
    assert all_close(condition.enforce(net, r), R_inf), "Infinity Dirichlet BC not satisfied"

