import torch
import numpy as np
import random
from neurodiffeq.conditions import NoCondition
from neurodiffeq.conditions import IVP
from neurodiffeq.conditions import EnsembleCondition
from neurodiffeq.conditions import DirichletBVP
from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.conditions import IBVP1D
from neurodiffeq.networks import FCNN
from neurodiffeq.neurodiffeq import diff
from pytest import raises

MAGIC = 42
torch.manual_seed(MAGIC)
np.random.seed(MAGIC)

N_SAMPLES = 10
ones = torch.ones(N_SAMPLES, 1, requires_grad=True)

x0, x1 = random.random(), random.random()
y0, y1 = random.random(), random.random()


def all_close(x_tensor, y_tensor, rtol=1e-4, atol=1e-6, equal_nan=False):
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
