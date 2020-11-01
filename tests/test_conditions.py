import torch
import numpy as np
import random
from neurodiffeq.conditions import NoCondition
from neurodiffeq.conditions import IVP
from neurodiffeq.conditions import EnsembleCondition
from neurodiffeq.conditions import DirichletBVP
from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.networks import FCNN
from neurodiffeq.neurodiffeq import diff

MAGIC = 42
torch.manual_seed(MAGIC)
np.random.seed(MAGIC)

N_SAMPLES = 10
ones = torch.ones(N_SAMPLES, 1, requires_grad=True)

x0, x1 = random.random(), random.random()
y0, y1 = random.random(), random.random()


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
    assert torch.isclose(y, y0 * ones).all(), "y(x_0) != y_0"
    assert torch.isclose(diff(y, x), y1 * ones).all(), "y'(x_0) != y'_0"


def test_ensemble_condition():
    net = FCNN(1, 2)
    cond = EnsembleCondition(
        IVP(x0, y0),
        IVP(x1, y0, y1),
    )

    x = x0 * ones
    y = cond.enforce(net, x)
    ya = y[:, 0:1]
    assert (ya == y0).all(), "y(x_0) != y_0"
    x = x1 * ones
    y = cond.enforce(net, x)
    yb = y[:, 1:2]
    assert torch.isclose(yb, y0 * ones).all(), "y(x_0) != y_0"
    assert torch.isclose(diff(yb, x), y1 * ones).all(), "y'(x_0) != y'_0"

    net = FCNN(1, 1)
    cond = EnsembleCondition(
        IVP(x0, y0),
    )
    x = x0 * ones
    y = cond.enforce(net, x)
    assert torch.isclose(y, y0 * ones).all(), "y(x_0) != y_0"


def test_dirichlet_bvp():
    cond = DirichletBVP(x0, y0, x1, y1)
    net = FCNN(1, 1)

    x = x0 * ones
    y = cond.enforce(net, x)
    assert torch.isclose(y, y0 * ones).all(), "y(x_0) != y_0"

    x = x1 * ones
    y = cond.enforce(net, x)
    assert torch.isclose(y, y1 * ones).all(), "y(x_1) != y_1"


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
    assert torch.isclose(condition.enforce(net, x, y), f0(y)).all(), "left boundary not satisfied"

    x = x1 * ones
    y = torch.linspace(y0, y1, ones.numel(), requires_grad=True).reshape(-1, 1)
    assert torch.isclose(condition.enforce(net, x, y), f1(y)).all(), "right boundary not satisfied"

    x = torch.linspace(x0, x1, ones.numel(), requires_grad=True).reshape(-1, 1)
    y = y0 * ones
    assert torch.isclose(condition.enforce(net, x, y), g0(x)).all(), "lower boundary not satisfied"

    x = torch.linspace(x0, x1, ones.numel(), requires_grad=True).reshape(-1, 1)
    y = y1 * ones
    assert torch.isclose(condition.enforce(net, x, y), g1(x)).all(), "upper boundary not satisfied"
