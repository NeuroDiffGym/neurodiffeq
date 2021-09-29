import torch
import pytest
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.operators import grad
from neurodiffeq.operators import _split_u_x
from neurodiffeq.operators import div
from neurodiffeq.operators import curl
from neurodiffeq.operators import laplacian
from neurodiffeq.operators import vector_laplacian

N_SAMPLES = 10

f1 = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
g1 = lambda x, y, z: [2 * x, 2 * y, 2 * z]
f2 = lambda x, y, z: x ** 2 + y ** 2
g2 = lambda x, y, z: [2 * x, 2 * y, torch.zeros_like(z, requires_grad=z.requires_grad)]
f3 = lambda x, y, z: torch.zeros_like(x, requires_grad=True)
g3 = lambda x, y, z: list(map(lambda v: torch.zeros_like(v, requires_grad=v.requires_grad), [x, y, z]))


@pytest.fixture
def x3():
    return [torch.rand((N_SAMPLES, 1), requires_grad=True) for _ in range(3)]


@pytest.mark.parametrize(
    argnames=['u_len', 'x_len', 'err'],
    argvalues=[(0, 0, True), (3, 3, False), (3, 4, True)]
)
def test__split_u_x(u_len, x_len, err):
    us = [torch.zeros(N_SAMPLES, 1, requires_grad=True) for _ in range(u_len)]
    xs = [torch.zeros(N_SAMPLES, 1, requires_grad=True) for _ in range(x_len)]

    if not err:
        vs, ys = _split_u_x(*us, *xs)
        for u, v in zip(us, vs):
            assert u is v
        for x, y in zip(xs, ys):
            assert x is y

    else:
        with pytest.raises(RuntimeError):
            _split_u_x(*us, *xs)


@pytest.mark.parametrize(
    argnames=['f', 'g_func'],
    argvalues=list(zip([f1, f2, f3], [g1, g2, g3])),
)
def test_grad(x3, f, g_func):
    g3_true = g_func(*x3)
    g3_pred = grad(f(*x3), *x3)

    for x, g_true, g_pred in zip(x3, g3_true, g3_pred):
        assert isinstance(g_pred, torch.Tensor)
        assert g_pred.requires_grad
        assert torch.isclose(g_true, g_pred).all()


@pytest.mark.parametrize(argnames='n_dim', argvalues=[1, 2, 3])
def test_div(n_dim):
    xs = [torch.rand(N_SAMPLES, 1, requires_grad=True) for _ in range(n_dim)]
    nets = [FCNN(n_dim, 1) for _ in range(n_dim)]
    us = [net(torch.cat(xs, dim=1)) for net in nets]

    grads = [diff(u, x) for u, x in zip(us, xs)]
    ans = torch.cat(grads, dim=1).sum(dim=1, keepdim=True)

    output = div(*us, *xs)
    assert (output == ans).all()


def test_curl(x3):
    nets = [FCNN(3, 1) for _ in range(3)]
    x, y, z = x3
    u, v, w = [net(torch.cat(x3, dim=1)) for net in nets]

    ans = [
        diff(w, y) - diff(v, z),
        diff(u, z) - diff(w, x),
        diff(v, x) - diff(u, y),
    ]
    output = curl(u, v, w, x, y, z)

    for a, o in zip(ans, output):
        assert (a == o).all()


@pytest.mark.parametrize(argnames='n_dim', argvalues=[1, 2, 3])
def test_laplacian(n_dim):
    xs = [torch.rand(N_SAMPLES, 1, requires_grad=True) for _ in range(n_dim)]
    net = FCNN(n_dim, 1)
    u = net(torch.cat(xs, dim=1))
    ans = torch.cat([diff(u, x, order=2) for x in xs], dim=1).sum(dim=1, keepdim=True)
    output = laplacian(u, *xs)
    assert (ans == output).all()


def test_vector_laplacian(x3):
    nets = [FCNN(3, 1) for _ in range(3)]
    x, y, z = x3
    u, v, w = [net(torch.cat(x3, dim=1)) for net in nets]
    ans = [gd - cc for gd, cc in zip(
        grad(div(u, v, w, x, y, z), x, y, z),
        curl(*curl(u, v, w, x, y, z), x, y, z),
    )]
    output = vector_laplacian(u, v, w, x, y, z)

    for a, o in zip(ans, output):
        assert torch.isclose(o, a).all()
