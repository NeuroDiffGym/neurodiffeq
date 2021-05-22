import torch
import pytest
from neurodiffeq.operators import grad

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
