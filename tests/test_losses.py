import pytest
import torch
from neurodiffeq import diff
from neurodiffeq.losses import _losses as losses

N = 100


def pde_system(u, v, w, x, y):
    return [
        diff(u, x, order=2) + diff(u, y, order=2),
        diff(v, x, order=2) + diff(v, y, order=2),
        diff(w, x, order=2) + diff(w, y, order=2),
    ]


def get_rfx(n_input, n_output, n_equation):
    coords = [torch.rand((N, 1), requires_grad=True) for _ in range(n_input)]
    coords_tensor = torch.cat(coords, dim=1)
    funcs = [torch.sigmoid(torch.sum(coords_tensor, dim=1, keepdim=True)) for _ in range(n_output)]
    residual = [diff(funcs[0], coords[0]) + funcs[0] for _ in range(n_equation)]
    residual = torch.cat(residual, dim=1)
    return residual, funcs, coords


@pytest.mark.parametrize(argnames='n_input', argvalues=[1, 3])
@pytest.mark.parametrize(argnames='n_output', argvalues=[1, 3])
@pytest.mark.parametrize(argnames='n_equation', argvalues=[1, 3])
@pytest.mark.parametrize(
    argnames=('loss_name', 'loss_fn'),
    argvalues=losses.items(),
)
def test_losses(n_input, n_output, n_equation, loss_name, loss_fn):
    r, f, x = get_rfx(n_input, n_output, n_equation)
    loss = loss_fn(r, f, x)
    assert loss.shape == (), f"{loss_name} doesn't output scalar"
    assert loss.requires_grad, f"{loss_name} doesn't require gradient"
