import torch
import pytest
from neurodiffeq.neurodiffeq import safe_diff, unsafe_diff, diff

N_SAMPLES = 10


def get_data(flatten_u=False, flatten_t=False, f=lambda x: x ** 2):
    t_shape = (N_SAMPLES,) if flatten_t else (N_SAMPLES, 1)
    u_shape = (N_SAMPLES,) if flatten_u else (N_SAMPLES, 1)
    t = torch.rand(t_shape, requires_grad=True)
    u = f(t).view(u_shape)
    return u, t


def check_output(t, output, f_prime=lambda x: 2 * x):
    assert t.shape == output.shape
    assert torch.isclose(output, t * 2).all()


def test_safe_diff():
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=True, flatten_t=True)
        check_output(t, safe_diff(u, t))
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=True, flatten_t=False)
        check_output(t, safe_diff(u, t))
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=False, flatten_t=True)
        check_output(t, safe_diff(u, t))

    u, t = get_data(flatten_u=False, flatten_t=False)
    check_output(t, safe_diff(u, t))


def test_unsafe_diff():
    u, t = get_data(flatten_u=True, flatten_t=True)
    check_output(t, unsafe_diff(u, t))
    u, t = get_data(flatten_u=True, flatten_t=False)
    check_output(t, unsafe_diff(u, t))
    u, t = get_data(flatten_u=False, flatten_t=True)
    check_output(t, unsafe_diff(u, t))
    u, t = get_data(flatten_u=False, flatten_t=False)
    check_output(t, unsafe_diff(u, t))


def test_diff():
    # with default shape_check
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=True, flatten_t=True)
        check_output(t, diff(u, t))
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=True, flatten_t=False)
        check_output(t, diff(u, t))
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=False, flatten_t=True)
        check_output(t, diff(u, t))
    u, t = get_data(flatten_u=False, flatten_t=False)
    check_output(t, diff(u, t))

    # with shape_check = True
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=True, flatten_t=True)
        check_output(t, diff(u, t, shape_check=True))
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=True, flatten_t=False)
        check_output(t, diff(u, t, shape_check=True))
    with pytest.raises(ValueError):
        u, t = get_data(flatten_u=False, flatten_t=True)
        check_output(t, diff(u, t, shape_check=True))
    u, t = get_data(flatten_u=False, flatten_t=False)
    check_output(t, diff(u, t, shape_check=True))

    # with shape_check = False
    u, t = get_data(flatten_u=False, flatten_t=False)
    check_output(t, diff(u, t))
    u, t = get_data(flatten_u=True, flatten_t=True)
    check_output(t, diff(u, t, shape_check=False))
    u, t = get_data(flatten_u=True, flatten_t=False)
    check_output(t, diff(u, t, shape_check=False))
    u, t = get_data(flatten_u=False, flatten_t=True)
    check_output(t, diff(u, t, shape_check=False))
    u, t = get_data(flatten_u=False, flatten_t=False)
    check_output(t, diff(u, t, shape_check=False))


def test_higher_order_derivatives():
    u, t = get_data(flatten_u=False, flatten_t=False, f=lambda x: x ** 2)
    assert torch.isclose(diff(u, t), t * 2).all()
    assert torch.isclose(diff(u, t, order=2), 2 * torch.ones_like(t)).all()
    for order in range(3, 10):
        assert torch.isclose(diff(u, t, order=order), torch.zeros_like(t)).all()

    u, t = get_data(flatten_u=False, flatten_t=False, f=torch.exp)
    for order in range(1, 10):
        assert torch.isclose(diff(u, t, order=order), u).all()


def test_legacy_siganature():
    u, t = get_data(flatten_u=False, flatten_t=False)
    with pytest.warns(FutureWarning):
        diff(x=u, t=t)
    with pytest.warns(FutureWarning):
        safe_diff(x=u, t=t)
    with pytest.warns(FutureWarning):
        unsafe_diff(x=u, t=t)
