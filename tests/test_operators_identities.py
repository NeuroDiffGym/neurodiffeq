import pytest
import numpy as np
import torch
from neurodiffeq.networks import FCNN
from neurodiffeq.conditions import NoCondition
from neurodiffeq.operators import grad as cartesian_grad, spherical_grad, cylindrical_grad
from neurodiffeq.operators import div as cartesian_div, spherical_div, cylindrical_div
from neurodiffeq.operators import curl as cartesian_curl, spherical_curl, cylindrical_curl
from neurodiffeq.operators import laplacian as cartesian_laplacian, spherical_laplacian, cylindrical_laplacian
from neurodiffeq.operators import vector_laplacian as cartesian_vector_laplacian, spherical_vector_laplacian, \
    cylindrical_vector_laplacian

EPS = 1e-4
N_SAMPLES = 1000

CARTESIAN_X = (
    torch.rand(N_SAMPLES, 1, requires_grad=True),  # x
    torch.rand(N_SAMPLES, 1, requires_grad=True),  # y
    torch.rand(N_SAMPLES, 1, requires_grad=True),  # z
)

SPHERICAL_X = (
    torch.rand(N_SAMPLES, 1, requires_grad=True) + 0.1,  # r
    torch.rand(N_SAMPLES, 1, requires_grad=True) * (np.pi - 0.2) + 0.1,  # theta
    torch.rand(N_SAMPLES, 1, requires_grad=True) * np.pi * 2,  # phi
)

CYLINDRICAL_X = (
    torch.rand(N_SAMPLES, 1, requires_grad=True) + 0.1,  # rho
    torch.rand(N_SAMPLES, 1, requires_grad=True) * np.pi * 2,  # phi
    torch.rand(N_SAMPLES, 1, requires_grad=True),  # z
)


def vector_field(x):
    cond = NoCondition()
    return tuple(cond.enforce(FCNN(3, 1), *x) for _ in range(3))


def scalar_field(x):
    cond = NoCondition()
    return cond.enforce(FCNN(3, 1), *x)


def is_zero(t):
    if isinstance(t, (tuple, list)):
        for i in t:
            if not is_zero(i):
                return False
        return True
    elif isinstance(t, torch.Tensor):
        return t.detach().cpu().max() < EPS
    else:
        raise ValueError(f"t must be list, tuple or tensor; got {type(t)}")


@pytest.mark.parametrize(
    argnames=['x', '_curl', '_div'],
    argvalues=[
        (CARTESIAN_X, cartesian_curl, cartesian_div),
        (CYLINDRICAL_X, cylindrical_curl, cylindrical_div),
        (SPHERICAL_X, spherical_curl, spherical_div),
    ]
)
def test_div_curl(x, _curl, _div):
    U = vector_field(x)
    curl_u = _curl(*U, *x)
    div_curl_u = _div(*curl_u, *x)
    assert is_zero(div_curl_u), div_curl_u


@pytest.mark.parametrize(
    argnames=['x', '_grad', '_curl'],
    argvalues=[
        (CARTESIAN_X, cartesian_grad, cartesian_curl),
        (CYLINDRICAL_X, cylindrical_grad, cylindrical_curl),
        (SPHERICAL_X, spherical_grad, spherical_curl),
    ]
)
def test_curl_grad(x, _grad, _curl):
    u = scalar_field(x)
    grad_u = _grad(u, *x)
    curl_grad_u = _curl(*grad_u, *x)
    assert is_zero(curl_grad_u), curl_grad_u


@pytest.mark.parametrize(
    argnames=['x', '_grad', '_div', '_laplacian'],
    argvalues=[
        (CARTESIAN_X, cartesian_grad, cartesian_div, cartesian_laplacian),
        (CYLINDRICAL_X, cylindrical_grad, cylindrical_div, cylindrical_laplacian),
        (SPHERICAL_X, spherical_grad, spherical_div, spherical_laplacian),
    ]
)
def test_div_grad(x, _div, _grad, _laplacian):
    u = scalar_field(x)
    grad_u = _grad(u, *x)
    div_grad_u = _div(*grad_u, *x)
    lap_u = _laplacian(u, *x)
    delta = div_grad_u - lap_u
    assert is_zero(delta), delta


@pytest.mark.parametrize(
    argnames=['x', '_grad', '_div', '_laplacian'],
    argvalues=[
        (CARTESIAN_X, cartesian_grad, cartesian_div, cartesian_laplacian),
        (CYLINDRICAL_X, cylindrical_grad, cylindrical_div, cylindrical_laplacian),
        (SPHERICAL_X, spherical_grad, spherical_div, spherical_laplacian),
    ]
)
def test_laplacian(x, _div, _grad, _laplacian):
    test_div_grad(x, _div, _grad, _laplacian)


@pytest.mark.parametrize(
    argnames=['x', '_grad', '_div', '_curl', '_vector_laplacian'],
    argvalues=[
        (CARTESIAN_X, cartesian_grad, cartesian_div, cartesian_curl, cartesian_vector_laplacian),
        (CYLINDRICAL_X, cylindrical_grad, cylindrical_div, cylindrical_curl, cylindrical_vector_laplacian),
        (SPHERICAL_X, spherical_grad, spherical_div, spherical_curl, spherical_vector_laplacian),
    ]
)
def test_curl_curl(x, _grad, _div, _curl, _vector_laplacian):
    U = vector_field(x)
    curl_curl_u = _curl(*_curl(*U, *x), *x)
    grad_div_u = _grad(_div(*U, *x), *x)
    vec_lap_u = _vector_laplacian(*U, *x)

    vec_delta = [cc - (gd - vl) for cc, gd, vl in zip(curl_curl_u, grad_div_u, vec_lap_u)]
    assert is_zero(vec_delta), vec_delta


@pytest.mark.parametrize(
    argnames=['x', '_grad', '_div', '_curl', '_vector_laplacian'],
    argvalues=[
        (CARTESIAN_X, cartesian_grad, cartesian_div, cartesian_curl, cartesian_vector_laplacian),
        (CYLINDRICAL_X, cylindrical_grad, cylindrical_div, cylindrical_curl, cylindrical_vector_laplacian),
        (SPHERICAL_X, spherical_grad, spherical_div, spherical_curl, spherical_vector_laplacian),
    ]
)
def test_vector_laplacian(x, _grad, _div, _curl, _vector_laplacian):
    test_curl_curl(x, _grad, _div, _curl, _vector_laplacian)
