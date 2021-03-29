import pytest
import torch
import torch.nn as nn
from torch import sin, cos
import numpy as np
from neurodiffeq.generators import GeneratorSpherical
from neurodiffeq.function_basis import ZonalSphericalHarmonics
from neurodiffeq.networks import FCNN
from neurodiffeq.operators import spherical_curl
from neurodiffeq.operators import spherical_grad
from neurodiffeq.operators import spherical_div
from neurodiffeq.operators import spherical_laplacian
from neurodiffeq.operators import spherical_vector_laplacian


@pytest.fixture(autouse=True)
def magic():
    torch.manual_seed(42)
    np.random.seed(42)


class HarmonicsNN(nn.Module):
    def __init__(self, degrees, harmonics_fn):
        super(HarmonicsNN, self).__init__()
        self.net_r = FCNN(1, n_output_units=len(degrees))
        self.harmonics_fn = harmonics_fn

    def forward(self, r, theta, phi):
        R = self.net_r(r)
        Y = self.harmonics_fn(theta, phi)
        return (R * Y).sum(dim=1, keepdim=True)


EPS = 1e-4


@pytest.fixture
def x():
    n_points, r_min, r_max = 1024, 1.0, 10.0
    g = GeneratorSpherical(n_points, r_min=r_min, r_max=r_max)
    return [t.reshape(-1, 1) for t in g.get_examples()]


@pytest.fixture
def degrees():
    return list(range(10))


@pytest.fixture
def harmonics_fn(degrees):
    return ZonalSphericalHarmonics(degrees=degrees)


@pytest.fixture
def F(degrees, harmonics_fn):
    return [HarmonicsNN(degrees, harmonics_fn) for _ in range(3)]


@pytest.fixture
def U(F, x):
    return list(map(lambda f: f(*x), F))


@pytest.fixture
def u(degrees, harmonics_fn, x):
    return HarmonicsNN(degrees, harmonics_fn)(*x)


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


def test_div_curl(U, x):
    curl_u = spherical_curl(*U, *x)
    div_curl_u = spherical_div(*curl_u, *x)
    assert is_zero(div_curl_u), div_curl_u


def test_curl_grad(u, x):
    grad_u = spherical_grad(u, *x)
    curl_grad_u = spherical_curl(*grad_u, *x)
    assert is_zero(curl_grad_u), curl_grad_u


def test_div_grad(u, x):
    grad_u = spherical_grad(u, *x)
    div_grad_u = spherical_div(*grad_u, *x)
    lap_u = spherical_laplacian(u, *x)
    delta = div_grad_u - lap_u
    assert is_zero(delta), delta


def test_laplacian(u, x):
    test_div_grad(u, x)


def test_curl_curl(U, x):
    curl_curl_u = spherical_curl(*spherical_curl(*U, *x), *x)
    grad_div_u = spherical_grad(spherical_div(*U, *x), *x)
    vec_lap_u = spherical_vector_laplacian(*U, *x)

    vec_delta = [cc - (gd - vl) for cc, gd, vl in zip(curl_curl_u, grad_div_u, vec_lap_u)]
    assert is_zero(vec_delta), vec_delta


def test_vec_laplacian(U, x):
    test_curl_curl(U, x)
