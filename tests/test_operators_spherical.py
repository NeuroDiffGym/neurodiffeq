import pytest
import torch
import torch.nn as nn
from torch import sin, cos
import numpy as np
from neurodiffeq import diff
from neurodiffeq.generators import GeneratorSpherical
from neurodiffeq.function_basis import ZonalSphericalHarmonics
from neurodiffeq.networks import FCNN
from neurodiffeq.operators import spherical_curl
from neurodiffeq.operators import spherical_grad
from neurodiffeq.operators import spherical_div
from neurodiffeq.operators import spherical_laplacian
from neurodiffeq.operators import spherical_vector_laplacian
from neurodiffeq.operators import spherical_to_cartesian, cartesian_to_spherical


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
degrees = list(range(10))


@pytest.fixture
def x():
    n_points, r_min, r_max = 1024, 1.0, 10.0
    g = GeneratorSpherical(n_points, r_min=r_min, r_max=r_max)
    return [t.reshape(-1, 1) for t in g.get_examples()]


@pytest.fixture
def U(x):
    F = [HarmonicsNN(degrees, ZonalSphericalHarmonics(degrees=degrees)) for _ in range(3)]
    return tuple(f(*x) for f in F)


@pytest.fixture
def u(x):
    return HarmonicsNN(degrees, ZonalSphericalHarmonics(degrees=degrees))(*x)


def test_cartesian_to_spherical():
    x = torch.rand(1000, 1, requires_grad=True)
    y = torch.rand(1000, 1, requires_grad=True)
    z = torch.rand(1000, 1, requires_grad=True)
    r, theta, phi = cartesian_to_spherical(x, y, z)
    assert torch.allclose(r * torch.sin(theta) * cos(phi), x)
    assert torch.allclose(r * torch.sin(theta) * sin(phi), y)
    assert torch.allclose(r * torch.cos(theta), z)


def test_spherical_to_cartesian():
    r = torch.rand(1000, 1, requires_grad=True)
    theta = torch.rand(1000, 1, requires_grad=True) * np.pi
    phi = torch.rand(1000, 1, requires_grad=True) * np.pi * 2
    x, y, z = spherical_to_cartesian(r, theta, phi)
    assert torch.allclose(r * torch.sin(theta) * cos(phi), x)
    assert torch.allclose(r * torch.sin(theta) * sin(phi), y)
    assert torch.allclose(r * torch.cos(theta), z)


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


def test_spherical_div(U, x):
    out = spherical_div(*U, *x)
    ur, utheta, uphi = U
    r, theta, phi = x
    ans = diff(r ** 2 * ur, r) / r ** 2 + \
          diff(utheta * sin(theta), theta) / (r * sin(theta)) + \
          diff(uphi, phi) / (r * sin(theta))

    assert torch.allclose(out, ans)


def test_spherical_grad(u, x):
    out_r, out_theta, out_phi = spherical_grad(u, *x)
    r, theta, phi = x
    assert torch.allclose(out_r, diff(u, r))
    assert torch.allclose(out_theta, diff(u, theta) / r)
    assert torch.allclose(out_phi, diff(u, phi) / (r * sin(theta)))


def test_spherical_curl(U, x):
    out_r, out_theta, out_phi = spherical_curl(*U, *x)
    ur, utheta, uphi = U
    r, theta, phi = x
    assert torch.allclose(out_r, (diff(uphi * sin(theta), theta) - diff(utheta, phi)) / (r * sin(theta)))
    assert torch.allclose(out_theta, (diff(ur, phi) / sin(theta) - diff(r * uphi, r)) / r)
    assert torch.allclose(out_phi, (diff(r * utheta, r) - diff(ur, theta)) / r)


def test_spherical_laplacian(u, x):
    out = spherical_laplacian(u, *x)
    r, theta, phi = x
    assert torch.allclose(
        out,
        diff(r ** 2 * diff(u, r), r) / r ** 2
        + diff(sin(theta) * diff(u, theta), theta) / (r ** 2 * sin(theta))
        + diff(u, phi, order=2) / (r ** 2 * sin(theta) ** 2)
    )


def test_spherical_vector_laplacian(U, x):
    out_r, out_theta, out_phi = spherical_vector_laplacian(*U, *x)
    ur, utheta, uphi = U
    r, theta, phi = x

    def scalar_lap(u):
        return diff(r ** 2 * diff(u, r), r) / r ** 2 \
               + diff(sin(theta) * diff(u, theta), theta) / (r ** 2 * sin(theta)) \
               + diff(u, phi, order=2) / (r ** 2 * sin(theta) ** 2)

    assert torch.allclose(
        out_r,
        scalar_lap(ur)
        - 2 * ur / r ** 2
        - 2 / (r ** 2 * sin(theta)) * diff(utheta * sin(theta), theta)
        - 2 / (r ** 2 * sin(theta)) * diff(uphi, phi)
    )
    assert torch.allclose(
        out_theta,
        scalar_lap(utheta)
        - utheta / (r ** 2 * sin(theta) ** 2)
        + 2 / r ** 2 * diff(ur, theta)
        - 2 * cos(theta) / (r ** 2 * sin(theta) ** 2) * diff(uphi, phi)
    )
    assert torch.allclose(
        out_phi,
        scalar_lap(uphi)
        - uphi / (r ** 2 * sin(theta) ** 2)
        + 2 / (r ** 2 * sin(theta)) * diff(ur, phi)
        + 2 * cos(theta) / (r ** 2 * sin(theta) ** 2) * diff(utheta, phi)
    )
