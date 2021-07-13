import pytest
import torch
import numpy as np
from neurodiffeq import diff
from neurodiffeq.generators import Generator1D, SamplerGenerator, EnsembleGenerator
from neurodiffeq.networks import FCNN
from neurodiffeq.conditions import NoCondition
from neurodiffeq.operators import cylindrical_curl
from neurodiffeq.operators import cylindrical_grad
from neurodiffeq.operators import cylindrical_div
from neurodiffeq.operators import cylindrical_laplacian
from neurodiffeq.operators import cylindrical_vector_laplacian
from neurodiffeq.operators import cylindrical_to_cartesian, cartesian_to_cylindrical

R_MIN, R_MAX = 1.0, 10.
Z_MIN, Z_MAX = -10., 10.
GRID = (10, 10, 10)


@pytest.fixture
def x():
    g = EnsembleGenerator(
        Generator1D(GRID[0], R_MIN, R_MAX, method='uniform'),
        Generator1D(GRID[1], 0, np.pi * 2, method='uniform'),
        Generator1D(GRID[2], Z_MIN, Z_MAX, method='uniform'),
    )
    return SamplerGenerator(g).get_examples()


@pytest.fixture
def u(x):
    cond = NoCondition()
    net = FCNN(3, 1)
    return cond.enforce(net, *x)


@pytest.fixture
def U(x):
    cond = NoCondition()
    nets = [FCNN(3, 1) for _ in range(3)]
    return tuple(cond.enforce(net, *x) for net in nets)


def test_cylindrical_to_cartesian(x):
    rho, phi, zs = x
    x, y, zc = cylindrical_to_cartesian(*x)

    assert torch.allclose(rho * torch.cos(phi), x)
    assert torch.allclose(rho * torch.sin(phi), y)
    assert torch.allclose(zs, zc)


def test_cartesian_to_cylindrical():
    x = torch.rand(1024, requires_grad=True)
    y = torch.rand(1024, requires_grad=True)
    zc = torch.rand(1024, requires_grad=True)
    rho, phi, zs = cartesian_to_cylindrical(x, y, zc)

    assert torch.allclose(rho * torch.cos(phi), x)
    assert torch.allclose(rho * torch.sin(phi), y)
    assert torch.allclose(zs, zc)


def test_cylindrical_grad(u, x):
    out_rho, out_phi, out_z = cylindrical_grad(u, *x)
    rho, phi, z = x
    assert torch.allclose(out_rho, diff(u, rho))
    assert torch.allclose(out_phi, diff(u, phi) / rho)
    assert torch.allclose(out_z, diff(u, z))


def test_cylindrical_div(U, x):
    out = cylindrical_div(*U, *x)
    rho, phi, z = x
    urho, uphi, uz = U

    assert torch.allclose(out, diff(rho * urho, rho) / rho + diff(uphi, phi) / rho + diff(uz, z))


def test_cylindrical_curl(U, x):
    out_rho, out_phi, out_z = cylindrical_curl(*U, *x)
    rho, phi, z = x
    urho, uphi, uz = U

    assert torch.allclose(out_rho, diff(uz, phi) / rho - diff(uphi, z))
    assert torch.allclose(out_phi, diff(urho, z) - diff(uz, rho))
    assert torch.allclose(out_z, (diff(rho * uphi, rho) - diff(urho, phi)) / rho)


def test_cylindrical_laplacian(u, x):
    out = cylindrical_laplacian(u, *x)
    rho, phi, z = x

    assert torch.allclose(
        out,
        diff(rho * diff(u, rho), rho) / rho + diff(u, phi, order=2) / rho ** 2 + diff(u, z, order=2)
    )


def test_cylindrical_vector_laplacian(U, x):
    out_rho, out_phi, out_z = cylindrical_vector_laplacian(*U, *x)
    rho, phi, z = x
    urho, uphi, uz = U

    def scalar_lap(u):
        return diff(rho * diff(u, rho), rho) / rho + diff(u, phi, order=2) / rho ** 2 + diff(u, z, order=2)

    assert torch.allclose(out_rho, scalar_lap(urho) - urho / rho ** 2 - 2 / rho ** 2 * diff(uphi, phi))
    assert torch.allclose(out_phi, scalar_lap(uphi) - uphi / rho ** 2 + 2 / rho ** 2 * diff(urho, phi))
    assert torch.allclose(out_z, scalar_lap(uz))
