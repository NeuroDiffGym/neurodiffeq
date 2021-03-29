import pytest
import torch
import numpy as np
import torch.nn as nn
from numpy import isclose
from neurodiffeq.function_basis import LegendrePolynomial
from neurodiffeq.function_basis import LegendreBasis
from neurodiffeq.function_basis import ZonalSphericalHarmonics
from neurodiffeq.function_basis import ZonalSphericalHarmonicsLaplacian
from neurodiffeq.neurodiffeq import safe_diff as diff
from scipy.special import legendre  # legendre polynomials
from scipy.special import sph_harm  # spherical harmonics


@pytest.fixture
def n_samples():
    return 50


@pytest.fixture
def shape(n_samples):
    return (n_samples, 1)


@pytest.fixture
def max_degree():
    return 20


def test_legendre_polynomials(shape, max_degree):
    x1 = np.random.rand(*shape)
    x2 = torch.tensor(x1, requires_grad=True)

    for d in range(max_degree):
        p1 = legendre(d)(x1)
        p2 = LegendrePolynomial(d)(x2)
        assert p2.requires_grad, f"output seems detached from the graph"
        p2 = p2.detach().cpu().numpy()
        assert isclose(p2, p1).all(), f"p1 = {p1}, p2 = {p2}, delta = {p1 - p2}, max_delta = {np.max(abs(p1 - p2))}"


def test_legendre_basis(shape, max_degree):
    x1 = np.random.rand(*shape)
    x2 = torch.tensor(x1, requires_grad=True)

    y1 = np.concatenate(
        [legendre(d)(x1) for d in range(max_degree + 1)],
        axis=1,
    )
    net = LegendreBasis(max_degree=max_degree)
    y2 = net(x2)
    assert y2.requires_grad, f"output seems detached from the graph"

    y2 = y2.detach().cpu().numpy()
    assert isclose(y2, y1).all(), f"y1 = {y1}, y2 = {y2}, delta = {y1 - y2}, max_delta = {np.max(abs(y1 - y2))}"


def test_zero_order_spherical_harmonics(shape, max_degree):
    # note that in scipy, theta is azimuthal angle (0, 2 pi) while phi is polar angle (0, pi)
    thetas1 = np.random.rand(*shape) * np.pi * 2
    phis1 = np.random.rand(*shape) * np.pi
    # in neurodiffeq, theta and phi should be exchanged
    thetas2 = torch.tensor(phis1, requires_grad=True)
    phis2 = torch.tensor(thetas1, requires_grad=True)
    order = 0

    y1 = np.concatenate(
        [sph_harm(order, degree, thetas1, phis1) for degree in range(max_degree + 1)],
        axis=1,
    )
    assert (np.imag(y1) == 0).all(), f"y1 has non-zero imaginary part: {y1}"
    y1 = np.real(y1)

    net = ZonalSphericalHarmonics(max_degree)
    y2 = net(thetas2, phis2)
    assert y2.requires_grad, f"output seems detached from the graph"

    y2 = y2.detach().cpu().numpy()
    assert isclose(y2, y1, atol=1e-5, rtol=1e-3).all(), \
        f"y1 = {y1}, y2 = {y2}, delta = {y1 - y2}, max_delta = {np.max(abs(y1 - y2))}"


def test_zero_order_spherical_harmonics_laplacian(shape, max_degree):
    # Somehow, if changing default dtype to float32, the test fails by a large margin
    N_FLOAT = np.float64
    T_FLOAT = torch.float64

    THETA_EPS = 0.1
    r_values = np.random.rand(*shape).astype(N_FLOAT) + 1.1
    theta_values = np.random.uniform(THETA_EPS, np.pi - THETA_EPS, size=shape).astype(N_FLOAT)
    phi_values = np.random.rand(*shape).astype(N_FLOAT) * np.pi * 2

    net = nn.Sequential(
        nn.Linear(1, 10),
        nn.Tanh(),
        nn.Linear(10, max_degree + 1),
    ).to(T_FLOAT)

    harmonics = ZonalSphericalHarmonics(max_degree=max_degree)

    r1 = torch.tensor(r_values, requires_grad=True)
    theta1 = torch.tensor(theta_values, requires_grad=True)
    phi1 = torch.tensor(phi_values, requires_grad=True)
    coeffs1 = net(r1)

    us = torch.sum(coeffs1 * harmonics(theta1, phi1), dim=1, keepdim=True)

    def laplacian1(u, r, theta, phi):
        r_lap = diff(u * r, r, order=2) / r
        theta_lap = diff(diff(u, theta) * torch.sin(theta), theta) / (r ** 2) / torch.sin(theta)
        phi_lap = diff(u, phi, order=2) / (r ** 2) / torch.sin(theta) ** 2
        return r_lap + theta_lap + phi_lap

    lap1 = laplacian1(us, r1, theta1, phi1)
    assert lap1.requires_grad, "lap1 seems detached from graph"

    r2 = torch.tensor(r_values, requires_grad=True)
    theta2 = torch.tensor(theta_values, requires_grad=True)
    phi2 = torch.tensor(phi_values, requires_grad=True)
    coeffs2 = net(r2)

    laplacian2 = ZonalSphericalHarmonicsLaplacian(max_degree=max_degree)
    lap2 = laplacian2(coeffs2, r2, theta2, phi2)
    assert lap2.requires_grad, "lap2 seems detached from graph"

    assert torch.isclose(lap2, lap1, rtol=1e-3, atol=1e-5).all(), \
        f"lap1 = {lap1}\nlap2 = {lap2}\ndelta = {lap1 - lap2}\nmax_delta = {(lap1 - lap2).abs().max().item()}"
