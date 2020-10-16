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
n_points = 1024
r_min = 1.0
r_max = 1.0
generator = GeneratorSpherical(n_points, r_min=r_min, r_max=r_max)
r, theta, phi = [t.reshape(-1, 1) for t in generator.get_examples()]

degrees = list(range(10))
harmoincs_fn = ZonalSphericalHarmonics(degrees=degrees)

F_r, F_theta, F_phi = [HarmonicsNN(degrees, harmoincs_fn) for _ in range(3)]
vector_u = (F_r(r, theta, phi), F_theta(r, theta, phi), F_phi(r, theta, phi))

f = HarmonicsNN(degrees, harmoincs_fn)
scalar_u = f(r, theta, phi)

curl = lambda a, b, c: spherical_curl(a, b, c, r, theta, phi)
grad = lambda a: spherical_grad(a, r, theta, phi)
div = lambda a, b, c: spherical_div(a, b, c, r, theta, phi)
lap = lambda a: spherical_laplacian(a, r, theta, phi)
vec_lap = lambda a, b, c: spherical_vector_laplacian(a, b, c, r, theta, phi)


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


def test_div_curl():
    curl_u = curl(*vector_u)
    div_curl_u = div(*curl_u)
    assert is_zero(div_curl_u), div_curl_u


def test_curl_grad():
    grad_u = grad(scalar_u)
    curl_grad_u = curl(*grad_u)
    assert is_zero(curl_grad_u), curl_grad_u


def test_div_grad():
    grad_u = grad(scalar_u)
    div_grad_u = div(*grad_u)
    lap_u = lap(scalar_u)
    delta = div_grad_u - lap_u
    assert is_zero(delta), delta


def test_laplacian():
    test_div_grad()


def test_curl_curl():
    curl_curl_u = curl(*curl(*vector_u))
    grad_div_u = grad(div(*vector_u))
    vec_lap_u = vec_lap(*vector_u)

    vec_delta = [cc - (gd - vl) for cc, gd, vl in zip(curl_curl_u, grad_div_u, vec_lap_u)]
    assert is_zero(vec_delta), vec_delta


def test_vec_laplacian():
    test_curl_curl()
