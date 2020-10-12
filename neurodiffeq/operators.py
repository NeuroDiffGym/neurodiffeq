import torch
from torch import sin, cos
from neurodiffeq import diff


def spherical_curl(u_r, u_theta, u_phi, r, theta, phi):
    d_r = lambda u: diff(u, r)
    d_theta = lambda u: diff(u, theta)
    d_phi = lambda u: diff(u, phi)

    curl_r = (d_theta(u_phi * sin(theta)) - d_phi(u_theta)) / (r * sin(theta))
    curl_theta = (d_phi(u_r) / sin(theta) - d_r(u_phi * r)) / r
    curl_phi = (d_r(u_theta * r) - d_theta(u_r)) / r

    return curl_r, curl_theta, curl_phi


def spherical_grad(u, r, theta, phi):
    grad_r = diff(u, r)
    grad_theta = diff(u, theta) / r
    grad_phi = diff(u, phi) / (r * sin(theta))
    return grad_r, grad_theta, grad_phi


def spherical_div(u_r, u_theta, u_phi, r, theta, phi):
    div_r = diff(u_r * r ** 2, r) / r ** 2
    div_theta = diff(u_theta * sin(theta), theta) / (r * sin(theta))
    div_phi = diff(u_phi, phi) / (r * sin(theta))
    return div_r + div_theta + div_phi


def spherical_laplacian(u, r, theta, phi):
    d_r = lambda u: diff(u, r)
    d_theta = lambda u: diff(u, theta)
    d_phi = lambda u: diff(u, phi)

    lap_r = d_r(d_r(u) * r ** 2) / r ** 2
    lap_theta = d_theta(d_theta(u) * sin(theta)) / (r ** 2 * sin(theta))
    lap_phi = d_phi(d_phi(u)) / (r ** 2 * sin(theta) ** 2)
    return lap_r + lap_theta + lap_phi


def spherical_vector_laplacian(u_r, u_theta, u_phi, r, theta, phi):
    d_theta = lambda u: diff(u, theta)
    d_phi = lambda u: diff(u, phi)
    scalar_lap = lambda u: spherical_laplacian(u, r, theta, phi)

    lap_r = \
        scalar_lap(u_r) \
        - 2 * u_r / r ** 2 \
        - 2 * d_theta(u_theta * sin(theta)) / (r ** 2 * sin(theta)) \
        - 2 * d_phi(u_phi) / (r ** 2 * sin(theta))

    lap_theta = \
        scalar_lap(u_theta) \
        - u_theta / (r ** 2 * sin(theta) ** 2) \
        + 2 * d_theta(u_r) / r ** 2 \
        - 2 * cos(theta) * d_phi(u_phi) / (r ** 2 * sin(theta) ** 2)

    lap_phi = \
        scalar_lap(u_phi) \
        - u_phi / (r ** 2 * sin(theta) ** 2) \
        + 2 * d_phi(u_r) / (r ** 2 * sin(theta)) \
        + 2 * cos(theta) * d_phi(u_theta) / (r ** 2 * sin(theta) ** 2)

    return lap_r, lap_theta, lap_phi
