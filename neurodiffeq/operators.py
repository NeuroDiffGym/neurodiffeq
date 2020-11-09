import torch
from torch import sin, cos
from neurodiffeq.neurodiffeq import safe_diff as diff


def spherical_curl(u_r, u_theta, u_phi, r, theta, phi):
    r"""Derives and evaluates the spherical curl of a spherical vector field :math:`u`.

    :param u_r: The :math:`r`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_r: `torch.Tensor`
    :param u_theta: The :math:`\theta`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_theta: `torch.Tensor`
    :param u_phi: The :math:`\phi`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_phi: `torch.Tensor`
    :param r: A vector of :math:`r`-coordinate values, must have shape (n_samples, 1).
    :type r: `torch.Tensor`
    :param theta: A vector of :math:`\theta`-coordinate values, must have shape (n_samples, 1).
    :type theta: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :return: The :math:`r`, :math:`\theta`, and :math:`\phi` components of the curl, each with shape (n_samples, 1).
    :rtype: tuple[`torch.Tensor`]
    """

    d_r = lambda u: diff(u, r)
    d_theta = lambda u: diff(u, theta)
    d_phi = lambda u: diff(u, phi)

    curl_r = (d_theta(u_phi * sin(theta)) - d_phi(u_theta)) / (r * sin(theta))
    curl_theta = (d_phi(u_r) / sin(theta) - d_r(u_phi * r)) / r
    curl_phi = (d_r(u_theta * r) - d_theta(u_r)) / r

    return curl_r, curl_theta, curl_phi


def spherical_grad(u, r, theta, phi):
    r"""Derives and evaluates the spherical gradient of a spherical scalar field :math:`u`.

    :param u: A scalar field :math:`u`, must have shape (n_samples, 1).
    :type u: `torch.Tensor`
    :param r: A vector of :math:`r`-coordinate values, must have shape (n_samples, 1).
    :type r: `torch.Tensor`
    :param theta: A vector of :math:`\theta`-coordinate values, must have shape (n_samples, 1).
    :type theta: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :return: The :math:`r`, :math:`\theta`, and :math:`\phi` components of the gradient, each with shape (n_samples, 1).
    :rtype: tuple[`torch.Tensor`]
    """
    grad_r = diff(u, r)
    grad_theta = diff(u, theta) / r
    grad_phi = diff(u, phi) / (r * sin(theta))
    return grad_r, grad_theta, grad_phi


def spherical_div(u_r, u_theta, u_phi, r, theta, phi):
    r"""Derives and evaluates the spherical divergence of a spherical vector field :math:`u`.

    :param u_r: The :math:`r`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_r: `torch.Tensor`
    :param u_theta: The :math:`\theta`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_theta: `torch.Tensor`
    :param u_phi: The :math:`\phi`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_phi: `torch.Tensor`
    :param r: A vector of :math:`r`-coordinate values, must have shape (n_samples, 1).
    :type r: `torch.Tensor`
    :param theta: A vector of :math:`\theta`-coordinate values, must have shape (n_samples, 1).
    :type theta: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :return: The divergence evaluated at :math:`(r, \theta, \phi)`, with shape (n_samples, 1).
    :rtype: `torch.Tensor`
    """
    div_r = diff(u_r * r ** 2, r) / r ** 2
    div_theta = diff(u_theta * sin(theta), theta) / (r * sin(theta))
    div_phi = diff(u_phi, phi) / (r * sin(theta))
    return div_r + div_theta + div_phi


def spherical_laplacian(u, r, theta, phi):
    r"""Derives and evaluates the spherical laplacian of a spherical scalar field :math:`u`.

    :param u: A scalar field :math:`u`, must have shape (n_samples, 1).
    :type u: `torch.Tensor`
    :param r: A vector of :math:`r`-coordinate values, must have shape (n_samples, 1).
    :type r: `torch.Tensor`
    :param theta: A vector of :math:`\theta`-coordinate values, must have shape (n_samples, 1).
    :type theta: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :return: The laplacian evaluated at :math:`(r, \theta, \phi)`, with shape (n_samples, 1).
    :rtype: `torch.Tensor`
    """
    d_r = lambda u: diff(u, r)
    d_theta = lambda u: diff(u, theta)
    d_phi = lambda u: diff(u, phi)

    lap_r = d_r(d_r(u) * r ** 2) / r ** 2
    lap_theta = d_theta(d_theta(u) * sin(theta)) / (r ** 2 * sin(theta))
    lap_phi = d_phi(d_phi(u)) / (r ** 2 * sin(theta) ** 2)
    return lap_r + lap_theta + lap_phi


def spherical_vector_laplacian(u_r, u_theta, u_phi, r, theta, phi):
    r"""Derives and evaluates the spherical laplacian of a spherical vector field :math:`u`.

    :param u_r: The :math:`r`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_r: `torch.Tensor`
    :param u_theta: The :math:`\theta`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_theta: `torch.Tensor`
    :param u_phi: The :math:`\phi`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_phi: `torch.Tensor`
    :param r: A vector of :math:`r`-coordinate values, must have shape (n_samples, 1).
    :type r: `torch.Tensor`
    :param theta: A vector of :math:`\theta`-coordinate values, must have shape (n_samples, 1).
    :type theta: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :return: The laplacian evaluated at :math:`(r, \theta, \phi)`, with shape (n_samples, 1).
    :rtype: `torch.Tensor`
    """
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
