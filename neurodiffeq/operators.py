import torch
from torch import sin, cos
from torch import autograd
from .neurodiffeq import safe_diff as diff


def _split_u_x(*us_xs):
    if len(us_xs) == 0 or len(us_xs) % 2 != 0:
        raise RuntimeError("Number of us and xs must be equal and positive")
    us = us_xs[:len(us_xs) // 2]
    xs = us_xs[len(us_xs) // 2:]
    return us, xs


def grad(u, *xs):
    r"""Gradient of tensor u with respect to a tuple of tensors xs.
    Given :math:`u` and :math:`x_1`, ..., :math:`x_n`, the function returns
    :math:`\frac{\partial u}{\partial x_1}`, ..., :math:`\frac{\partial u}{\partial x_n}`

    :param u: The :math:`u` described above.
    :type u: `torch.Tensor`
    :param xs: The sequence of :math:`x_i` described above.
    :type xs: `torch.Tensor`
    :return: A tuple of :math:`\frac{\partial u}{\partial x_1}`, ..., :math:`\frac{\partial u}{\partial x_n}`
    :rtype: List[`torch.Tensor`]
    """
    grads = []
    for x, g in zip(xs, autograd.grad(u, xs, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)):
        if g is None:
            grads.append(torch.zeros_like(x, requires_grad=True))
        else:
            grads.append(g.requires_grad_(True))
    return grads


def div(*us_xs):
    r"""Derives and evaluates the divergence of a :math:`n`-dimensional vector field :math:`\mathbf{u}`
    with respect to :math:`\mathbf{x}`.

    :param us_xs:
        The input must have :math:`2n` tensors, each of shape (n_samples, 1)
        with the former :math:`n` tensors being the entries of :math:`u`
        and the latter :math:`n` tensors being the entries of :math:`x`.
    :type us_xs: `torch.Tensor`
    :return: The divergence evaluated at :math:`x`, with shape (n_samples, 1).
    :rtype: `torch.Tensor`
    """
    us, xs = _split_u_x(*us_xs)
    return sum(diff(u, x) for u, x in zip(us, xs))


def curl(u_x, u_y, u_z, x, y, z):
    r"""Derives and evaluates the curl of a vector field :math:`\mathbf{u}` in three dimensional cartesian coordinates.

    :param u_x: The :math:`x`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_x: `torch.Tensor`
    :param u_y: The :math:`y`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_y: `torch.Tensor`
    :param u_z: The :math:`z`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_z: `torch.Tensor`
    :param x: A vector of :math:`x`-coordinate values, must have shape (n_samples, 1).
    :type x: `torch.Tensor`
    :param y: A vector of :math:`y`-coordinate values, must have shape (n_samples, 1).
    :type y: `torch.Tensor`
    :param z: A vector of :math:`z`-coordinate values, must have shape (n_samples, 1).
    :type z: `torch.Tensor`
    :return: The :math:`x`, :math:`y`, and :math:`z` components of the curl, each with shape (n_samples, 1).
    :rtype: tuple[`torch.Tensor`]
    """
    dxy, dxz = grad(u_x, y, z)
    dyx, dyz = grad(u_y, x, z)
    dzx, dzy = grad(u_z, x, y)

    return dzy - dyz, dxz - dzx, dyx - dxy


def laplacian(u, *xs):
    r"""Derives and evaluates the laplacian of a scalar field :math:`u`
    with respect to :math:`\mathbf{x}=[x_1, x_2, \dots]`

    :param u: A scalar field :math:`u`, must have shape (n_samples, 1).
    :type u: `torch.Tensor`
    :param xs: The sequence of :math:`x_i` described above. Each with shape (n_samples, 1)
    :type xs: `torch.Tensor`
    :return: The laplacian of :math:`u` evaluated at :math:`\mathbf{x}`, with shape (n_samples, 1).
    :rtype: `torch.Tensor`
    """
    gs = grad(u, *xs)
    return sum(diff(g, x) for g, x in zip(gs, xs))


def vector_laplacian(u_x, u_y, u_z, x, y, z):
    r"""Derives and evaluates the vector laplacian of a vector field :math:`\mathbf{u}`
    in three dimensional cartesian coordinates.

    :param u_x: The :math:`x`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_x: `torch.Tensor`
    :param u_y: The :math:`y`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_y: `torch.Tensor`
    :param u_z: The :math:`z`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_z: `torch.Tensor`
    :param x: A vector of :math:`x`-coordinate values, must have shape (n_samples, 1).
    :type x: `torch.Tensor`
    :param y: A vector of :math:`y`-coordinate values, must have shape (n_samples, 1).
    :type y: `torch.Tensor`
    :param z: A vector of :math:`z`-coordinate values, must have shape (n_samples, 1).
    :type z: `torch.Tensor`
    :return:
        Components of vector laplacian of :math:`\mathbf{u}` evaluated at :math:`\mathbf{x}`,
        each with shape (n_samples, 1).
    :rtype: tuple[`torch.Tensor`]
    """
    return laplacian(u_x, x, y, z), laplacian(u_y, x, y, z), laplacian(u_z, x, y, z)


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
