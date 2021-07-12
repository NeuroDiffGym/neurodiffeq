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

    ur_dth, ur_dph = grad(u_r, theta, phi)
    uth_dr, uth_dph = grad(u_theta, r, phi)
    uph_dr, uph_dth = grad(u_phi, r, theta)
    csc_th = 1 / sin(theta)
    r_inv = 1 / r

    curl_r = r_inv * (uph_dth + (u_phi * cos(theta) - uth_dph) * csc_th)
    curl_th = r_inv * (csc_th * ur_dph - u_phi) - uph_dr
    curl_ph = uth_dr + r_inv * (u_theta - ur_dth)

    return curl_r, curl_th, curl_ph


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
    u_dr, u_dth, u_dph = grad(u, r, theta, phi)
    r_inv = 1 / r
    return u_dr, u_dth * r_inv, u_dph * r_inv / sin(theta)


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
    sin_th = sin(theta)
    return (diff(u_r * r ** 2, r) / r + (diff(u_theta * sin_th, theta) + diff(u_phi, phi)) / sin_th) / r


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
    u_dr, u_dth, u_dph = grad(u, r, theta, phi)
    sin_th = sin(theta)
    r2 = r ** 2

    return (diff(r2 * u_dr, r) + diff(sin_th * u_dth, theta) / sin_th + diff(u_dph, phi) / sin_th ** 2) / r2


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
    ur_dr, ur_dth, ur_dph = grad(u_r, r, theta, phi)
    uth_dr, uth_dth, uth_dph = grad(u_theta, r, theta, phi)
    uph_dr, uph_dth, uph_dph = grad(u_phi, r, theta, phi)
    sin_th, cos_th = sin(theta), cos(theta)
    sin2_th = sin_th ** 2
    r2 = r ** 2

    scalar_lap_r = (diff(r2 * ur_dr, r) + diff(sin_th * ur_dth, theta) / sin_th + diff(ur_dph, phi) / sin2_th) / r2
    scalar_lap_th = (diff(r2 * uth_dr, r) + diff(sin_th * uth_dth, theta) / sin_th + diff(uth_dph, phi) / sin2_th) / r2
    scalar_lap_ph = (diff(r2 * uph_dr, r) + diff(sin_th * uph_dth, theta) / sin_th + diff(uph_dph, phi) / sin2_th) / r2

    vec_lap_r = scalar_lap_r - 2 * (u_r + uth_dth + (cos_th * u_theta + uph_dph) / sin_th) / r2
    vec_lap_th = scalar_lap_th + (2 * ur_dth - (u_theta + 2 * cos_th * uph_dph) / sin2_th) / r2
    vec_lap_ph = scalar_lap_ph + ((2 * cos_th * uth_dph - u_phi) / sin_th + 2 * ur_dph) / (r2 * sin_th)

    return vec_lap_r, vec_lap_th, vec_lap_ph
