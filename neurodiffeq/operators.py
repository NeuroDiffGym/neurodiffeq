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


def spherical_to_cartesian(r, theta, phi):
    r"""Convert spherical coordinate :math:`(r, \theta, \phi)` to cartesian coordinates :math:`(x, y, z)`.
    The input shapes of r, theta, and phi must be the same.

    :param r: The :math:`r`-component of spherical coordinates.
    :type r: `torch.Tensor`
    :param theta: The :math:`\theta`-component (polar angle) of spherical coordinates.
    :type theta: `torch.Tensor`
    :param phi: The :math:`\phi`-component (azimuthal angle) of spherical coordinates.
    :type phi: `torch.Tensor`
    :return: The :math:`x`-, :math:`y`-, and :math:`z`-component in cartesian coordinates.
    :rtype: tuple[`torch.Tensor`]
    """
    rho = r * sin(theta)
    return rho * cos(phi), rho * sin(phi), r * cos(theta)


def cartesian_to_spherical(x, y, z):
    r"""Convert cartesian coordinates :math:`(x, y, z)` to spherical coordinate :math:`(r, \theta, \phi)`.
    The input shapes of x, y, and z must be the same.
    If either the polar angle :math:`\theta` or the azimuthal angle :math:`phi` is not defined,
    the default value will be 0.

    :param x: The :math:`x`-component of cartesian coordinates.
    :type x: `torch.Tensor`
    :param y: The :math:`y`-component of cartesian coordinates.
    :type y: `torch.Tensor`
    :param z: The :math:`z`-component of cartesian coordinates.
    :type z: `torch.Tensor`
    :return: The :math:`r`-, :math:`\theta`-, and :math:`\phi`-component in spherical coordinates.
    :rtype: tuple[`torch.Tensor`]
    """
    rho2 = x ** 2 + y ** 2
    return torch.sqrt(rho2 + z ** 2), torch.atan2(torch.sqrt(rho2), z), torch.atan2(y, x)


def cylindrical_grad(u, rho, phi, z):
    r"""Derives and evaluates the cylindrical gradient of a cylindrical scalar field :math:`u`.

    :param u: A scalar field :math:`u`, must have shape (n_samples, 1).
    :type u: `torch.Tensor`
    :param rho: A vector of :math:`\rho`-coordinate values, must have shape (n_samples, 1).
    :type rho: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :param z: A vector of :math:`z`-coordinate values, must have shape (n_samples, 1).
    :type z: `torch.Tensor`
    :return: The :math:`\rho`, :math:`\phi`, and :math:`z` components of the gradient, each with shape (n_samples, 1).
    :rtype: tuple[`torch.Tensor`]
    """
    u_drho, u_dphi, u_dz = grad(u, rho, phi, z)
    return u_drho, u_dphi / rho, u_dz


def cylindrical_div(u_rho, u_phi, u_z, rho, phi, z):
    r"""Derives and evaluates the cylindrical divergence of a cylindrical vector field :math:`u`.

    :param u_rho: The :math:`\rho`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_rho: `torch.Tensor`
    :param u_phi: The :math:`\phi`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_phi: `torch.Tensor`
    :param u_z: The :math:`z`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_z: `torch.Tensor`
    :param rho: A vector of :math:`\rho`-coordinate values, must have shape (n_samples, 1).
    :type rho: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :param z: A vector of :math:`z`-coordinate values, must have shape (n_samples, 1).
    :type z: `torch.Tensor`
    :return: The divergence evaluated at :math:`(\rho, \phi, z)`, with shape (n_samples, 1).
    :rtype: `torch.Tensor`
    """
    return diff(u_rho, rho) + (u_rho + diff(u_phi, phi)) / rho + diff(u_z, z)


def cylindrical_curl(u_rho, u_phi, u_z, rho, phi, z):
    r"""Derives and evaluates the cylindrical curl of a cylindrical vector field :math:`u`.

    :param u_rho: The :math:`\rho`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_rho: `torch.Tensor`
    :param u_phi: The :math:`\phi`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_phi: `torch.Tensor`
    :param u_z: The :math:`z`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_z: `torch.Tensor`
    :param rho: A vector of :math:`\rho`-coordinate values, must have shape (n_samples, 1).
    :type rho: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :param z: A vector of :math:`z`-coordinate values, must have shape (n_samples, 1).
    :type z: `torch.Tensor`
    :return: The :math:`\rho`, :math:`\phi`, and :math:`z` components of the curl, each with shape (n_samples, 1).
    :rtype: tuple[`torch.Tensor`]
    """
    urho_dphi, urho_dz = grad(u_rho, phi, z)
    uphi_drho, uphi_dz = grad(u_phi, rho, z)
    uz_drho, uz_dphi = grad(u_z, rho, phi)

    return (
        uz_dphi / rho - uphi_dz,
        urho_dz - uz_drho,
        uphi_drho + (u_phi - urho_dphi) / rho
    )


def cylindrical_laplacian(u, rho, phi, z):
    r"""Derives and evaluates the cylindrical laplacian of a cylindrical scalar field :math:`u`.

    :param u: A scalar field :math:`u`, must have shape (n_samples, 1).
    :type u: `torch.Tensor`
    :param rho: A vector of :math:`\rho`-coordinate values, must have shape (n_samples, 1).
    :type rho: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :param z: A vector of :math:`z`-coordinate values, must have shape (n_samples, 1).
    :type z: `torch.Tensor`
    :return: The laplacian evaluated at :math:`(\rho, \phi, z)`, with shape (n_samples, 1).
    :rtype: `torch.Tensor`
    """
    u_drho, u_dphi, u_dz = grad(u, rho, phi, z)
    return diff(u_drho, rho) + u_drho / rho + diff(u_dphi, phi) / rho ** 2 + diff(u_dz, z)


def cylindrical_vector_laplacian(u_rho, u_phi, u_z, rho, phi, z):
    r"""Derives and evaluates the cylindrical laplacian of a cylindrical vector field :math:`u`.

    :param u_rho: The :math:`\rho`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_rho: `torch.Tensor`
    :param u_phi: The :math:`\phi`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_phi: `torch.Tensor`
    :param u_z: The :math:`z`-component of the vector field :math:`u`, must have shape (n_samples, 1).
    :type u_z: `torch.Tensor`
    :param rho: A vector of :math:`\rho`-coordinate values, must have shape (n_samples, 1).
    :type rho: `torch.Tensor`
    :param phi: A vector of :math:`\phi`-coordinate values, must have shape (n_samples, 1).
    :type phi: `torch.Tensor`
    :param z: A vector of :math:`z`-coordinate values, must have shape (n_samples, 1).
    :type z: `torch.Tensor`
    :return: The laplacian evaluated at :math:`(\rho, \phi, z)`, with shape (n_samples, 1).
    :rtype: `torch.Tensor`
    """
    rho2 = rho ** 2
    urho_drho, urho_dphi, urho_dz = grad(u_rho, rho, phi, z)
    uphi_drho, uphi_dphi, uphi_dz = grad(u_phi, rho, phi, z)
    uz_drho, uz_dphi, uz_dz = grad(u_z, rho, phi, z)

    scalar_lap_rho = diff(urho_drho, rho) + urho_drho / rho + diff(urho_dphi, phi) / rho ** 2 + diff(urho_dz, z)
    scalar_lap_phi = diff(uphi_drho, rho) + uphi_drho / rho + diff(uphi_dphi, phi) / rho ** 2 + diff(uphi_dz, z)
    scalar_lap_z = diff(uz_drho, rho) + uz_drho / rho + diff(uz_dphi, phi) / rho ** 2 + diff(uz_dz, z)

    return (
        scalar_lap_rho - (u_rho + 2 * uphi_dphi) / rho2,
        scalar_lap_phi + (2 * urho_dphi - u_phi) / rho2,
        scalar_lap_z,
    )


def cylindrical_to_cartesian(rho, phi, z):
    r"""Convert cylindrical coordinate :math:`(\rho, \phi, z)` to cartesian coordinates :math:`(x, y, z)`.
    The input shapes of rho, phi, and z must be the same.

    :param rho: The :math:`\rho`-component of cylindrical coordinates.
    :type rho: `torch.Tensor`
    :param phi: The :math:`\phi`-component (azimuthal angle) of cylindrical coordinates.
    :type phi: `torch.Tensor`
    :param z: The :math:`z`-component of cylindrical coordinates.
    :type z: `torch.Tensor`
    :return: The :math:`x`-, :math:`y`-, and :math:`z`-component in cartesian coordinates.
    :rtype: tuple[`torch.Tensor`]
    """
    return rho * cos(phi), rho * sin(phi), z


def cartesian_to_cylindrical(x, y, z):
    r"""Convert cartesian coordinates :math:`(x, y, z)` to cylindrical coordinate :math:`(\rho, \phi, z)`.
    The input shapes of x, y, and z must be the same.
    If the azimuthal angle :math:`phi` is undefined, the default value will be 0.

    :param x: The :math:`x`-component of cartesian coordinates.
    :type x: `torch.Tensor`
    :param y: The :math:`y`-component of cartesian coordinates.
    :type y: `torch.Tensor`
    :param z: The :math:`z`-component of cartesian coordinates.
    :type z: `torch.Tensor`
    :return: The :math:`\rho`-, :math:`\phi`-, and :math:`z`-component in cylindrical coordinates.
    :rtype: tuple[`torch.Tensor`]
    """
    return torch.sqrt(x ** 2 + y ** 2), torch.atan2(y, x), z
