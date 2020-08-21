import torch
import torch.nn as nn
from torch import sin, cos
from .neurodiffeq import diff


def get_fourier_series(degree, sine=True):
    if degree == 0:
        return lambda th: torch.ones_like(th) * 0.5  # the coefficient 0.5 is to make the series orthonormal
    elif sine:
        return lambda th: sin(degree * th)
    else:
        return lambda th: cos(degree * th)


def get_cylindrical_fourier_series(degree, sine=True):
    fourier = get_fourier_series(degree, sine)
    return lambda z, ph: fourier(ph)


class RealFourierSeries(nn.Module):
    """an array of orthonormal fourier series; There is no trainable parameters in this module
    :param max_degree: highest degree for the fourier series
    :type max_degree: int
    """

    def __init__(self, max_degree=12):
        super(RealFourierSeries, self).__init__()
        self.harmonics = []
        self.max_degree = max_degree
        for degree in range(self.max_degree + 1):
            if degree == 0:
                self.harmonics.append(get_cylindrical_fourier_series(0))
            else:
                self.harmonics.append(get_cylindrical_fourier_series(degree, sine=True))
                self.harmonics.append(get_cylindrical_fourier_series(degree, sine=False))

    def forward(self, z, phi):
        """ compute the value of each component evaluated at each point
        :param z: z in cylindrical coordinates, must have shape (-1, 1)
        :type z: `torch.Tensor`
        :param phi: phis in cynlindrical coordinates, must have the same shape as z
        :type phi: `torch.Tensor`
        :return: fourier series evaluated at each point, will be of shape (-1, n_components)
        :rtype: `torch.Tensor`
        """
        if len(z.shape) != 2 or z.shape[1] != 1:
            raise ValueError(f'z must be of shape (-1, 1); got {z.shape}')
        if z.shape != phi.shape:
            raise ValueError(f'z and phi must be of the same shape; got f{z.shape} and f{phi.shape}')
        components = [Y(z, phi) for Y in self.harmonics]
        return torch.cat(components, dim=1)


class FourierLaplacian:
    def __init__(self, max_degree=12):
        self.harmonics_fn = RealFourierSeries(max_degree=max_degree)
        laplacian_coefficients = [0] + [- deg ** 2 for deg in range(1, max_degree+1) for sign in range(2)]
        self.laplacian_coefficients = torch.tensor(laplacian_coefficients).type(torch.float)

    def __call__(self, R, r, z, phi):
        # We would hope to do `radial_component = diff(R, r) / r + diff(R, r, order=2)`
        # But because of this issue https://github.com/odegym/neurodiffeq/issues/44#issuecomment-594998619,
        # we have to separate columns in R, compute derivatives, and manually concatenate them back together
        radial_component = torch.cat([
            diff(R[:, j], r) / r + diff(R[:, j], r, order=2) for j in range(R.shape[1])
        ], dim=1)

        angular_component = self.laplacian_coefficients * R / r ** 2
        products = (radial_component + angular_component) * self.harmonics_fn(z, phi)
        return torch.sum(products, dim=1, keepdim=True)
