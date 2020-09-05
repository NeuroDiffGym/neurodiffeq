import torch
import numpy as np
from torch import sin, cos
from .neurodiffeq import diff
from .version_utils import warn_deprecate_class
from scipy.special import legendre
from abc import ABC, abstractmethod


class LegendrePolynomial:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = legendre(degree)

    def __call__(self, x):
        if self.degree == 0:
            return torch.ones_like(x, requires_grad=x.requires_grad)
        elif self.degree == 1:
            return x * 1
        else:
            return sum(coeff * x ** (self.degree - i) for i, coeff in enumerate(self.coefficients))


class FunctionBasis(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BasisOperator(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class CustomBasis(FunctionBasis):
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, *xs):
        return torch.cat([fn(*xs) for fn in self.fns], dim=1)


class LegendreBasis(FunctionBasis):
    def __init__(self, max_degree):
        polynomials = [LegendrePolynomial(d) for d in range(max_degree + 1)]
        self.basis_module = CustomBasis(polynomials)

    def __call__(self, x):
        return self.basis_module(x)


class ZonalSphericalHarmonics(FunctionBasis):
    def __init__(self, max_degree):
        self.max_degree = max_degree
        coefficients = [np.sqrt((2 * l + 1) / (4 * np.pi)) for l in range(max_degree + 1)]
        polynomials = [LegendrePolynomial(d) for d in range(max_degree + 1)]

        # The `c=c` and `fn=fn` in the lambda is needed due this issue:
        # https://stackoverflow.com/questions/28268439/python-list-comprehension-with-lambdas
        fns = [
            lambda theta, c=c, fn=fn: fn(torch.cos(theta)) * c
            for c, fn in zip(coefficients, polynomials)
        ]
        self.basis_module = CustomBasis(fns)

    def __call__(self, theta, phi):
        return self.basis_module(theta)


ZeroOrderSphericalHarmonics = warn_deprecate_class(ZonalSphericalHarmonics)


class ZonalSphericalHarmonicsLaplacian(BasisOperator):
    def __init__(self, max_degree):
        self.harmonics_fn = ZonalSphericalHarmonics(max_degree)
        laplacian_coefficients = [-l * (l + 1) for l in range(max_degree + 1)]
        self.laplacian_coefficients = torch.tensor(laplacian_coefficients, dtype=torch.float)

    def __call__(self, base_coeffs, r, theta, phi):
        coeffs_times_r = base_coeffs * r
        radial_components = [
            diff(coeffs_times_r[:, j], r, order=2) for j in range(base_coeffs.shape[1])
        ]
        radial_components = torch.cat(radial_components, dim=1) / r

        angular_components = self.laplacian_coefficients * base_coeffs / r ** 2
        products = (radial_components + angular_components) * self.harmonics_fn(theta, phi)
        return torch.sum(products, dim=1, keepdim=True)


ZeroOrderSphericalHarmonicsLaplacian = warn_deprecate_class(ZonalSphericalHarmonicsLaplacian)


def _get_real_fourier_term(degree, sine=True):
    if degree == 0:
        return lambda th: torch.ones_like(th) * 0.5  # the coefficient 0.5 is to make the series orthonormal
    elif sine:
        return lambda th: sin(degree * th)
    else:
        return lambda th: cos(degree * th)


class RealFourierSeries(FunctionBasis):
    """
    :param max_degree: highest degree for the fourier series
    :type max_degree: int
    """

    def __init__(self, max_degree=12):
        harmonics = []
        self.max_degree = max_degree
        for degree in range(self.max_degree + 1):
            if degree == 0:
                harmonics.append(_get_real_fourier_term(0))
            else:
                harmonics.append(_get_real_fourier_term(degree, sine=True))
                harmonics.append(_get_real_fourier_term(degree, sine=False))
        self.basis_module = CustomBasis(harmonics)

    def __call__(self, phi):
        """compute the value of each fourier component
        :param phi: a vector of angles, must have shape (-1, 1)
        :type phi: `torch.Tensor`
        :return: fourier series evaluated at each angle, will be of shape (-1, n_components)
        :rtype: `torch.Tensor`
        """
        return self.basis_module(phi)


class FourierLaplacian(BasisOperator):
    """A Laplacian operator (in polar coordinates) acting on :math:`\\sum_i R_i(r)F(\\phi)}` where :math:`F` is a Fourier component
    :param max_degree: highest degree for the fourier series
    :type max_degree: int
    """

    def __init__(self, max_degree=12):
        self.harmonics_fn = RealFourierSeries(max_degree=max_degree)
        laplacian_coefficients = [0] + [- deg ** 2 for deg in range(1, max_degree + 1) for sign in range(2)]
        self.laplacian_coefficients = torch.tensor(laplacian_coefficients, dtype=torch.float)

    def __call__(self, R, r, phi):
        """calculates laplacian (in polar coordinates) of :math:`\\sum_i R_i(r)F_i(\\phi)`
        :param R: coefficients of fourier series; should depend on r in general; must be of shape (n_samples, 2 * max_degree + 1)
        :type R: torch.Tensor
        :param r: radius corresponding to `R`, must be of shape (n_samples, 1)
        :type r: torch.Tensor
        :param phi: angles fed to the Fourier series, must be of the same shape as `r`
        :type phi: torch.Tensor
        :return: values of Laplacian (in polar coordinates) of shape (n_samples, 1)
        :rtype: torch.Tensor
        """
        # We would hope to do `radial_component = diff(R, r) / r + diff(R, r, order=2)`
        # But because of this issue https://github.com/odegym/neurodiffeq/issues/44#issuecomment-594998619,
        # we have to separate columns in R, compute derivatives, and manually concatenate them back together
        radial_component = torch.cat([
            diff(R[:, j], r) / r + diff(R[:, j], r, order=2) for j in range(R.shape[1])
        ], dim=1)

        angular_component = self.laplacian_coefficients * R / r ** 2
        products = (radial_component + angular_component) * self.harmonics_fn(phi)
        return torch.sum(products, dim=1, keepdim=True)
