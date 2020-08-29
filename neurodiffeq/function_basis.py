import torch
import torch.nn as nn
import numpy as np
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


class BasisOperator(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


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
