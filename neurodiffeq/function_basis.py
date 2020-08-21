import torch
import torch.nn as nn
import numpy as np
from .neurodiffeq import diff
from math import factorial


class LegendrePolynomial:
    def __init__(self, degree):
        self.degree = degree

        if degree == 0:
            self.p = lambda x: torch.ones_like(x, requires_grad=x.requies_grad)
        elif degree == 1:
            self.p = lambda x: x * 1
        elif degree == 2:
            self.p = lambda x: (3 * x ** 2 - 1) / 2
        elif degree == 3:
            self.p = lambda x: (5 * x ** 3 - 3 * x) / 2
        elif degree == 4:
            self.p = lambda x: (35 * x ** 4 - 30 * x ** 2 + 3) / 8
        elif degree == 5:
            self.p = lambda x: (63 * x ** 5 - 70 * x ** 3 + 15 * x) / 8
        elif degree == 6:
            self.p = lambda x: (231 * x ** 6 - 315 * x ** 4 + 105 * x ** 2 - 5) / 16
        elif degree == 7:
            self.p = lambda x: (429 * x ** 7 - 693 * x ** 5 + 315 * x ** 3
                                - 35 * x) / 16
        elif degree == 8:
            self.p = lambda x: (6435 * x ** 8 - 12012 * x ** 6 + 6930 * x ** 4
                                - 1260 * x ** 2 + 35) / 128
        elif degree == 9:
            self.p = lambda x: (12155 * x ** 9 - 25740 * x ** 7 + 18018 * x ** 5
                                - 4620 * x ** 3 + 315 * x) / 128
        elif degree == 10:
            self.p = lambda x: (46189 * x ** 10 - 109395 * x ** 8 + 90090 * x ** 6
                                - 30030 * x ** 4 + 3465 * x ** 2 - 63) / 256
        else:
            normalizer = factorial(degree) * (2 ** degree)
            self.p = lambda x: diff((x ** 2 - 1) ** degree, x, order=degree) / normalizer

    def __call__(self, x):
        return self.p(x)


class FunctionBasis(nn.Module):
    pass


class CustomBasis(FunctionBasis):
    def __init__(self, fns, input_arg_count):
        super(CustomBasis, self).__init__()
        self.fns = fns
        self.input_arg_count = input_arg_count

    def forward(self, *xs):
        if len(xs) != self.input_arg_count:
            raise TypeError(f"{self.__class__.__name__}.forward() accepts {self.input_arg_count} argument(s), "
                            f"got {len(xs)} argument(s) instead.")
        return torch.cat([fn(*xs) for fn in self.fns], dim=1)


class LegendreBasis(FunctionBasis):
    def __init__(self, max_degree):
        super(LegendreBasis, self).__init__()
        polynomials = [LegendrePolynomial(d) for d in range(max_degree + 1)]
        self.basis_module = CustomBasis(polynomials, 1)

    def forward(self, x):
        return self.basis_module(x)


class ZeroOrderSphericalHarmonics(FunctionBasis):
    def __init__(self, max_degree):
        super(ZeroOrderSphericalHarmonics, self).__init__()
        coefficients = [np.sqrt((2 * l + 1) / (4 * np.pi)) for l in range(max_degree + 1)]
        polynomials = [LegendrePolynomial(d) for d in range(max_degree + 1)]
        fns = [
            lambda th, ph: c * fn(torch.cos(ph))
            for c, fn in zip(coefficients, polynomials)
        ]
        self.basis_module = CustomBasis(fns, 2)

    def forward(self, x):
        return self.basis_module(x)


class BasisOperator:
    def __init__(self, input_arg_count, output_arg_count=1):
        self.input_arg_count = input_arg_count
        self.output_arg_count = output_arg_count


class ZeroOrderSphericalHarmonicsLaplacian(BasisOperator):
    def __init__(self, max_degree):
        super(ZeroOrderSphericalHarmonicsLaplacian, self).__init__(
            input_arg_count=1,
            output_arg_count=3,
        )
        self.harmonics_fn = ZeroOrderSphericalHarmonics(max_degree)
        laplacian_coefficients = [-l * (l + 1) for l in range(max_degree + 1)]
        self.laplacian_coefficients = torch.tensor(laplacian_coefficients, dtype=torch.float)

    def __call__(self, base_coeffs, r, theta, phi):
        radial_components = [
            diff(base_coeffs[:, j] * r[:, 0], r, order=2)
            for j in range(base_coeffs.shape[1])
        ]
        radial_components = torch.cat(radial_components, dim=1) / r

        angular_components = self.laplacian_coefficients * base_coeffs / r ** 2
        products = (radial_components + angular_components) * self.harmonics_fn(theta, phi)
        return torch.sum(products, dim=1, keepdim=True)
