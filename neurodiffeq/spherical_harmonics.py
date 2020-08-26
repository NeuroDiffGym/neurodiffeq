import torch
import torch.nn as nn
from torch import sin, cos
from neurodiffeq.neurodiffeq import diff

# List of real spherical harmonics (normalized) with degree l<=4; see following link
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
# Note that the normalization term doesn't include the factor :math:`\\sqrt{\\frac{1}{\\pi}}`
# Correctness of these lambda functions are tested in `test_pde_spherical`

# l = 0
Y0_0 = lambda th, ph: torch.ones_like(th) * 0.5
# l = 1
Y1n1 = lambda th, ph: sin(th) * sin(ph) * 0.866025404
Y1_0 = lambda th, ph: cos(th) * 0.866025404
Y1p1 = lambda th, ph: sin(th) * cos(ph) * 0.866025404
# l = 2
Y2n2 = lambda th, ph: sin(th) ** 2 * sin(ph) * cos(ph) * 1.936491673
Y2n1 = lambda th, ph: sin(th) * cos(th) * sin(ph) * 1.936491673
Y2_0 = lambda th, ph: (2 * cos(th) ** 2 - sin(th) ** 2) * 0.559016994
Y2p1 = lambda th, ph: sin(th) * cos(th) * cos(ph) * 1.936491673
Y2p2 = lambda th, ph: sin(th) ** 2 * cos(2 * ph) * 0.968245837
# l = 3
Y3n3 = lambda th, ph: sin(th) ** 3 * (3 * cos(ph) ** 2 * sin(ph) - sin(ph) ** 3) * 1.045825033
Y3n2 = lambda th, ph: sin(th) ** 2 * cos(th) * cos(ph) * sin(ph) * 5.123475383
Y3n1 = lambda th, ph: sin(th) * (4 * cos(th) ** 2 - sin(th) ** 2) * sin(ph) * 0.810092587
Y3_0 = lambda th, ph: (2 * cos(th) ** 3 - 3 * cos(th) * sin(th) ** 2) * 0.661437828
Y3p1 = lambda th, ph: sin(th) * (4 * cos(th) ** 2 - sin(th) ** 2) * cos(ph) * 0.810092587
Y3p2 = lambda th, ph: cos(th) * sin(th) ** 2 * cos(2 * ph) * 2.561737691
Y3p3 = lambda th, ph: sin(th) ** 3 * (cos(ph) ** 3 - 3 * sin(ph) ** 2 * cos(ph)) * 1.045825033
# l = 4
Y4n4 = lambda th, ph: sin(th) ** 4 * (sin(ph) * cos(ph) * cos(2 * ph)) * 4.437059837
Y4n3 = lambda th, ph: sin(th) ** 3 * cos(th) * (3 * cos(ph) ** 2 * sin(ph) - sin(ph) ** 3) * 3.1374751
Y4n2 = lambda th, ph: sin(th) ** 2 * (sin(ph) * cos(ph)) * (7 * cos(th) ** 2 - 1) * 1.677050983
Y4n1 = lambda th, ph: sin(th) * cos(th) * sin(ph) * (7 * cos(th) ** 2 - 3) * 1.185854123
Y4_0 = lambda th, ph: (35 * cos(th) ** 4 - 30 * cos(th) ** 2 + 3) * 0.1875
Y4p1 = lambda th, ph: sin(th) * cos(th) * cos(ph) * (7 * cos(th) ** 2 - 3) * 1.185854123
Y4p2 = lambda th, ph: sin(th) ** 2 * cos(2 * ph) * (7 * cos(th) ** 2 - 1) * 0.838525492
Y4p3 = lambda th, ph: sin(th) ** 3 * cos(th) * (cos(ph) ** 3 - 3 * cos(ph) * sin(ph) ** 2) * 3.1374751
Y4p4 = lambda th, ph: sin(th) ** 4 * (cos(ph) ** 4 - 6 * cos(ph) ** 2 * sin(ph) ** 2 + sin(ph) ** 4) * 1.109264959


class RealSphericalHarmonics:
    """an array of (unnormalized) spherical harmonics_fn of degree from -l to +l
    There is no trainable parameters in this module
    :param max_degree: highest degree (currently only supports l<=4) for the spherical harmonics_fn
    :type max_degree: int
    """

    def __init__(self, max_degree=4):
        super(RealSphericalHarmonics, self).__init__()
        self.harmonics = []
        self.max_degree = max_degree
        if max_degree >= 0:
            self.harmonics += [Y0_0]
        if max_degree >= 1:
            self.harmonics += [Y1n1, Y1_0, Y1p1]
        if max_degree >= 2:
            self.harmonics += [Y2n2, Y2n1, Y2_0, Y2p1, Y2p2]
        if max_degree >= 3:
            self.harmonics += [Y3n3, Y3n2, Y3n1, Y3_0, Y3p1, Y3p2, Y3p3]
        if max_degree >= 4:
            self.harmonics += [Y4n4, Y4n3, Y4n2, Y4n1, Y4_0, Y4p1, Y4p2, Y4p3, Y4p4]
        if max_degree >= 5:
            raise NotImplementedError(f'max_degree = {max_degree} not implemented for {self.__class__.__name__} yet')

    def __call__(self, theta, phi):
        """ compute the value of each spherical harmonic component evaluated at each point
        :param theta: theta in spherical coordinates, must have shape (-1, 1)
        :type theta: `torch.Tensor`
        :param phi: phis in spherical coordinates, must have the same shape as theta
        :type phi: `torch.Tensor`
        :return: spherical harmonics evaluated at each point, will be of shape (-1, n_components)
        :rtype: `torch.Tensor`
        """
        if len(theta.shape) != 2 or theta.shape[1] != 1:
            raise ValueError(f'theta must be of shape (-1, 1); got {theta.shape}')
        if theta.shape != phi.shape:
            raise ValueError(f'theta/phi must be of the same shape; got f{theta.shape} and f{phi.shape}')
        components = [Y(theta, phi) for Y in self.harmonics]
        return torch.cat(components, dim=1)


class HarmonicsLaplacian:
    """
        Laplacian of spherical harmonics can be reduced in the following way. Using this method, we can avoid the :math:`\\frac{1}{\\sin \\theta}` singularity
        :math:```
        \\begin{aligned}
        &\\nabla^{2} R_{l, m}(r) Y_{l,m}(\\theta, \\phi)\\\\
        &=\\left(\\nabla_{r}^{2}+\\nabla_{\\theta}^{2}+\\nabla_{\\phi}^{2}\\right)\\left(R_{l, m}(r) Y_{l, m}(\\theta, \\phi)\\right)\\\\
        &=Y_{l, m} \\nabla_{r}^{2} R_{l, m}+R_{l, m}\\left(\\left(\\nabla_{\\theta}^{2}+\\nabla_{\\phi}^{2}\\right) Y_{l, m}\\right)\\\\
        &=Y_{l, m} \\nabla_{r}^{2} R_{l, m}+R_{l, m} \\frac{-l(l+1)}{r^{2}} Y_{l, m}\\\\
        &=Y_{l, m}\\left(\\nabla_{r}^{2} R_{l, m}+\\frac{-l(l+1)}{r^{2}} R_{l, m}\\right)
        \\end{aligned}```
    """

    def __init__(self, max_degree=4):
        self.harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)
        laplacian_coefficients = [-l * (l + 1) for l in range(max_degree + 1) for m in range(-l, l + 1)]
        self.laplacian_coefficients = torch.tensor(laplacian_coefficients).type(torch.float)

    def __call__(self, R, r, theta, phi):
        # We would hope to do `radial_component = diff(R * r, r, order=2) / r`
        # But because of this issue https://github.com/odegym/neurodiffeq/issues/44#issuecomment-594998619,
        # we have to separate columns in R, compute derivates, and manually concatenate them back together
        radial_component = torch.cat([diff(R[:, j] * r[:, 0], r, order=2) for j in range(R.shape[1])], dim=1) / r

        angular_component = self.laplacian_coefficients * R / r ** 2
        products = (radial_component + angular_component) * self.harmonics_fn(theta, phi)
        return torch.sum(products, dim=1, keepdim=True)
