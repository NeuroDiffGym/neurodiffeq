import torch
import torch.nn as nn
from torch import sin, cos

# List of real spherical harmonics (unnormalized) with degree l<=4; see following link
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics

# l = 0
Y0_0 = lambda th, ph: torch.ones_like(th)
# l = 1
Y1n1 = lambda th, ph: sin(th) * sin(ph)
Y1_0 = lambda th, ph: cos(th)
Y1p1 = lambda th, ph: sin(th) * cos(ph)
# l = 2
Y2n2 = lambda th, ph: sin(th) ** 2 * sin(ph) * cos(ph)
Y2n1 = lambda th, ph: sin(th) * cos(th) * sin(ph)
Y2_0 = lambda th, ph: 2 * cos(th) ** 2 - sin(th) ** 2
Y2p1 = lambda th, ph: sin(th) * cos(th) * cos(ph)
Y2p2 = lambda th, ph: sin(th) ** 2 * cos(2 * ph)
# l = 3
Y3n3 = lambda th, ph: sin(th) ** 3 * (3 * cos(ph) ** 2 * sin(ph) - sin(ph) ** 3)
Y3n2 = lambda th, ph: sin(th) ** 2 * cos(th) * cos(ph) * sin(ph)
Y3n1 = lambda th, ph: sin(th) * (4 * cos(th) ** 2 - sin(th) ** 2) * sin(ph)
Y3_0 = lambda th, ph: 2 * cos(th) ** 3 - 3 * cos(th) * sin(th) ** 2
Y3p1 = lambda th, ph: sin(th) * (4 * cos(th) ** 2 - sin(th) ** 2) * cos(ph)
Y3p2 = lambda th, ph: cos(th) * sin(th) ** 2 * cos(2 * ph)
Y3p3 = lambda th, ph: sin(th) ** 3 * (cos(ph) ** 3 - 3 * sin(ph) ** 2 * cos(ph))
# l = 4
Y4n4 = lambda th, ph: sin(th) ** 4 * (sin(ph) * cos(ph) * cos(2 * ph))
Y4n3 = lambda th, ph: sin(th) ** 3 * cos(th) * (3 * cos(ph) ** 2 * sin(ph) - sin(ph) ** 3)
Y4n2 = lambda th, ph: sin(th) ** 2 * (sin(ph) * cos(ph)) * (7 * cos(th) ** 2 - 1)
Y4n1 = lambda th, ph: sin(th) * cos(th) * sin(ph) * (7 * cos(th) ** 2 - 3)
Y4_0 = lambda th, ph: 35 * cos(th) ** 4 - 30 * cos(th) ** 2 + 3
Y4p1 = lambda th, ph: sin(th) * cos(th) * cos(ph) * (7 * cos(th) ** 2 - 3)
Y4p2 = lambda th, ph: sin(th) ** 2 * cos(2 * ph) * (7 * cos(th) ** 2 - 1)
Y4p3 = lambda th, ph: sin(th) ** 3 * cos(th) * (cos(ph) ** 3 - 3 * cos(ph) * sin(ph) ** 2)
Y4p4 = lambda th, ph: sin(th) ** 4 * (cos(ph) ** 4 - 6 * cos(ph) ** 2 * sin(ph) ** 2 + sin(ph) ** 4)


class RealSphericalHarmonics(nn.Module):
    """an array of (unnormalized) spherical harmonics_fn of degree from -l to +l
    There is no trainable parameters in this module
    :param max_degree: highest degree (currently only supports l<=4) for the spherical harmonics_fn
    :type max_degree: int
    """

    def __init__(self, max_degree=4):
        super(RealSphericalHarmonics, self).__init__()
        self.harmonics = []
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

    def forward(self, theta, phi):
        if len(theta.shape) >= 2 or len(phi.shape) >= 2:
            raise ValueError(f'theta/phi must be both of shape (n,); got f{theta.shape} and f{phi.shape}')
        components = [Y(theta, phi) for Y in self.harmonics]
        return torch.stack(components, dim=1)
