import torch
import torch.nn as nn
from .spherical_harmonics import RealSphericalHarmonics

class FCNN(nn.Module):
    """A fully connected neural network.

    :param n_input_units: number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_input_units: number of units in the output layer, defaults to 1.
    :type n_input_units: int
    :param n_hidden_units: number of hidden units in each hidden layer, defaults to 32.
    :type n_hidden_units: int
    :param n_hidden_layers: number of hidden layers, defaults to 1.
    :type n_hidden_layers: int
    :param actv: the activation layer used in each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: `torch.nn.Module`
    """
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32, n_hidden_layers=1,
                 actv=nn.Tanh):
        r"""Initializer method.
        """
        super(FCNN, self).__init__()

        layers = []
        layers.append(nn.Linear(n_input_units, n_hidden_units))
        layers.append(actv())
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(actv())
        layers.append(nn.Linear(n_hidden_units, n_output_units))
        self.NN = torch.nn.Sequential(*layers)

    def forward(self, t):
        x = self.NN(t)
        return x

class SinActv(nn.Module):
    """The sin activation function.
    """
    def __init__(self):
        """Initializer method.
        """
        super().__init__()

    def forward(self, input_):
        return torch.sin(input_)


class SphericalHarmonicsNN(nn.Module):
    """A network that takes in spherical coordinates and returns a linear combination of spherical harmonics_fn

    The network takes in :math:`(r, \\theta, \\phi)` and returns the inner product :math:`R(r) \cdot Y(\\theta, \\phi)` where
        :math:`Y` is a vector of finitely many components of the spherical harmonics_fn, and
        :math:`R` is a vector of coefficients of each corresponding component

    :param r_net: network for approximating R(r), input should be 1-d, output should have the dimension of the number of spherical harmonic elements
    :type r_net: nn.Module
    :param max_degree: Highest degree for spherical harmonics_fn; default is 4
    :type max_degree: int
    """

    def __init__(self, r_net=None, max_degree=4):
        super(SphericalHarmonicsNN, self).__init__()
        self.harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)
        if r_net:
            self.r_net = r_net
        else:
            self.r_net = FCNN(n_input_units=1, n_output_units=(max_degree + 1) ** 2)

    def forward(self, inp: torch.Tensor):
        if len(inp.shape) != 2 or inp.shape[1] != 3:
            raise ValueError(f'Illegal input shape {inp.shape}, must be (N, 3)')
        r = torch.stack((inp[:, 0],), dim=1)
        theta = inp[:, 1]
        phi = inp[:, 2]
        coefficients = self.r_net(r)
        harmonics = self.harmonics_fn(theta, phi)
        return torch.sum(coefficients * harmonics, dim=1)
