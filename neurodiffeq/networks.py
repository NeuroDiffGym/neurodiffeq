import torch
import torch.nn as nn
from warnings import warn


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
    :param actv: the activation layer constructor after each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: class
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


class Resnet(nn.Module):
    """A residual network with a trainable linear skip connection between input and output

    :param n_input_units: number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_input_units: number of units in the output layer, defaults to 1.
    :type n_input_units: int
    :param n_hidden_units: number of hidden units in each hidden layer, defaults to 32.
    :type n_hidden_units: int
    :param n_hidden_layers: number of hidden layers, defaults to 1.
    :type n_hidden_layers: int
    :param actv: the activation layer constructor after each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: class
    """
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=32, n_hidden_layers=1, actv=nn.Tanh):
        super(Resnet, self).__init__()

        self.residual = FCNN(
            n_input_units=n_input_units,
            n_output_units=n_output_units,
            n_hidden_units=n_hidden_units,
            n_hidden_layers=n_hidden_layers,
            actv=actv,
        )
        self.skip_connection = nn.Linear(n_input_units, n_output_units, bias=False)

    def forward(self, t):
        x = self.skip_connection(t) + self.residual(t)
        return x


class MonomialNN(nn.Module):
    def __init__(self, n_input_units, degrees=1):
        super(MonomialNN, self).__init__()

        self.n_input_units = n_input_units
        if isinstance(degrees, int):
            degrees = [d for d in range(1, degrees + 1)]
        self.degrees = tuple(degrees)

        if len(self.degrees) == 0:
            raise ValueError(f"No degrees used, check `degrees` argument again")
        if 0 in degrees:
            warn("One of the degrees is 0 which might introduce redundant features")
        if len(set(self.degrees)) < len(self.degrees):
            warn(f"Duplicate degrees found: {self.degrees}")

    def forward(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.n_input_units:
            raise ValueError(f"Expected input shape (-1, {self.n_input_units}), got {x.shape} instead")
        return torch.cat([x ** d for d in self.degrees], dim=1)

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"n_input_units={self.n_input_units}" \
               f", " \
               f"degrees={self.degrees}" \
               f")"

    def __str__(self):
        return self.__repr__()


class SinActv(nn.Module):
    """The sin activation function.
    """

    def __init__(self):
        """Initializer method.
        """
        super().__init__()

    def forward(self, input_):
        return torch.sin(input_)
