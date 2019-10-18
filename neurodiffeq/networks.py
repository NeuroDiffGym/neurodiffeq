import torch
import torch.nn as nn

class FCNN(nn.Module):
    """A fully connected neural network.

    :param n_input_units: number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_hidden_units: number of hidden units in each hidden layer, defaults to 32.
    :type n_hidden_units: int
    :param n_hidden_layers: number of hidden layers, defaults to 1.
    :type n_hidden_layers: int
    :param actv: the activation layer used in each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: `torch.nn.Module`
    """
    def __init__(self, n_input_units=1, n_hidden_units=32, n_hidden_layers=1, 
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
        layers.append(nn.Linear(n_hidden_units, 1))
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