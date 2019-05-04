import torch
import torch.nn as nn

class FCNN(nn.Module):
    """
    A fully connected neural network.
    """
    def __init__(self, n_input_units=1, n_hidden_units=32, n_hidden_layers=1, 
                 actv=nn.Tanh):
        """
        :param n_input_units: number of units in the input layer
        :param n_hidden_units: number of hidden units in each hidden layer
        :param n_hidden_layers: number of hidden layers
        :param actv: the activation layer used in each hidden layer
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
    def __init__(self):
        super().__init__()
        
    def forward(self, input_):
        return torch.sin(input_)