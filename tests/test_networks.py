import torch
import numpy as np
import torch.nn as nn
from neurodiffeq.networks import FCNN
from neurodiffeq.networks import Resnet
from neurodiffeq.networks import MonomialNN
from neurodiffeq.networks import SinActv

MAGIC = 42
torch.manual_seed(MAGIC)
np.random.seed(MAGIC)

N_TESTS = 5


def _test_shape(n_samples, n_features_in, n_features_out, model_constructor, *args, **kwargs):
    net = model_constructor(*args, **kwargs)
    x = torch.rand(n_samples, n_features_in).requires_grad_(True)
    y = net(x)
    assert y.shape == (n_samples, n_features_out)


def test_fcnn():
    for _ in range(N_TESTS):
        n_samples = np.random.randint(30, 100)
        n_features_in = np.random.randint(1, 5)
        n_features_out = np.random.randint(1, 5)
        n_hidden_units = np.random.randint(30, 60)
        n_hidden_layers = np.random.randint(0, 4)
        _test_shape(
            n_samples, n_features_in, n_features_out, FCNN,
            n_input_units=n_features_in,
            n_output_units=n_features_out,
            n_hidden_units=n_hidden_units,
            n_hidden_layers=n_hidden_layers,
        )

            n_hidden_layers=n_hidden_layers,
        )