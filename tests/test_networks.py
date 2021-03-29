import torch
import numpy as np
import torch.nn as nn
import pytest
from neurodiffeq.networks import FCNN
from neurodiffeq.networks import Resnet
from neurodiffeq.networks import MonomialNN
from neurodiffeq.networks import SinActv
from neurodiffeq.networks import Swish

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
        hidden_units = [np.random.randint(1, 10) for _ in range(np.random.randint(2, 4))]

        with pytest.warns(FutureWarning):
            _test_shape(
                n_samples, n_features_in, n_features_out, FCNN,
                n_input_units=n_features_in,
                n_output_units=n_features_out,
                n_hidden_units=n_hidden_units,
            )
        with pytest.warns(FutureWarning):
            _test_shape(
                n_samples, n_features_in, n_features_out, FCNN,
                n_input_units=n_features_in,
                n_output_units=n_features_out,
                n_hidden_layers=n_hidden_layers,
            )
        with pytest.warns(FutureWarning):
            _test_shape(
                n_samples, n_features_in, n_features_out, FCNN,
                n_input_units=n_features_in,
                n_output_units=n_features_out,
                n_hidden_units=n_hidden_units,
                n_hidden_layers=n_hidden_layers,
            )
        with pytest.warns(FutureWarning):
            _test_shape(
                n_samples, n_features_in, n_features_out, FCNN,
                n_input_units=n_features_in,
                n_output_units=n_features_out,
                n_hidden_units=n_hidden_units,
                n_hidden_layers=n_hidden_layers,
                hidden_units=hidden_units
            )

        _test_shape(
            n_samples, n_features_in, n_features_out, FCNN,
            n_input_units=n_features_in,
            n_output_units=n_features_out,
            hidden_units=hidden_units
        )


def test_resnet():
    for _ in range(N_TESTS):
        n_samples = np.random.randint(30, 100)
        n_features_in = np.random.randint(1, 5)
        n_features_out = np.random.randint(1, 5)
        n_hidden_units = np.random.randint(30, 60)
        n_hidden_layers = np.random.randint(0, 4)
        hidden_units = [np.random.randint(1, 10) for _ in range(np.random.randint(2, 4))]
        with pytest.warns(FutureWarning):
            _test_shape(
                n_samples, n_features_in, n_features_out, Resnet,
                n_input_units=n_features_in,
                n_output_units=n_features_out,
                n_hidden_units=n_hidden_units,
            )
        with pytest.warns(FutureWarning):
            _test_shape(
                n_samples, n_features_in, n_features_out, Resnet,
                n_input_units=n_features_in,
                n_output_units=n_features_out,
                n_hidden_layers=n_hidden_layers,
            )
        with pytest.warns(FutureWarning):
            _test_shape(
                n_samples, n_features_in, n_features_out, Resnet,
                n_input_units=n_features_in,
                n_output_units=n_features_out,
                n_hidden_units=n_hidden_units,
                n_hidden_layers=n_hidden_layers,
            )
        with pytest.warns(FutureWarning):
            _test_shape(
                n_samples, n_features_in, n_features_out, Resnet,
                n_input_units=n_features_in,
                n_output_units=n_features_out,
                n_hidden_units=n_hidden_units,
                n_hidden_layers=n_hidden_layers,
                hidden_units=hidden_units
            )

        _test_shape(
            n_samples, n_features_in, n_features_out, Resnet,
            n_input_units=n_features_in,
            n_output_units=n_features_out,
            hidden_units=hidden_units
        )


def test_monomial_nn():
    ALL_DEGREES = list(range(1, N_TESTS + 1))
    ALL_DEGREES += [-d for d in ALL_DEGREES]

    for test_id in range(N_TESTS):
        degrees = np.random.choice(ALL_DEGREES, size=test_id + 1, replace=False)
        n_samples = np.random.randint(30, 100)
        n_features_in = np.random.randint(1, 5)
        net = MonomialNN(degrees)
        x = torch.rand(n_samples, n_features_in) + 0.5
        y = net(x)
        for i, d in enumerate(degrees):
            x_d = y[:, i * n_features_in: (i + 1) * n_features_in]
            assert (x_d - x ** d).abs().max() < 1e-3


def test_swish():
    x = torch.rand(10, 5)

    f = Swish()
    assert len(list(f.parameters())) == 0
    assert torch.isclose(f(x), x * torch.sigmoid(x)).all()

    beta = 3.0
    f = Swish(beta, trainable=True)
    assert len(list(f.parameters())) == 1
    assert list(f.parameters())[0].shape == ()
    assert torch.isclose(f(x), x * torch.sigmoid(beta * x)).all()
