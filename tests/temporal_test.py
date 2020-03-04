from math import pi as PI
import torch
from torch import nn
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.temporal import generator_1dspatial, generator_temporal
from neurodiffeq.temporal import FirstOrderInitialCondition, BoundaryCondition
from neurodiffeq.temporal import SingleNetworkApproximator1DSpatialTemporal


def test_generator_1dspatial():
    s_gen = generator_1dspatial(size=32, x_min=-4, x_max=2, random=False)
    for _ in range(3):
        x = next(s_gen)
        assert x.shape == torch.Size([32])
        assert (x >= -4).all()
        assert (x <= 2).all()
        assert not x.requires_grad
    assert (x == next(s_gen)).all()

    s_gen = generator_1dspatial(size=32, x_min=-4, x_max=2, random=True)
    for _ in range(3):
        x = next(s_gen)
        assert x.shape == torch.Size([32])
        assert (x >= -4).all()
        assert (x <= 2).all()
        assert not x.requires_grad
    assert not (x == next(s_gen)).all()


def test_generator_temporal():
    t_gen = generator_temporal(size=32, t_min=0, t_max=42, random=False)
    for _ in range(3):
        t = next(t_gen)
        assert t.shape == torch.Size([32])
        assert (t >= 0).all()
        assert (t <= 42).all()
        assert not t.requires_grad
    assert (t == next(t_gen)).all()

    t_gen = generator_temporal(size=32, t_min=0, t_max=42, random=True)
    for _ in range(3):
        t = next(t_gen)
        assert t.shape == torch.Size([32])
        assert (t >= 0).all()
        assert (t <= 42).all()
        assert not t.requires_grad
    assert not (t == next(t_gen)).all()


def test_first_order_initial_condition():
    initial_condition = FirstOrderInitialCondition(u0=lambda x: torch.sin(x))
    x = torch.linspace(0, 1, 32)
    assert (initial_condition.u0(x) == torch.sin(x)).all()


def test_boundary_condition():
    def points_gen():
        while True:
            yield torch.tensor([0.])
    boundary_condition = BoundaryCondition(
        form=lambda u, x, t: t,
        points_generator=points_gen()
    )
    x = next(boundary_condition.points_generator)
    assert (x == torch.tensor([0.])).all()

    t_gen = generator_temporal(size=32, t_min=0, t_max=42, random=True)
    t = next(t_gen)
    xt = torch.cartesian_prod(x, t)
    xx, tt = xt[:, 0], xt[:, 1]
    def dummy_u(x, t):
        return t
    uu = dummy_u(xx, tt)
    assert (boundary_condition.form(uu, xx, tt) == tt).all()


def test_fully_connected_neural_network_approximator_1dspatial_temporal():
    DIFFUSIVITY, X_MIN, X_MAX, T_MIN, T_MAX = 0.3, 0.0, 2.0, 0.0, 3.0

    def heat_equation_1d(u, x, t):
        return diff(u, t) - DIFFUSIVITY * diff(u, x, order=2)

    initial_condition = FirstOrderInitialCondition(u0=lambda x: torch.sin(PI * x))

    def points_gen_lo():
        while True:
            yield torch.tensor([X_MIN])
    dirichlet_boundary_lo = BoundaryCondition(
        form=lambda u, x, t: torch.zeros_like(u),
        points_generator=points_gen_lo()
    )
    def points_gen_hi():
        while True:
            yield torch.tensor([X_MAX])
    dirichlet_boundary_hi = BoundaryCondition(
        form=lambda u, x, t: torch.zeros_like(u),
        points_generator=points_gen_hi()
    )

    fcnn = FCNN(
        n_input_units=2,
        n_output_units=1,
        n_hidden_units=32,
        n_hidden_layers=1,
        actv=nn.Tanh
    )
    fcnn_approximator = SingleNetworkApproximator1DSpatialTemporal(
        single_network=fcnn,
        pde=heat_equation_1d,
        initial_condition=initial_condition,
        boundary_conditions=[dirichlet_boundary_lo, dirichlet_boundary_hi]
    )

    xx, tt = torch.rand(16), torch.rand(16)
    assert fcnn_approximator(xx, tt).shape == torch.Size([16])
    assert next(fcnn_approximator.parameters()).shape == torch.Size([32, 2])
    x, t = torch.rand(4), torch.rand(4)
    assert fcnn_approximator.loss(x, t).shape == torch.Size([])
    xx, tt = torch.rand(16), torch.zeros(16)
    assert fcnn_approximator(xx, tt).isclose(torch.sin(PI * xx)).all()




