from math import pi as PI
import torch
from torch import nn, optim
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.temporal import generator_1dspatial, generator_temporal
from neurodiffeq.temporal import generator_2dspatial_segment, generator_2dspatial_rectangle
from neurodiffeq.temporal import FirstOrderInitialCondition, BoundaryCondition
from neurodiffeq.temporal import SingleNetworkApproximator1DSpatialTemporal
from neurodiffeq.temporal import Monitor1DSpatialTemporal
from neurodiffeq.temporal import _train_1dspatial_temporal, _valid_1dspatial_temporal, _solve_1dspatial_temporal
import matplotlib
matplotlib.use('Agg') # use a non-GUI backend, so plots are not shown during testing


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


def test_generator_2dspatial_segment():
    s_gen = generator_2dspatial_segment(size=32, start=(4., 2.), end=(-2., -4.), random=False)
    for _ in range(3):
        x, y = next(s_gen)
        assert x.shape == torch.Size([32]) and y.shape == torch.Size([32])
        assert (x - y - 2.).isclose(torch.zeros_like(x)).all()
        assert not x.requires_grad
    x_, y_ = next(s_gen)
    assert (x == x_).all() and (y == y_).all()

    s_gen = generator_2dspatial_segment(size=32, start=(4., 2.), end=(-2., -4.), random=True)
    for _ in range(3):
        x, y = next(s_gen)
        assert x.shape == torch.Size([32]) and y.shape == torch.Size([32])
        assert (x - y - 2.).isclose(torch.zeros_like(x), atol=1e-6).all()
        assert not x.requires_grad
    x_, y_ = next(s_gen)
    assert not (x == x_).all()
    assert not (y == y_).all()


def test_generator_2dspatial_rectangle():
    s_gen = generator_2dspatial_rectangle(size=(8, 8), x_min=-2., x_max=4., y_min=-4., y_max=2., random=False)
    for _ in range(3):
        x, y = next(s_gen)
        assert x.shape == torch.Size([64]) and y.shape == torch.Size([64])
        assert (x >= -2.).all() and (x <= 4.).all()
        assert (y >= -4.).all() and (y <= 2.).all()
        assert not x.requires_grad
    x_, y_ = next(s_gen)
    assert (x == x_).all() and (y == y_).all()

    s_gen = generator_2dspatial_rectangle(size=(8, 8), x_min=-2., x_max=4., y_min=-4., y_max=2., random=True)
    for _ in range(3):
        x, y = next(s_gen)
        assert x.shape == torch.Size([64]) and y.shape == torch.Size([64])
        assert (x >= -2.).all() and (x <= 4.).all()
        assert (y >= -4.).all() and (y <= 2.).all()
        assert not x.requires_grad
    x_, y_ = next(s_gen)
    assert not (x == x_).all()
    assert not (y == y_).all()


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
    xt = torch.cartesian_prod(x, t)
    xx = torch.squeeze(xt[:, 0])
    xx.requires_grad = True
    tt = torch.squeeze(xt[:, 1])
    tt.requires_grad = True
    assert fcnn_approximator.calculate_loss(xx, tt, x, t).shape == torch.Size([])
    def dummy_mse(uu, xx, tt):
        return torch.mean((uu - (xx+tt))**2)
    metrics = {'dummy_mse': dummy_mse}
    assert fcnn_approximator.calculate_metrics(xx, tt, x, t, metrics)['dummy_mse'].shape == torch.Size([])
    xx, tt = torch.rand(16), torch.zeros(16)
    assert fcnn_approximator(xx, tt).isclose(torch.sin(PI * xx)).all()


def test_monitor_1dspatial_temporal():
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

    dummy_history = {
        'train_loss': [100, 10, 1],
        'valid_loss': [200, 20, 2],
        'train_rmse': [1, 0.1, 0.01],
        'valid_rmse': [2, 0.2, 0.02]
    }

    monitor = Monitor1DSpatialTemporal(
        check_on_x=torch.linspace(X_MIN, X_MAX, 32),
        check_on_t=torch.linspace(T_MIN, T_MAX, 4),
        check_every=10
    )
    monitor.check(fcnn_approximator, dummy_history)

def test__train_1dspatial_temporal():
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

    s_gen = generator_1dspatial(size=32, x_min=X_MIN, x_max=X_MAX)
    t_gen = generator_temporal(size=32, t_min=T_MIN, t_max=T_MAX)

    adam = optim.Adam(fcnn_approximator.parameters())

    def dummy_mse(uu, xx, tt):
        return torch.mean((uu - (xx+tt))**2)
    metrics = {'dummy_mse': dummy_mse}

    train_epoch_loss, train_epoch_metrics = _train_1dspatial_temporal(s_gen, t_gen, fcnn_approximator, adam, metrics, shuffle=True, batch_size=100)
    assert train_epoch_loss > 0
    assert train_epoch_metrics['dummy_mse'] > 0


def test__valid_1dspatial_temporal():
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

    s_gen = generator_1dspatial(size=32, x_min=X_MIN, x_max=X_MAX)
    t_gen = generator_temporal(size=32, t_min=T_MIN, t_max=T_MAX)

    def dummy_mse(uu, xx, tt):
        return torch.mean((uu - (xx + tt)) ** 2)

    metrics = {'dummy_mse': dummy_mse}

    valid_epoch_loss, valid_epoch_metrics = _valid_1dspatial_temporal(s_gen, t_gen, fcnn_approximator, metrics)
    assert valid_epoch_loss > 0
    assert valid_epoch_metrics['dummy_mse'] > 0


def test__solve_1dspatial_temporal():
    DIFFUSIVITY, X_MIN, X_MAX, T_MIN, T_MAX = 0.3, 0.0, 2.0, 0.0, 6.0

    def heat_equation_1d(u, x, t):
        return diff(u, t) - DIFFUSIVITY * diff(u, x, order=2)

    def analytical_solution(xx, tt):
        return torch.sin(PI * xx / X_MAX) * torch.exp(-DIFFUSIVITY * PI ** 2 * tt / X_MAX ** 2)

    def rmse(uu, xx, tt):
        error = uu - analytical_solution(xx, tt)
        return torch.mean(error ** 2) ** 0.5

    metrics = {'rmse': rmse}

    initial_condition = FirstOrderInitialCondition(u0=lambda x: torch.sin(PI * x / X_MAX))

    # a generator seems too much trouble for 1D boundary, but may be more flexible for 2D boundary
    def points_gen_lo():
        while True:
            yield torch.tensor([X_MIN])

    dirichlet_boundary_lo = BoundaryCondition(
        form=lambda u, x, t: u,
        points_generator=points_gen_lo()
    )

    def points_gen_hi():
        while True:
            yield torch.tensor([X_MAX])

    dirichlet_boundary_hi = BoundaryCondition(
        form=lambda u, x, t: u,
        points_generator=points_gen_hi()
    )

    train_gen_spatial = generator_1dspatial(size=32, x_min=X_MIN, x_max=X_MAX)
    train_gen_temporal = generator_temporal(size=32, t_min=T_MIN, t_max=T_MAX)
    valid_gen_spatial = generator_1dspatial(size=32, x_min=X_MIN, x_max=X_MAX, random=False)
    valid_gen_temporal = generator_temporal(size=32, t_min=T_MIN, t_max=T_MAX, random=False)
    monitor = Monitor1DSpatialTemporal(
        check_on_x=torch.linspace(X_MIN, X_MAX, 32),
        check_on_t=torch.linspace(T_MIN, T_MAX, 4),
        check_every=10
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
    adam = optim.Adam(fcnn_approximator.parameters())

    heat_equation_1d_solution, _ = _solve_1dspatial_temporal(
        train_generator_spatial=train_gen_spatial,
        train_generator_temporal=train_gen_temporal,
        valid_generator_spatial=valid_gen_spatial,
        valid_generator_temporal=valid_gen_temporal,
        approximator=fcnn_approximator,
        optimizer=adam,
        batch_size=64,
        max_epochs=1,
        shuffle=True,
        metrics=metrics,
        monitor=monitor
    )

    xx, tt = torch.rand(16), torch.rand(16)
    assert heat_equation_1d_solution(xx, tt).shape == torch.Size([16])
    xx, tt = torch.rand(16), torch.zeros(16)
    assert fcnn_approximator(xx, tt).isclose(torch.sin(PI * xx / X_MAX)).all()