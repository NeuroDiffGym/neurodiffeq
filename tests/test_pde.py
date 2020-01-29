import numpy as np
from numpy import isclose
import matplotlib
matplotlib.use('Agg') # use a non-GUI backend, so plots are not shown during testing

from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.pde import DirichletBVP2D, IBVP1D, Condition, _network_output_2input
from neurodiffeq.pde import solve2D, solve2D_system, ExampleGenerator2D, Monitor2D, make_animation

from pytest import raises

import torch
import torch.nn as nn
torch.manual_seed(42)
np.random.seed(42)


def test_monitor():

    laplace = lambda u, x, y: diff(u, x, order=2) + diff(u, y, order=2)
    bc = DirichletBVP2D(
        x_min=0, x_min_val=lambda y: torch.sin(np.pi * y),
        x_max=1, x_max_val=lambda y: 0,
        y_min=0, y_min_val=lambda x: 0,
        y_max=1, y_max_val=lambda x: 0
    )

    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)
    solution_neural_net_laplace, _ = solve2D(
        pde=laplace, condition=bc, xy_min=(0, 0), xy_max=(1, 1),
        net=net, max_epochs=3,
        train_generator=ExampleGenerator2D((32, 32), (0, 0), (1, 1), method='equally-spaced-noisy'),
        batch_size=64,
        monitor=Monitor2D(check_every=1, xy_min=(0, 0), xy_max=(1, 1))
    )
    print('Monitor test passed.')


def test_train_generator():
    laplace = lambda u, x, y: diff(u, x, order=2) + diff(u, y, order=2)
    bc = DirichletBVP2D(
        x_min=0, x_min_val=lambda y: torch.sin(np.pi * y),
        x_max=1, x_max_val=lambda y: 0,
        y_min=0, y_min_val=lambda x: 0,
        y_max=1, y_max_val=lambda x: 0
    )

    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)
    solution_neural_net_laplace, _ = solve2D(
        pde=laplace, condition=bc, xy_min=(0, 0), xy_max=(1, 1),
        net=net, max_epochs=3,
        train_generator=ExampleGenerator2D((32, 32), (0, 0), (1, 1), method='equally-spaced-noisy'),
        batch_size=64,
        monitor=Monitor2D(check_every=1, xy_min=(0, 0), xy_max=(1, 1))
    )

    train_gen = ExampleGenerator2D((32, 32), (0, 0), (1, 1), method='equally-spaced')
    solution_neural_net_laplace, _ = solve2D(
        pde=laplace, condition=bc, xy_min=(0, 0), xy_max=(1, 1),
        net=net, max_epochs=3, train_generator=train_gen, batch_size=64
    )
    train_gen = ExampleGenerator2D((32, 32), (0, 0), (1, 1), method='equally-spaced-noisy')
    solution_neural_net_laplace, _ = solve2D(
        pde=laplace, condition=bc, xy_min=(0, 0), xy_max=(1, 1),
        net=net, max_epochs=3, train_generator=train_gen, batch_size=64
    )

    with raises(ValueError):
        train_gen = ExampleGenerator2D((32, 32), (0, 0), (1, 1), method='magic')
    print('ExampleGenerator test passed.')

    valid_gen = ExampleGenerator2D((32, 32), (0, 0), (1, 1), method='equally-spaced-noisy')
    train_gen = ExampleGenerator2D((32, 32), (0, 0), (1, 1), method='equally-spaced')
    solution_neural_net_laplace, _ = solve2D(
        pde=laplace, condition=bc,
        net=net, max_epochs=3, train_generator=train_gen, valid_generator=valid_gen, batch_size=64
    )

    with raises(RuntimeError):
        solution_neural_net_laplace, _ = solve2D(
            pde=laplace, condition=bc,
            net=net, max_epochs=3, batch_size=64
        )


def test_ibvp():
    with raises(NotImplementedError):
        IBVP1D(
            t_min=0, t_min_val=lambda x: 0,
            x_min=0, x_min_val=lambda t: None, x_min_prime=lambda t: None,
            x_max=1, x_max_val=lambda t: None, x_max_prime=lambda t: None,
        )
    with raises(NotImplementedError):
        IBVP1D(
            t_min=0, t_min_val=lambda x: 0,
            x_min=0, x_min_val=lambda t: 0, x_min_prime=lambda t: 0,
            x_max=1, x_max_val=lambda t: None, x_max_prime=lambda t: None,
        )
    print('IBVP test passed.')


def test_laplace():

    laplace = lambda u, x, y: diff(u, x, order=2) + diff(u, y, order=2)
    bc = DirichletBVP2D(
        x_min=0, x_min_val=lambda y: torch.sin(np.pi*y),
        x_max=1, x_max_val=lambda y: 0,
        y_min=0, y_min_val=lambda x: 0,
        y_max=1, y_max_val=lambda x: 0
    )

    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)
    solution_neural_net_laplace, _ = solve2D(
        pde=laplace, condition=bc, xy_min=(0, 0), xy_max=(1, 1),
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D((32, 32), (0, 0), (1, 1), method='equally-spaced-noisy', xy_noise_std=(0.01, 0.01)),
        batch_size=64
    )
    solution_analytical_laplace = lambda x, y: np.sin(np.pi * y) * np.sinh(np.pi * (1 - x)) / np.sinh(np.pi)

    xs, ys = np.linspace(0, 1, 101), np.linspace(0, 1, 101)
    xx, yy = np.meshgrid(xs, ys)
    sol_net = solution_neural_net_laplace(xx, yy, as_type='np')
    sol_ana = solution_analytical_laplace(xx, yy)
    assert isclose(sol_net, sol_ana, atol=0.01).all()
    print('Laplace test passed.')


def test_heat():

    k, L, T = 0.3, 2, 3
    heat = lambda u, x, t: diff(u, t) - k * diff(u, x, order=2)

    ibvp = IBVP1D(
        x_min=0, x_min_val=lambda t: 0,
        x_max=L, x_max_val=lambda t: 0,
        t_min=0, t_min_val=lambda x: torch.sin(np.pi * x / L)
    )
    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)

    def mse(u, x, y):
        true_u = torch.sin(np.pi * y) * torch.sinh(np.pi * (1 - x)) / np.sinh(np.pi)
        return torch.mean((u - true_u) ** 2)

    solution_neural_net_heat, _ = solve2D(
        pde=heat, condition=ibvp, xy_min=(0, 0), xy_max=(L, T),
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D((32, 32), (0, 0), (L, T), method='equally-spaced-noisy'),
        batch_size=64, metrics={'mse': mse}
    )
    solution_analytical_heat = lambda x, t: np.sin(np.pi * x / L) * np.exp(-k * np.pi ** 2 * t / L ** 2)

    xs = np.linspace(0, L, 101)
    ts = np.linspace(0, T, 101)
    xx, tt = np.meshgrid(xs, ts)
    make_animation(solution_neural_net_heat, xs, ts) # test animation
    sol_ana = solution_analytical_heat(xx, tt)
    sol_net = solution_neural_net_heat(xx, tt, as_type='np')
    assert isclose(sol_net, sol_ana, atol=0.01).all()
    print('Heat test passed.')


def test_neumann_boundaries_1():

    k, L, T = 0.3, 2, 3
    heat = lambda u, x, t: diff(u, t) - k * diff(u, x, order=2)
    solution_analytical_heat = lambda x, t: np.sin(np.pi * x / L) * np.exp(-k * np.pi ** 2 * t / L ** 2)

    # Dirichlet on the left Neumann on the right
    ibvp = IBVP1D(
        x_min=0, x_min_val=lambda t: 0,
        x_max=L, x_max_prime=lambda t: -np.pi / L * torch.exp(-k * np.pi ** 2 * t / L ** 2),
        t_min=0, t_min_val=lambda x: torch.sin(np.pi * x / L)
    )

    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)

    solution_neural_net_heat, _ = solve2D(
        pde=heat, condition=ibvp, xy_min=(0, 0), xy_max=(L, T),
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D((32, 32), (0, 0), (L, T), method='equally-spaced-noisy'),
        batch_size=64
    )

    xs = np.linspace(0, L, 101)
    ts = np.linspace(0, T, 101)
    xx, tt = np.meshgrid(xs, ts)
    make_animation(solution_neural_net_heat, xs, ts)  # test animation
    sol_ana = solution_analytical_heat(xx, tt)
    sol_net = solution_neural_net_heat(xx, tt, as_type='np')
    assert isclose(sol_net, sol_ana, atol=0.1).all()
    print('Dirichlet on the left Neumann on the right test passed.')

def test_neumann_boundaries_2():

    k, L, T = 0.3, 2, 3
    heat = lambda u, x, t: diff(u, t) - k * diff(u, x, order=2)
    solution_analytical_heat = lambda x, t: np.sin(np.pi * x / L) * np.exp(-k * np.pi ** 2 * t / L ** 2)

    # Neumann on the left Dirichlet on the right
    ibvp = IBVP1D(
        x_min=0, x_min_prime=lambda t: np.pi / L * torch.exp(-k * np.pi ** 2 * t / L ** 2),
        x_max=L, x_max_val=lambda t: 0,
        t_min=0, t_min_val=lambda x: torch.sin(np.pi * x / L)
    )

    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)

    solution_neural_net_heat, _ = solve2D(
        pde=heat, condition=ibvp, xy_min=(0, 0), xy_max=(L, T),
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D((32, 32), (0, 0), (L, T), method='equally-spaced-noisy'),
        batch_size=64
    )

    xs = np.linspace(0, L, 101)
    ts = np.linspace(0, T, 101)
    xx, tt = np.meshgrid(xs, ts)
    make_animation(solution_neural_net_heat, xs, ts)  # test animation
    sol_ana = solution_analytical_heat(xx, tt)
    sol_net = solution_neural_net_heat(xx, tt, as_type='np')
    assert isclose(sol_net, sol_ana, atol=0.1).all()
    print('Neumann on the left Dirichlet on the right test passed.')

def test_neumann_boundaries_3():
    k, L, T = 0.3, 2, 3
    heat = lambda u, x, t: diff(u, t) - k * diff(u, x, order=2)
    solution_analytical_heat = lambda x, t: np.sin(np.pi * x / L) * np.exp(-k * np.pi ** 2 * t / L ** 2)

    # Neumann on both sides
    ibvp = IBVP1D(
        x_min=0, x_min_prime=lambda t: np.pi / L * torch.exp(-k * np.pi ** 2 * t / L ** 2),
        x_max=L, x_max_prime=lambda t: -np.pi / L * torch.exp(-k * np.pi ** 2 * t / L ** 2),
        t_min=0, t_min_val=lambda x: torch.sin(np.pi * x / L)
    )

    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)

    solution_neural_net_heat, _ = solve2D(
        pde=heat, condition=ibvp, xy_min=(0, 0), xy_max=(L, T),
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D((32, 32), (0, 0), (L, T), method='equally-spaced-noisy'),
        batch_size=64
    )

    xs = np.linspace(0, L, 101)
    ts = np.linspace(0, T, 101)
    xx, tt = np.meshgrid(xs, ts)
    make_animation(solution_neural_net_heat, xs, ts)  # test animation
    sol_ana = solution_analytical_heat(xx, tt)
    sol_net = solution_neural_net_heat(xx, tt, as_type='np')
    assert isclose(sol_net, sol_ana, atol=0.1).all()
    print('Neumann on the left Dirichlet on the right test passed.')


def test_pde_system():
    def _network_output_2input(net, xs, ys, ith_unit):
        xys = torch.cat((xs, ys), 1)
        nn_output = net(xys)
        if ith_unit is not None:
            return nn_output[:, ith_unit].reshape(-1, 1)
        else:
            return nn_output

    class BCOnU(Condition):
        """for u(x, y), impose u(x, -1) = u(x, 1) = 0; dudx(0, y) = dudy(L, y) = 0"""

        def __init__(self, x_min, x_max, y_min, y_max):
            super().__init__()
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max

        def enforce(self, net, x, y):
            uxy = _network_output_2input(net, x, y, self.ith_unit)

            x_ones = torch.ones_like(x, requires_grad=True)
            x_ones_min = self.x_min * x_ones
            x_ones_max = self.x_max * x_ones
            uxminy = _network_output_2input(net, x_ones_min, y, self.ith_unit)
            uxmaxy = _network_output_2input(net, x_ones_max, y, self.ith_unit)

            x_tilde = (x - self.x_min) / (self.x_max - self.x_min)
            y_tilde = (y - self.y_min) / (self.y_max - self.y_min)

            return y_tilde * (1 - y_tilde) * (
                    uxy - x_tilde * (self.x_max - self.x_min) * diff(uxminy, x_ones_min) \
                    + 0.5 * x_tilde ** 2 * (self.x_max - self.x_min) * (
                            diff(uxminy, x_ones_min) - diff(uxmaxy, x_ones_max)
                    )
            )

    class BCOnP(Condition):
        """for p(x, y), impose p(0, y) = p_max; p(L, y) = p_min"""

        def __init__(self, x_min, x_max, p_x_min, p_x_max):
            super().__init__()
            self.x_min = x_min
            self.x_max = x_max
            self.p_x_min = p_x_min
            self.p_x_max = p_x_max

        def enforce(self, net, x, y):
            uxy = _network_output_2input(net, x, y, self.ith_unit)
            x_tilde = (x - self.x_min) / (self.x_max - self.x_min)

            return (1 - x_tilde) * self.p_x_min + x_tilde * self.p_x_max \
                   + x_tilde * (1 - x_tilde) * uxy

    L = 2.0
    mu = 1.0
    P1, P2 = 1.0, 0.0
    def poiseuille(u, v, p, x, y):
        return [
            mu * (diff(u, x, order=2) + diff(u, y, order=2)) - diff(p, x),
            mu * (diff(v, x, order=2) + diff(v, y, order=2)) - diff(p, y),
            diff(u, x) + diff(v, y)
        ]
    def zero_divergence(u, v, p, x, y):
        return torch.sum( (diff(u, x) + diff(v, y))**2 )

    bc_on_u = BCOnU(
        x_min=0,
        x_max=L,
        y_min=-1,
        y_max=1,
    )
    bc_on_v = DirichletBVP2D(
        x_min=0, x_min_val=lambda y: 0,
        x_max=L, x_max_val=lambda y: 0,
        y_min=-1, y_min_val=lambda x: 0,
        y_max=1, y_max_val=lambda x: 0
    )
    bc_on_p = BCOnP(
        x_min=0,
        x_max=L,
        p_x_min=P1,
        p_x_max=P2,
    )
    conditions = [bc_on_u, bc_on_v, bc_on_p]

    nets = [
        FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1, actv=nn.Softplus)
        for _ in range(3)
    ]

    # use one neural network for each dependent variable
    solution_neural_net_poiseuille, _ = solve2D_system(
        pde_system=poiseuille, conditions=conditions, xy_min=(0, -1), xy_max=(L, 1),
        train_generator=ExampleGenerator2D((32, 32), (0, -1), (L, 1), method='equally-spaced-noisy'),
        max_epochs=300, batch_size=64, nets=nets, additional_loss_term=zero_divergence,
        monitor=Monitor2D(check_every=10, xy_min=(0, -1), xy_max=(L, 1))
    )

    def solution_analytical_poiseuille(xs, ys):
        us = (P1 - P2) / (L * 2 * mu) * (1 - ys ** 2)
        vs = np.zeros_like(xs)
        ps = P1 + (P2 - P1) * xs / L
        return [us, vs, ps]

    xs, ys = np.linspace(0, L, 101), np.linspace(-1, 1, 101)
    xx, yy = np.meshgrid(xs, ys)
    u_ana, v_ana, p_ana = solution_analytical_poiseuille(xx, yy)
    u_net, v_net, p_net = solution_neural_net_poiseuille(xx, yy, as_type='np')

    assert isclose(u_ana, u_net, atol=0.01).all()
    assert isclose(v_ana, v_net, atol=0.01).all()
    assert isclose(p_ana, p_net, atol=0.01).all()

    # use a single neural network
    net = FCNN(n_input_units=2, n_output_units=3, n_hidden_units=32, n_hidden_layers=1, actv=nn.Softplus)
    solution_neural_net_poiseuille, _ = solve2D_system(
        pde_system=poiseuille, conditions=conditions, xy_min=(0, -1), xy_max=(L, 1),
        train_generator=ExampleGenerator2D((32, 32), (0, -1), (L, 1), method='equally-spaced-noisy'),
        max_epochs=300, batch_size=64, single_net=net, additional_loss_term=zero_divergence,
        monitor=Monitor2D(check_every=10, xy_min=(0, -1), xy_max=(L, 1))
    )

    u_ana, v_ana, p_ana = solution_analytical_poiseuille(xx, yy)
    u_net, v_net, p_net = solution_neural_net_poiseuille(xx, yy, as_type='np')

    assert isclose(u_ana, u_net, atol=0.01).all()
    assert isclose(v_ana, v_net, atol=0.01).all()
    assert isclose(p_ana, p_net, atol=0.01).all()
