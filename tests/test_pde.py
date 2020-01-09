import numpy as np
import torch
from numpy import isclose

from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.pde import DirichletBVP2D, IBVP1D
from neurodiffeq.pde import solve2D, ExampleGenerator2D, Monitor2D, make_animation

from pytest import raises

import torch
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
        pde=laplace, condition=bc, xy_min=[0, 0], xy_max=[1, 1],
        net=net, max_epochs=3,
        train_generator=ExampleGenerator2D([32, 32], [0, 0], [1, 1], method='equally-spaced-noisy'),
        batch_size=64,
        monitor=Monitor2D(check_every=1, xy_min=[0, 0], xy_max=[1, 1])
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
        pde=laplace, condition=bc, xy_min=[0, 0], xy_max=[1, 1],
        net=net, max_epochs=3,
        train_generator=ExampleGenerator2D([32, 32], [0, 0], [1, 1], method='equally-spaced-noisy'),
        batch_size=64,
        monitor=Monitor2D(check_every=1, xy_min=[0, 0], xy_max=[1, 1])
    )

    train_gen = ExampleGenerator2D([32, 32], [0, 0], [1, 1], method='equally-spaced')
    solution_neural_net_laplace, _ = solve2D(
        pde=laplace, condition=bc, xy_min=[0, 0], xy_max=[1, 1],
        net=net, max_epochs=3, train_generator=train_gen, batch_size=64
    )
    train_gen = ExampleGenerator2D([32, 32], [0, 0], [1, 1], method='equally-spaced-noisy')
    solution_neural_net_laplace, _ = solve2D(
        pde=laplace, condition=bc, xy_min=[0, 0], xy_max=[1, 1],
        net=net, max_epochs=3, train_generator=train_gen, batch_size=64
    )

    with raises(ValueError):
        train_gen = ExampleGenerator2D([32, 32], [0, 0], [1, 1], method='magic')
    print('ExampleGenerator test passed.')


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
        pde=laplace, condition=bc, xy_min=[0, 0], xy_max=[1, 1],
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D([32, 32], [0, 0], [1, 1], method='equally-spaced-noisy', xy_noise_std=(0.01, 0.01)),
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

    solution_neural_net_heat, _ = solve2D(
        pde=heat, condition=ibvp, xy_min=[0, 0], xy_max=[L, T],
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D([32, 32], [0, 0], [L, T], method='equally-spaced-noisy'),
        batch_size=64
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


def test_neumann_boundaries():

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
        pde=heat, condition=ibvp, xy_min=[0, 0], xy_max=[L, T],
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D([32, 32], [0, 0], [L, T], method='equally-spaced-noisy'),
        batch_size=64
    )

    xs = np.linspace(0, L, 101)
    ts = np.linspace(0, T, 101)
    xx, tt = np.meshgrid(xs, ts)
    make_animation(solution_neural_net_heat, xs, ts)  # test animation
    sol_ana = solution_analytical_heat(xx, tt)
    sol_net = solution_neural_net_heat(xx, tt, as_type='np')
    assert isclose(sol_net, sol_ana, atol=0.01).all()
    print('Dirichlet on the left Neumann on the right test passed.')

    # Neumann on the left Dirichlet on the right
    ibvp = IBVP1D(
        x_min=0, x_min_prime=lambda t: np.pi / L * torch.exp(-k * np.pi ** 2 * t / L ** 2),
        x_max=L, x_max_val=lambda t: 0,
        t_min=0, t_min_val=lambda x: torch.sin(np.pi * x / L)
    )

    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)

    solution_neural_net_heat, _ = solve2D(
        pde=heat, condition=ibvp, xy_min=[0, 0], xy_max=[L, T],
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D([32, 32], [0, 0], [L, T], method='equally-spaced-noisy'),
        batch_size=64
    )

    xs = np.linspace(0, L, 101)
    ts = np.linspace(0, T, 101)
    xx, tt = np.meshgrid(xs, ts)
    make_animation(solution_neural_net_heat, xs, ts)  # test animation
    sol_ana = solution_analytical_heat(xx, tt)
    sol_net = solution_neural_net_heat(xx, tt, as_type='np')
    assert isclose(sol_net, sol_ana, atol=0.01).all()
    print('Neumann on the left Dirichlet on the right test passed.')

    # Neumann on both sides
    ibvp = IBVP1D(
        x_min=0, x_min_prime=lambda t: np.pi / L * torch.exp(-k * np.pi ** 2 * t / L ** 2),
        x_max=L, x_max_prime=lambda t: -np.pi / L * torch.exp(-k * np.pi ** 2 * t / L ** 2),
        t_min=0, t_min_val=lambda x: torch.sin(np.pi * x / L)
    )

    net = FCNN(n_input_units=2, n_hidden_units=32, n_hidden_layers=1)

    solution_neural_net_heat, _ = solve2D(
        pde=heat, condition=ibvp, xy_min=[0, 0], xy_max=[L, T],
        net=net, max_epochs=300,
        train_generator=ExampleGenerator2D([32, 32], [0, 0], [L, T], method='equally-spaced-noisy'),
        batch_size=64
    )

    xs = np.linspace(0, L, 101)
    ts = np.linspace(0, T, 101)
    xx, tt = np.meshgrid(xs, ts)
    make_animation(solution_neural_net_heat, xs, ts)  # test animation
    sol_ana = solution_analytical_heat(xx, tt)
    sol_net = solution_neural_net_heat(xx, tt, as_type='np')
    assert isclose(sol_net, sol_ana, atol=0.01).all()
    print('Neumann on the left Dirichlet on the right test passed.')
