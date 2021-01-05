import numpy as np
from numpy import isclose
from pytest import raises, warns
from scipy.integrate import odeint
import matplotlib

matplotlib.use('Agg')  # use a non-GUI backend, so plots are not shown during testing

from neurodiffeq.neurodiffeq import safe_diff as diff
from neurodiffeq.networks import FCNN, SinActv
from neurodiffeq.ode import IVP, DirichletBVP
from neurodiffeq.ode import solve, solve_system, Monitor
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.solvers import Solution1D, Solver1D
from neurodiffeq.generators import Generator1D, BaseGenerator

import torch

torch.manual_seed(42)
np.random.seed(42)


def test_monitor():
    exponential = lambda u, t: diff(u, t) - u
    init_val_ex = IVP(t_0=0.0, u_0=1.0)
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.0, t_max=2.0,
                           max_epochs=3,
                           monitor=Monitor1D(t_min=0.0, t_max=2.0, check_every=1))

    with warns(DeprecationWarning):
        solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                               t_min=0.0, t_max=2.0,
                               max_epochs=3,
                               monitor=Monitor(t_min=0.0, t_max=2.0, check_every=1))


def test_train_generator():
    exponential = lambda u, t: diff(u, t) - u
    init_val_ex = IVP(t_0=0.0, u_0=1.0)

    train_gen = Generator1D(size=32, t_min=0.0, t_max=2.0, method='uniform')
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.0, t_max=2.0,
                           train_generator=train_gen,
                           max_epochs=3)
    train_gen = Generator1D(size=32, t_min=0.0, t_max=2.0, method='equally-spaced')
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.0, t_max=2.0,
                           train_generator=train_gen,
                           max_epochs=3)
    train_gen = Generator1D(size=32, t_min=0.0, t_max=2.0, method='equally-spaced-noisy')
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.0, t_max=2.0,
                           train_generator=train_gen,
                           max_epochs=3)
    train_gen = Generator1D(size=32, t_min=0.0, t_max=2.0, method='equally-spaced-noisy', noise_std=0.01)
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.0, t_max=2.0,
                           train_generator=train_gen,
                           max_epochs=3)
    train_gen = Generator1D(size=32, t_min=np.log10(0.1), t_max=np.log10(2.0), method='log-spaced')
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.1, t_max=2.0,
                           train_generator=train_gen,
                           max_epochs=3)
    train_gen = Generator1D(size=32, t_min=np.log10(0.1), t_max=np.log10(2.0), method='log-spaced-noisy')
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.1, t_max=2.0,
                           train_generator=train_gen,
                           max_epochs=3)
    train_gen = Generator1D(size=32, t_min=np.log10(0.1), t_max=np.log10(2.0), method='log-spaced-noisy',
                            noise_std=0.01)
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.1, t_max=2.0,
                           train_generator=train_gen,
                           max_epochs=3)

    with raises(ValueError):
        train_gen = Generator1D(size=32, t_min=0.0, t_max=2.0, method='magic')


def test_ode():
    def mse(u, t):
        true_u = torch.sin(t)
        return torch.mean((u - true_u) ** 2)

    exponential = lambda u, t: diff(u, t) - u
    init_val_ex = IVP(t_0=0.0, u_0=1.0)
    solution_ex, loss_history = solve(ode=exponential, condition=init_val_ex,
                                      t_min=0.0, t_max=2.0, shuffle=False,
                                      max_epochs=10, return_best=True, metrics={'mse': mse})

    assert isinstance(solution_ex, Solution1D)
    assert isinstance(loss_history, dict)
    keys = ['train_loss', 'valid_loss']
    for key in keys:
        assert key in loss_history
        assert isinstance(loss_history[key], list)
    assert len(loss_history[keys[0]]) == len(loss_history[keys[1]])


def test_ode_system():
    parametric_circle = lambda u1, u2, t: [diff(u1, t) - u2,
                                           diff(u2, t) + u1]
    init_vals_pc = [
        IVP(t_0=0.0, u_0=0.0),
        IVP(t_0=0.0, u_0=1.0)
    ]

    solution_pc, loss_history = solve_system(ode_system=parametric_circle,
                                             conditions=init_vals_pc,
                                             t_min=0.0, t_max=2 * np.pi,
                                             max_epochs=10, )

    assert isinstance(solution_pc, Solution1D)
    assert isinstance(loss_history, dict)
    keys = ['train_loss', 'valid_loss']
    for key in keys:
        assert key in loss_history
        assert isinstance(loss_history[key], list)
    assert len(loss_history[keys[0]]) == len(loss_history[keys[1]])


def test_additional_loss_term():
    def particle_squarewell(y1, y2, t):
        return [
            (-1 / 2) * diff(y1, t, order=2) - 3 - (y2) * (y1),
            diff(y2, t)
        ]

    def zero_y2(y1, y2, t):
        return torch.sum(y2 ** 2)

    boundary_conditions = [
        DirichletBVP(t_0=0, u_0=0, t_1=2, u_1=0),
        DirichletBVP(t_0=0, u_0=0, t_1=2, u_1=0),
    ]

    solution_squarewell, loss_history = solve_system(
        ode_system=particle_squarewell, conditions=boundary_conditions,
        additional_loss_term=zero_y2,
        t_min=0.0, t_max=2.0,
        max_epochs=10,
    )
    assert isinstance(solution_squarewell, Solution1D)
    assert isinstance(loss_history, dict)
    keys = ['train_loss', 'valid_loss']
    for key in keys:
        assert key in loss_history
        assert isinstance(loss_history[key], list)
    assert len(loss_history[keys[0]]) == len(loss_history[keys[1]])


def test_solution():
    t0, u0 = np.random.rand() + 0, np.random.rand() + 0
    t1, u1 = np.random.rand() + 1, np.random.rand() + 1
    N_SAMPLES = 100

    def get_solution(use_single: bool) -> Solution1D:
        conditions = [IVP(t0, u0), IVP(t1, u1)]
        if use_single:
            net = FCNN(1, 2)
            for i, cond in enumerate(conditions):
                cond.set_impose_on(i)
            return Solution1D(net, conditions)
        else:
            nets = [FCNN(1, 1), FCNN(1, 1)]
            return Solution1D(nets, conditions)

    def check_output(us, shape, type, msg=""):
        msg += " "
        assert isinstance(us, (list, tuple)), msg + "returned type is not a list"
        assert len(us) == 2, msg + "returned length is not 2"
        assert isinstance(us[0], type) and isinstance(us[1], type), msg + f"returned element is not {type}"
        assert us[0].shape == shape and us[1].shape == shape, msg + f"returned element shape is not {shape}"
        assert us[0][0] == u0, msg + f"first condition is not properly imposed"
        assert us[1][-1] == u1, msg + f"second condition is not properly imposed"

    for use_single in [True, False]:
        solution = get_solution(use_single=use_single)
        ts = torch.linspace(t0, t1, N_SAMPLES)
        us = solution(ts)
        check_output(us, shape=(N_SAMPLES,), type=torch.Tensor, msg=f"[use_single={use_single}]")
        us = solution(ts, as_type='np')
        check_output(us, shape=(N_SAMPLES,), type=np.ndarray, msg=f"[use_single={use_single}]")

        ts = ts.reshape(-1, 1)
        us = solution(ts)
        check_output(us, shape=(N_SAMPLES, 1), type=torch.Tensor, msg=f"[use_single={use_single}]")
        us = solution(ts, as_type='np')
        check_output(us, shape=(N_SAMPLES, 1), type=np.ndarray, msg=f"[use_single={use_single}]")


def test_get_internals():
    parametric_circle = lambda x1, x2, t: [diff(x1, t) - x2, diff(x2, t) + x1]

    init_vals_pc = [
        IVP(t_0=0.0, u_0=0.0),
        IVP(t_0=0.0, u_0=1.0),
    ]

    solver = Solver1D(
        ode_system=parametric_circle,
        conditions=init_vals_pc,
        t_min=0.0,
        t_max=2*np.pi,
    )

    solver.fit(max_epochs=1)
    internals = solver.get_internals()
    assert isinstance(internals, dict)
    internals = solver.get_internals(return_type='list')
    assert isinstance(internals, dict)
    internals = solver.get_internals(return_type='dict')
    assert isinstance(internals, dict)
    internals = solver.get_internals(['generator', 'n_batches'], return_type='dict')
    assert isinstance(internals, dict)
    internals = solver.get_internals(['generator', 'n_batches'], return_type='list')
    assert isinstance(internals, list)
    internals = solver.get_internals('train_generator')
    assert isinstance(internals, BaseGenerator)
