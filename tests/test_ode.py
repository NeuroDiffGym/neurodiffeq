import numpy as np
from numpy import isclose
from pytest import raises
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg') # use a non-GUI backend, so plots are not shown during testing

from neurodiffeq.neurodiffeq import safe_diff as diff
from neurodiffeq.networks import FCNN, SinActv
from neurodiffeq.ode import IVP, DirichletBVP
from neurodiffeq.ode import solve, solve_system, Monitor
from neurodiffeq.generators import Generator1D

import torch
torch.manual_seed(42)
np.random.seed(42)


def test_monitor():
    exponential = lambda x, t: diff(x, t) - x
    init_val_ex = IVP(t_0=0.0, x_0=1.0)
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.0, t_max=2.0,
                           max_epochs=3,
                           monitor=Monitor(t_min=0.0, t_max=2.0, check_every=1))
    print('Monitor test passed.')


def test_train_generator():
    exponential = lambda x, t: diff(x, t) - x
    init_val_ex = IVP(t_0=0.0, x_0=1.0)

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
    train_gen = Generator1D(size=32, t_min=np.log10(0.1), t_max=np.log10(2.0), method='log-spaced-noisy', noise_std=0.01)
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.1, t_max=2.0,
                           train_generator=train_gen,
                           max_epochs=3)

    with raises(ValueError):
        train_gen = Generator1D(size=32, t_min=0.0, t_max=2.0, method='magic')
    print('ExampleGenerator test passed.')


def test_ode():

    def mse(x, t):
        true_x = torch.sin(t)
        return torch.mean((x - true_x) ** 2)
    exponential = lambda x, t: diff(x, t) - x
    init_val_ex = IVP(t_0=0.0, x_0=1.0)
    solution_ex, _ = solve(ode=exponential, condition=init_val_ex,
                           t_min=0.0, t_max=2.0, shuffle=False,
                           max_epochs=2000, return_best=True, metrics={'mse': mse})
    ts = np.linspace(0, 2.0, 100)
    x_net = solution_ex(ts, as_type='np')
    x_ana = np.exp(ts)
    assert isclose(x_net, x_ana, atol=0.1).all()
    print('solve basic test passed.')


def test_ode_system():

    parametric_circle = lambda x1, x2, t : [diff(x1, t) - x2,
                                            diff(x2, t) + x1]
    init_vals_pc = [
        IVP(t_0=0.0, x_0=0.0),
        IVP(t_0=0.0, x_0=1.0)
    ]

    solution_pc, _ = solve_system(ode_system=parametric_circle,
                                  conditions=init_vals_pc,
                                  t_min=0.0, t_max=2*np.pi,
                                  max_epochs=5000,)

    ts = np.linspace(0, 2*np.pi, 100)
    x1_net, x2_net = solution_pc(ts, as_type='np')
    x1_ana, x2_ana = np.sin(ts), np.cos(ts)
    assert isclose(x1_net, x1_ana, atol=0.1).all()
    assert isclose(x2_net, x2_ana, atol=0.1).all()
    print('solve_system basic test passed.')


def test_ode_ivp():
    oscillator = lambda x, t: diff(x, t, order=2) + x
    init_val_ho = IVP(t_0=0.0, x_0=0.0, x_0_prime=1.0)
    solution_ho, _ = solve(ode=oscillator, condition=init_val_ho,
                           max_epochs=3000,
                           t_min=0.0, t_max=2*np.pi)
    ts = np.linspace(0, 2*np.pi, 100)
    x_net = solution_ho(ts, as_type='np')
    x_ana = np.sin(ts)
    assert isclose(x_net, x_ana, atol=0.1).all()
    print('IVP basic test passed.')


def test_ode_bvp():
    oscillator = lambda x, t: diff(x, t, order=2) + x
    bound_val_ho = DirichletBVP(t_0=0.0, x_0=0.0, t_1=1.5*np.pi, x_1=-1.0)
    solution_ho, _ = solve(ode=oscillator, condition=bound_val_ho,
                           max_epochs=3000,
                           t_min=0.0, t_max=1.5*np.pi)
    ts = np.linspace(0, 1.5*np.pi, 100)
    x_net = solution_ho(ts, as_type='np')
    x_ana = np.sin(ts)
    assert isclose(x_net, x_ana, atol=0.1).all()
    print('BVP basic test passed.')


def test_lotka_volterra():
    alpha, beta, delta, gamma = 1, 1, 1, 1
    lotka_volterra = lambda x, y, t : [diff(x, t) - (alpha*x  - beta*x*y),
                                       diff(y, t) - (delta*x*y - gamma*y)]
    init_vals_lv = [
        IVP(t_0=0.0, x_0=1.5),
        IVP(t_0=0.0, x_0=1.0)
    ]
    nets_lv = [
        FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),
        FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv)
    ]
    solution_lv, _ = solve_system(ode_system=lotka_volterra, conditions=init_vals_lv,
                                  t_min=0.0, t_max=12, nets=nets_lv, max_epochs=12000,
                                  monitor=Monitor(t_min=0.0, t_max=12, check_every=100))
    ts = np.linspace(0, 12, 100)
    prey_net, pred_net = solution_lv(ts, as_type='np')

    def dPdt(P, t):
        return [P[0]*alpha - beta*P[0]*P[1], delta*P[0]*P[1] - gamma*P[1]]
    P0 = [1.5, 1.0]
    Ps = odeint(dPdt, P0, ts)
    prey_num = Ps[:,0]
    pred_num = Ps[:,1]
    assert isclose(prey_net, prey_num, atol=0.1).all()
    assert isclose(pred_net, pred_num, atol=0.1).all()
    print('Lotka Volterra test passed.')

def test_additional_loss_term():
    def particle_squarewell(y1, y2, t):
        return [
            (-1 / 2) * diff(y1, t, order=2) - 3 - (y2) * (y1),
            diff(y2, t)
        ]

    def zero_y2(y1, y2, t):
        return torch.sum(y2 ** 2)

    boundary_conditions = [
        DirichletBVP(t_0=0, x_0=0, t_1=2, x_1=0),
        DirichletBVP(t_0=0, x_0=0, t_1=2, x_1=0),
    ]

    solution_squarewell, _ = solve_system(
        ode_system=particle_squarewell, conditions=boundary_conditions,
        additional_loss_term=zero_y2,
        t_min=0.0, t_max=2.0,
        max_epochs=1000,
    )

    ts = np.linspace(0.0, 2.0, 100)
    _, y2 = solution_squarewell(ts, as_type='np')
    assert isclose(y2, np.zeros_like(y2), atol=0.02).all()
