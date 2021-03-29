import numpy as np
import random
from numpy import isclose
import pytest
import matplotlib

matplotlib.use('Agg')  # use a non-GUI backend, so plots are not shown during testing

from neurodiffeq.neurodiffeq import safe_diff as diff
from neurodiffeq.networks import FCNN
from neurodiffeq.pde import DirichletControlPoint, NeumannControlPoint, Point, CustomBoundaryCondition
from neurodiffeq.pde import solve2D, solve2D_system, Monitor2D, make_animation
from neurodiffeq.pde import Solution
from neurodiffeq.pde import Solution2D
from neurodiffeq.generators import PredefinedGenerator, Generator2D
from neurodiffeq.conditions import DirichletBVP2D, DirichletBVP

import torch
import torch.nn as nn
import torch.optim as optim


@pytest.fixture(autouse=True)
def magic():
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

    net = FCNN(n_input_units=2, hidden_units=(32, 32))
    with pytest.warns(FutureWarning):
        solve2D(
            pde=laplace, condition=bc, xy_min=(0, 0), xy_max=(1, 1),
            net=net, max_epochs=3,
            train_generator=Generator2D((32, 32), (0, 0), (1, 1), method='equally-spaced-noisy'),
            batch_size=64,
            monitor=Monitor2D(check_every=1, xy_min=(0, 0), xy_max=(1, 1))
        )


def test_train_generator():
    laplace = lambda u, x, y: diff(u, x, order=2) + diff(u, y, order=2)
    bc = DirichletBVP2D(
        x_min=0, x_min_val=lambda y: torch.sin(np.pi * y),
        x_max=1, x_max_val=lambda y: 0,
        y_min=0, y_min_val=lambda x: 0,
        y_max=1, y_max_val=lambda x: 0
    )

    net = FCNN(n_input_units=2, hidden_units=(32, 32))

    with pytest.raises(ValueError), pytest.warns(FutureWarning):
        solution_neural_net_laplace, _ = solve2D(
            pde=laplace, condition=bc,
            net=net, max_epochs=3, batch_size=64
        )


def test_laplace():
    laplace = lambda u, x, y: diff(u, x, order=2) + diff(u, y, order=2)
    bc = DirichletBVP2D(
        x_min=0, x_min_val=lambda y: torch.sin(np.pi * y),
        x_max=1, x_max_val=lambda y: 0,
        y_min=0, y_min_val=lambda x: 0,
        y_max=1, y_max_val=lambda x: 0,
    )

    net = FCNN(n_input_units=2, hidden_units=(32, 32))
    with pytest.warns(FutureWarning):
        solution_neural_net_laplace, loss_history = solve2D(
            pde=laplace, condition=bc, xy_min=(0, 0), xy_max=(1, 1),
            net=net, max_epochs=3,
            train_generator=Generator2D((32, 32), (0, 0), (1, 1), method='equally-spaced-noisy',
                                        xy_noise_std=(0.01, 0.01)),
            batch_size=64
        )
    assert isinstance(solution_neural_net_laplace, Solution2D)
    assert isinstance(loss_history, dict)
    keys = ['train_loss', 'valid_loss']
    for key in keys:
        assert key in loss_history
        assert isinstance(loss_history[key], list)
    assert len(loss_history[keys[0]]) == len(loss_history[keys[1]])


# def test_pde_system():
#     def _network_output_2input(net, xs, ys, ith_unit):
#         xys = torch.cat((xs, ys), 1)
#         nn_output = net(xys)
#         if ith_unit is not None:
#             return nn_output[:, ith_unit].reshape(-1, 1)
#         else:
#             return nn_output
#
#     class BCOnU(Condition):
#         """for u(x, y), impose u(x, -1) = u(x, 1) = 0; dudx(0, y) = dudy(L, y) = 0"""
#
#         def __init__(self, x_min, x_max, y_min, y_max):
#             super().__init__()
#             self.x_min = x_min
#             self.x_max = x_max
#             self.y_min = y_min
#             self.y_max = y_max
#
#         def enforce(self, net, x, y):
#             uxy = _network_output_2input(net, x, y, self.ith_unit)
#
#             x_ones = torch.ones_like(x, requires_grad=True)
#             x_ones_min = self.x_min * x_ones
#             x_ones_max = self.x_max * x_ones
#             uxminy = _network_output_2input(net, x_ones_min, y, self.ith_unit)
#             uxmaxy = _network_output_2input(net, x_ones_max, y, self.ith_unit)
#
#             x_tilde = (x - self.x_min) / (self.x_max - self.x_min)
#             y_tilde = (y - self.y_min) / (self.y_max - self.y_min)
#
#             return y_tilde * (1 - y_tilde) * (
#                     uxy - x_tilde * (self.x_max - self.x_min) * diff(uxminy, x_ones_min) \
#                     + 0.5 * x_tilde ** 2 * (self.x_max - self.x_min) * (
#                             diff(uxminy, x_ones_min) - diff(uxmaxy, x_ones_max)
#                     )
#             )
#
#     class BCOnP(Condition):
#         """for p(x, y), impose p(0, y) = p_max; p(L, y) = p_min"""
#
#         def __init__(self, x_min, x_max, p_x_min, p_x_max):
#             super().__init__()
#             self.x_min = x_min
#             self.x_max = x_max
#             self.p_x_min = p_x_min
#             self.p_x_max = p_x_max
#
#         def enforce(self, net, x, y):
#             uxy = _network_output_2input(net, x, y, self.ith_unit)
#             x_tilde = (x - self.x_min) / (self.x_max - self.x_min)
#
#             return (1 - x_tilde) * self.p_x_min + x_tilde * self.p_x_max \
#                    + x_tilde * (1 - x_tilde) * uxy
#
#     L = 2.0
#     mu = 1.0
#     P1, P2 = 1.0, 0.0
#     def poiseuille(u, v, p, x, y):
#         return [
#             mu * (diff(u, x, order=2) + diff(u, y, order=2)) - diff(p, x),
#             mu * (diff(v, x, order=2) + diff(v, y, order=2)) - diff(p, y),
#             diff(u, x) + diff(v, y)
#         ]
#     def zero_divergence(u, v, p, x, y):
#         return torch.sum( (diff(u, x) + diff(v, y))**2 )
#
#     bc_on_u = BCOnU(
#         x_min=0,
#         x_max=L,
#         y_min=-1,
#         y_max=1,
#     )
#     bc_on_v = DirichletBVP2D(
#         x_min=0, x_min_val=lambda y: 0,
#         x_max=L, x_max_val=lambda y: 0,
#         y_min=-1, y_min_val=lambda x: 0,
#         y_max=1, y_max_val=lambda x: 0
#     )
#     bc_on_p = BCOnP(
#         x_min=0,
#         x_max=L,
#         p_x_min=P1,
#         p_x_max=P2,
#     )
#     conditions = [bc_on_u, bc_on_v, bc_on_p]
#
#     nets = [
#         FCNN(n_input_units=2, hidden_units=(32, 32), actv=nn.Softplus)
#         for _ in range(3)
#     ]
#
#     # use one neural network for each dependent variable
#     solution_neural_net_poiseuille, _ = solve2D_system(
#         pde_system=poiseuille, conditions=conditions, xy_min=(0, -1), xy_max=(L, 1),
#         train_generator=Generator2D((32, 32), (0, -1), (L, 1), method='equally-spaced-noisy'),
#         max_epochs=300, batch_size=64, nets=nets, additional_loss_term=zero_divergence,
#         monitor=Monitor2D(check_every=10, xy_min=(0, -1), xy_max=(L, 1))
#     )
#
#     def solution_analytical_poiseuille(xs, ys):
#         us = (P1 - P2) / (L * 2 * mu) * (1 - ys ** 2)
#         vs = np.zeros_like(xs)
#         ps = P1 + (P2 - P1) * xs / L
#         return [us, vs, ps]
#
#     xs, ys = np.linspace(0, L, 101), np.linspace(-1, 1, 101)
#     xx, yy = np.meshgrid(xs, ys)
#     u_ana, v_ana, p_ana = solution_analytical_poiseuille(xx, yy)
#     u_net, v_net, p_net = solution_neural_net_poiseuille(xx, yy, to_numpy=True)
#
#     assert isclose(u_ana, u_net, atol=0.01).all()
#     assert isclose(v_ana, v_net, atol=0.01).all()
#     assert isclose(p_ana, p_net, atol=0.01).all()

def test_arbitrary_boundary():
    def solution_analytical_problem_c(x, y):
        return np.log(1 + x ** 2 + y ** 2)

    def gradient_solution_analytical_problem_c(x, y):
        return 2 * x / (1 + x ** 2 + y ** 2), 2 * y / (1 + x ** 2 + y ** 2),

    # creating control points for Dirichlet boundary conditions

    edge_length = 2.0 / np.sin(np.pi / 3) / 4
    points_on_each_edge = 11
    step_size = edge_length / (points_on_each_edge - 1)

    direction_theta = np.pi * 2 / 3
    left_turn_theta = np.pi * 1 / 3
    right_turn_theta = -np.pi * 2 / 3

    dirichlet_control_points_problem_c = []
    point_x, point_y = 0.0, -1.0
    for i_edge in range(6):
        for i_step in range(points_on_each_edge - 1):
            dirichlet_control_points_problem_c.append(
                DirichletControlPoint(
                    loc=(point_x, point_y),
                    val=solution_analytical_problem_c(point_x, point_y)
                )
            )
            point_x += step_size * np.cos(direction_theta)
            point_y += step_size * np.sin(direction_theta)
        direction_theta += left_turn_theta if (i_edge % 2 == 0) else right_turn_theta

    # dummy control points to form closed domain

    radius_circle = 1.0 / np.sin(np.pi / 6)
    center_circle_x = radius_circle * np.cos(np.pi / 6)
    center_circle_y = 0.0

    dirichlet_control_points_problem_c_dummy = []
    for theta in np.linspace(-np.pi * 5 / 6, np.pi * 5 / 6, 60):
        point_x = center_circle_x + radius_circle * np.cos(theta)
        point_y = center_circle_y + radius_circle * np.sin(theta)
        dirichlet_control_points_problem_c_dummy.append(
            DirichletControlPoint(
                loc=(point_x, point_y),
                val=solution_analytical_problem_c(point_x, point_y)
            )
        )

    # all Dirichlet control points

    dirichlet_control_points_problem_c_all = \
        dirichlet_control_points_problem_c + dirichlet_control_points_problem_c_dummy

    # creating control points for Neumann boundary condition

    edge_length = 2.0 / np.sin(np.pi / 3) / 4
    points_on_each_edge = 11
    step_size = edge_length / (points_on_each_edge - 1)

    normal_theta = np.pi / 6

    direction_theta = -np.pi * 1 / 3
    left_turn_theta = np.pi * 1 / 3
    right_turn_theta = -np.pi * 2 / 3

    neumann_control_points_problem_c = []
    point_x, point_y = 0.0, 1.0
    for i_edge in range(6):
        normal_x = np.cos(normal_theta)
        normal_y = np.sin(normal_theta)

        # skip the points on the "tip", their normal vector is undefined?
        point_x += step_size * np.cos(direction_theta)
        point_y += step_size * np.sin(direction_theta)

        for i_step in range(points_on_each_edge - 2):
            grad_x, grad_y = gradient_solution_analytical_problem_c(point_x, point_y)
            neumann_val = grad_x * normal_x + grad_y * normal_y
            neumann_control_points_problem_c.append(
                NeumannControlPoint(
                    loc=(point_x, point_y),
                    val=neumann_val,
                    normal_vector=(normal_x, normal_y)
                )
            )
            point_x += step_size * np.cos(direction_theta)
            point_y += step_size * np.sin(direction_theta)
        direction_theta += left_turn_theta if (i_edge % 2 == 0) else right_turn_theta
        normal_theta += left_turn_theta if (i_edge % 2 == 0) else right_turn_theta

    # dummy control points to form closed domain

    radius_circle = 1.0 / np.sin(np.pi / 6)
    center_circle_x = -radius_circle * np.cos(np.pi / 6)
    center_circle_y = 0.0

    neumann_control_points_problem_c_dummy = []
    for theta in np.linspace(np.pi * 1 / 6, np.pi * 11 / 6, 60):
        point_x = center_circle_x + radius_circle * np.cos(theta)
        point_y = center_circle_y + radius_circle * np.sin(theta)
        normal_x = np.cos(theta)
        normal_y = np.sin(theta)
        grad_x, grad_y = gradient_solution_analytical_problem_c(point_x, point_y)
        neumann_val = grad_x * normal_x + grad_y * normal_y
        neumann_control_points_problem_c_dummy.append(
            NeumannControlPoint(
                loc=(point_x, point_y),
                val=neumann_val,
                normal_vector=(normal_x, normal_y)
            )
        )

    # all Neumann control points

    neumann_control_points_problem_c_all = \
        neumann_control_points_problem_c + neumann_control_points_problem_c_dummy

    cbc_problem_c = CustomBoundaryCondition(
        center_point=Point(loc=(0.0, 0.0)),
        dirichlet_control_points=dirichlet_control_points_problem_c_all,
        neumann_control_points=neumann_control_points_problem_c_all
    )

    def get_grid(x_from_to, y_from_to, x_n_points=100, y_n_points=100, as_tensor=False):
        x_from, x_to = x_from_to
        y_from, y_to = y_from_to
        if as_tensor:
            x = torch.linspace(x_from, x_to, x_n_points)
            y = torch.linspace(y_from, y_to, y_n_points)
            return torch.meshgrid(x, y)
        else:
            x = np.linspace(x_from, x_to, x_n_points)
            y = np.linspace(y_from, y_to, y_n_points)
            return np.meshgrid(x, y)

    def to_np(tensor):
        return tensor.detach().cpu().numpy()

    xx_train, yy_train = get_grid(
        x_from_to=(-1, 1), y_from_to=(-1, 1),
        x_n_points=28, y_n_points=28,
        as_tensor=True
    )
    is_in_domain_train = cbc_problem_c.in_domain(xx_train, yy_train)
    xx_train, yy_train = to_np(xx_train), to_np(yy_train)
    xx_train, yy_train = xx_train[is_in_domain_train], yy_train[is_in_domain_train]
    train_gen = PredefinedGenerator(xx_train, yy_train)

    xx_valid, yy_valid = get_grid(
        x_from_to=(-1, 1), y_from_to=(-1, 1),
        x_n_points=10, y_n_points=10,
        as_tensor=True
    )
    is_in_domain_valid = cbc_problem_c.in_domain(xx_valid, yy_valid)
    xx_valid, yy_valid = to_np(xx_valid), to_np(yy_valid)
    xx_valid, yy_valid = xx_valid[is_in_domain_valid], yy_valid[is_in_domain_valid]
    valid_gen = PredefinedGenerator(xx_valid, yy_valid)

    def rmse(u, x, y):
        true_u = torch.log(1 + x ** 2 + y ** 2)
        return torch.mean((u - true_u) ** 2) ** 0.5

    # nabla^2 psi(x, y) = (e^(-x))(x-2+y^3+6y)
    def de_problem_c(u, x, y):
        return diff(u, x, order=2) + diff(u, y, order=2) + torch.exp(u) - 1.0 - x ** 2 - y ** 2 - 4.0 / (
                1.0 + x ** 2 + y ** 2) ** 2

    # fully connected network with one hidden layer (100 hidden units with ELU activation)
    net = FCNN(n_input_units=2, hidden_units=(100, 100), actv=nn.ELU)
    adam = optim.Adam(params=net.parameters(), lr=0.001)

    # train on 28 X 28 grid
    with pytest.warns(FutureWarning):
        solution_neural_net_problem_c, history_problem_c = solve2D(
            pde=de_problem_c, condition=cbc_problem_c,
            xy_min=(-1, -1), xy_max=(1, 1),
            train_generator=train_gen, valid_generator=valid_gen,
            net=net, max_epochs=1, batch_size=128, optimizer=adam,
            monitor=Monitor2D(check_every=1, xy_min=(-1, -1), xy_max=(1, 1), valid_generator=valid_gen),
            metrics={'rmse': rmse}
        )

    xs = torch.tensor([p.loc[0] for p in dirichlet_control_points_problem_c], requires_grad=True).reshape(-1, 1)
    ys = torch.tensor([p.loc[1] for p in dirichlet_control_points_problem_c], requires_grad=True).reshape(-1, 1)
    us = solution_neural_net_problem_c(xs, ys, to_numpy=True)
    true_us = solution_analytical_problem_c(to_np(xs), to_np(ys))
    assert isclose(us, true_us, atol=1e-4).all()

    xs = torch.tensor([p.loc[0] for p in neumann_control_points_problem_c], requires_grad=True).reshape(-1, 1)
    ys = torch.tensor([p.loc[1] for p in neumann_control_points_problem_c], requires_grad=True).reshape(-1, 1)
    us = solution_neural_net_problem_c(xs, ys)
    nxs = torch.tensor([p.normal_vector[0] for p in neumann_control_points_problem_c]).reshape(-1, 1)
    nys = torch.tensor([p.normal_vector[1] for p in neumann_control_points_problem_c]).reshape(-1, 1)
    normal_derivative = to_np(nxs * diff(us, xs) + nys * diff(us, ys)).flatten()
    true_normal_derivative = np.array([p.val for p in neumann_control_points_problem_c])
    assert isclose(normal_derivative, true_normal_derivative, atol=1e-2).all()


def test_solution():
    x_grids = 7
    y_grids = 11
    xy_grids = (x_grids, y_grids)
    N_SAMPLES = x_grids * y_grids
    x0, y0 = random.random(), random.random()
    x1, y1 = random.random() + 1, random.random() + 1
    generator = Generator2D(xy_grids, xy_min=(x0, y0), xy_max=(x1, y1), method='equally-spaced')

    u00, u01, u10, u11 = random.random(), random.random(), random.random(), random.random()
    v00, v01, v10, v11 = random.random(), random.random(), random.random(), random.random()

    def get_single_boundary_func(z0, w0, z1, w1):
        net = FCNN(1, 1)
        condition = DirichletBVP(z0, w0, z1, w1)

        def boundary_func(z):
            return condition.enforce(net, z)

        return boundary_func

    def get_all_boundary_funcs(w00, w01, w10, w11):
        wx0 = get_single_boundary_func(y0, w00, y1, w01)
        wx1 = get_single_boundary_func(y0, w10, y1, w11)
        wy0 = get_single_boundary_func(x0, w00, x1, w10)
        wy1 = get_single_boundary_func(x0, w01, x1, w11)
        return wx0, wx1, wy0, wy1

    ux0, ux1, uy0, uy1 = get_all_boundary_funcs(u00, u01, u10, u11)
    vx0, vx1, vy0, vy1 = get_all_boundary_funcs(v00, v01, v10, v11)

    def get_solution(use_single: bool) -> Solution2D:
        conditions = [
            DirichletBVP2D(x0, ux0, x1, ux1, y0, uy0, y1, uy1),
            DirichletBVP2D(x0, vx0, x1, vx1, y0, vy0, y1, vy1),
        ]
        if use_single:
            net = FCNN(2, 2)
            for i, cond in enumerate(conditions):
                with pytest.warns(DeprecationWarning):
                    cond.set_impose_on(i)
            return Solution2D(net, conditions)
        else:
            nets = [FCNN(2, 1), FCNN(2, 1)]
            return Solution2D(nets, conditions)

    def check_output(uv, shape, type, msg=""):
        msg += " "
        assert isinstance(uv, (list, tuple)), msg + "returned type is not a list"
        assert len(uv) == 2, msg + "returned length is not 2"
        assert isinstance(uv[0], type) and isinstance(uv[1], type), msg + f"returned element is not {type}"
        u, v = uv
        assert u.shape == shape and v.shape == shape, msg + f"returned element shape is not {shape}"
        u, v = u.reshape(*xy_grids), v.reshape(*xy_grids)
        x = torch.linspace(x0, x1, steps=x_grids, requires_grad=True).reshape(-1, 1)
        y = torch.linspace(y0, y1, steps=y_grids, requires_grad=True).reshape(-1, 1)

        if type == torch.Tensor:
            check_close = lambda a, b: torch.isclose(a, b).all()
        elif type == np.ndarray:
            check_close = lambda a, b: np.isclose(a, b.cpu().detach().numpy()).all
        else:
            raise ValueError(f"Unrecognized type={type}")

        assert check_close(u[0, :], torch.flatten(ux0(y))), msg + "u on x0 not satisfied"
        assert check_close(u[-1, :], torch.flatten(ux1(y))), msg + "u on x1 not satisfied"
        assert check_close(u[:, 0], torch.flatten(uy0(x))), msg + "u on y0 not satisfied"
        assert check_close(u[:, -1], torch.flatten(uy1(x))), msg + "u on y1 not satisfied"

        assert check_close(v[0, :], torch.flatten(vx0(y))), msg + "v on x0 not satisfied"
        assert check_close(v[-1, :], torch.flatten(vx1(y))), msg + "v on x1 not satisfied"
        assert check_close(v[:, 0], torch.flatten(vy0(x))), msg + "v on y0 not satisfied"
        assert check_close(v[:, -1], torch.flatten(vy1(x))), msg + "v on y1 not satisfied"

    for use_single in [True, False]:
        solution = get_solution(use_single=use_single)
        xs, ys = generator.get_examples()
        us = solution(xs, ys)
        check_output(us, shape=(N_SAMPLES,), type=torch.Tensor, msg=f"[use_single={use_single}]")
        with pytest.warns(FutureWarning):
            us = solution(xs, ys, as_type='np')
        check_output(us, shape=(N_SAMPLES,), type=np.ndarray, msg=f"[use_single={use_single}]")
        us = solution(xs, ys, to_numpy=True)
        check_output(us, shape=(N_SAMPLES,), type=np.ndarray, msg=f"[use_single={use_single}]")

        xs, ys = xs.reshape(-1, 1), ys.reshape(-1, 1)
        us = solution(xs, ys)
        check_output(us, shape=(N_SAMPLES, 1), type=torch.Tensor, msg=f"[use_single={use_single}]")
        with pytest.warns(FutureWarning):
            us = solution(xs, ys, as_type='np')
        check_output(us, shape=(N_SAMPLES, 1), type=np.ndarray, msg=f"[use_single={use_single}]")
        us = solution(xs, ys, to_numpy=True)
        check_output(us, shape=(N_SAMPLES, 1), type=np.ndarray, msg=f"[use_single={use_single}]")
