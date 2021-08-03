from neurodiffeq.hypersolver.hypersolver import DiscreteSolution1D
from neurodiffeq.hypersolver.numerical_solvers import Euler
import torch
import numpy as np


def test_discrete_solution_1d():
    try_solver = Euler()
    ts, us = try_solver.solve(lambda u, t: [-u], 1, 0, 1, 100)

    solution = DiscreteSolution1D(ts, us)

    other_ts = torch.rand(100)
    assert min(ts) < min(other_ts)
    assert max(other_ts) < max(ts)

    other_us, = solution(other_ts)
    true_us = torch.exp(-other_ts)
    assert torch.allclose(other_us, true_us, rtol=1e-2)

    ts, fs, gs = try_solver.solve(lambda u, v, t: [v, -u], [1, 1], 0, np.pi, 314)
    solution = DiscreteSolution1D(ts, fs, gs)
    other_ts = torch.rand(100) * np.pi

    other_fs, other_gs = solution(other_ts)
    fs_true = torch.sin(other_ts) + torch.cos(other_ts)
    gs_true = torch.cos(other_ts) - torch.sin(other_ts)

    assert torch.allclose(other_fs, fs_true, rtol=1e-2, atol=1e-2)
    assert torch.allclose(other_gs, gs_true, rtol=1e-2, atol=1e-2)
