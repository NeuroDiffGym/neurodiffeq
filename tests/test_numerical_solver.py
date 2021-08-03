from neurodiffeq.hypersolver.numerical_solvers import Euler
import torch
import numpy as np


def test_euler():
    try_solver = Euler()
    ts, us = try_solver.solve(lambda u, t: [-u], 1, 0, 1, 100)
    vs = torch.exp(-ts)
    print(us.shape, vs.shape)
    assert torch.allclose(us, vs, rtol=1e-2)

    ts, fs, gs = try_solver.solve(lambda u, v, t: [v, -u], [1, 1], 0, np.pi, 314)
    fs_true = torch.sin(ts) + torch.cos(ts)
    gs_true = torch.cos(ts) - torch.sin(ts)
    assert torch.allclose(fs, fs_true, rtol=1e-2, atol=1e-2)
    assert torch.allclose(gs, gs_true, rtol=1e-2, atol=1e-2)