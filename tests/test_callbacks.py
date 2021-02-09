import torch
import shutil
from pathlib import Path
import os
import pytest
from neurodiffeq import diff
from neurodiffeq.conditions import NoCondition
from neurodiffeq.solvers import Solver1D
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.callbacks import MonitorCallback, CheckpointCallback


@pytest.fixture
def tmp_dir():
    tmp = Path('./test-tmp')
    yield tmp
    if tmp.exists() and tmp.is_dir():
        shutil.rmtree(tmp)


@pytest.fixture
def solver():
    return Solver1D(
        ode_system=lambda u, t: [u + diff(u, t)],
        conditions=[NoCondition()],
        t_min=0.0,
        t_max=1.0,
    )


def test_monitor_callback(solver, tmp_dir):
    # pretend we have trained for 99 epochs
    solver.metrics_history['train_loss'] = [0.0] * 99
    assert solver.global_epoch == 99
    monitor = Monitor1D(0, 1, check_every=100)
    assert MonitorCallback(monitor, check_against_local=False).to_repaint(solver)
    assert not MonitorCallback(monitor, check_against_local=True).to_repaint(solver)

    # pretend we're training for the last epoch
    solver._max_local_epoch = 50
    solver.local_epoch = 49
    monitor = Monitor1D(0, 1, check_every=30)
    assert MonitorCallback(monitor, check_against_local=True, repaint_last=True).to_repaint(solver)

    with pytest.warns(FutureWarning):
        callback = MonitorCallback(monitor, check_against='local')
        assert callback.check_against_local
    with pytest.warns(FutureWarning):
        callback = MonitorCallback(monitor, check_against='global')
        assert not callback.check_against_local
    with pytest.raises(TypeError), pytest.warns(FutureWarning):
        MonitorCallback(monitor, check_against='something else')

    callback = MonitorCallback(monitor, fig_dir=tmp_dir)
    assert callback.logger
    callback(solver)

    assert tmp_dir.exists() and tmp_dir.is_dir()
