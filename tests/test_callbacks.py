import torch
import dill
import shutil
from pathlib import Path
import os
import pytest
from neurodiffeq import diff
from neurodiffeq.conditions import NoCondition
from neurodiffeq.solvers import Solver1D
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.callbacks import MonitorCallback, CheckpointCallback, ReportOnFitCallback, BaseCallback
from neurodiffeq.callbacks import AndCallback, OrCallback, NotCallback, XorCallback, TrueCallback, FalseCallback
from neurodiffeq.callbacks import OnFirstLocal, OnFirstGlobal, OnLastLocal, PeriodGlobal, PeriodLocal
from neurodiffeq.callbacks import ClosedIntervalGlobal, ClosedIntervalLocal, Random


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


@pytest.fixture
def dummy_cb():
    class PassCallback(BaseCallback):
        def __call__(self, solver):
            pass

    return PassCallback()


@pytest.fixture
def true_cb():
    return TrueCallback()


@pytest.fixture
def false_cb():
    return FalseCallback()


def _set_global_epoch(solver, epoch):
    solver.metrics_history['train_loss'] = [0.0] * epoch


def test_monitor_callback(solver, tmp_dir):
    # pretend we have trained for 100 epochs
    _set_global_epoch(solver, 100)
    solver.local_epoch = 1
    assert solver.global_epoch == 100

    monitor = Monitor1D(0, 1, check_every=100)
    assert not MonitorCallback(monitor, check_against_local=True).to_repaint(solver)
    assert MonitorCallback(monitor, check_against_local=False).to_repaint(solver)

    # pretend we're training for the last epoch
    solver._max_local_epoch = 50
    solver.local_epoch = 50
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


def test_checkpoint_callback(solver, tmp_dir):
    callback = CheckpointCallback(ckpt_dir=tmp_dir)
    solver._max_local_epoch = 50
    solver.local_epoch = 49
    callback(solver)
    assert os.listdir(tmp_dir) == []

    solver.local_epoch = 50
    callback(solver)
    content = os.listdir(tmp_dir)
    assert len(content) == 1 and content[0].endswith('.internals')

    with open(tmp_dir / content[0], 'rb') as f:
        internals = dill.load(f)

    assert isinstance(internals, dict)
    assert isinstance(internals.get('nets'), list)
    for net in internals.get('nets'):
        assert isinstance(net, torch.nn.Module)


def test_report_on_fit_callback(solver):
    callback = ReportOnFitCallback()
    callback(solver)


def test_true_callback(solver, true_cb):
    assert true_cb.condition(solver)


def test_false_callback(solver, false_cb):
    assert not false_cb.condition(dummy_cb)


def test_and_callback(solver, true_cb, false_cb):
    assert (true_cb & true_cb).condition(solver)
    assert not (false_cb & false_cb).condition(solver)
    assert not (true_cb & false_cb).condition(solver)
    assert not (false_cb & true_cb).condition(solver)


def test_or_callback(solver, true_cb, false_cb):
    assert (true_cb | true_cb).condition(solver)
    assert (true_cb | false_cb).condition(solver)
    assert (false_cb | true_cb).condition(solver)
    assert not (false_cb | false_cb).condition(solver)


def test_not_callback(solver, true_cb, false_cb):
    assert not (~true_cb).condition(solver)
    assert (~false_cb).condition(solver)


def test_xor_callback(solver, true_cb, false_cb):
    assert not (true_cb ^ true_cb).condition(solver)
    assert (true_cb ^ false_cb).condition(solver)
    assert (false_cb ^ true_cb).condition(solver)
    assert not (false_cb ^ false_cb).condition(solver)


def test_on_first_local(solver):
    solver.local_epoch = 1
    assert OnFirstLocal().condition(solver)
    solver.local_epoch = 2
    assert not OnFirstLocal().condition(solver)


def test_on_first_global(solver):
    _set_global_epoch(solver, 1)
    assert OnFirstGlobal().condition(solver)
    _set_global_epoch(solver, 2)
    assert not OnFirstGlobal().condition(solver)


def test_on_last_local(solver):
    solver.local_epoch = 9
    solver._max_local_epoch = 10
    assert not OnLastLocal().condition(solver)
    solver.local_epoch = 10
    assert OnLastLocal().condition(solver)


def test_period_local(solver):
    period = 3
    n_periods = 10
    for i in range(1, n_periods):
        solver.local_epoch = i * period
        assert PeriodLocal(period=period).condition(solver)

    for offset in range(1, period):
        for i in range(n_periods):
            solver.local_epoch = i * period + offset
            assert not PeriodLocal(period=period).condition(solver)


def test_period_global(solver):
    period = 3
    n_periods = 10
    for i in range(1, n_periods):
        _set_global_epoch(solver, i * period)
        assert PeriodGlobal(period=period).condition(solver)

    for offset in range(1, period):
        for i in range(n_periods):
            _set_global_epoch(solver, i * period + offset)
            assert not PeriodGlobal(period=period).condition(solver)


def test_closed_interval_local(solver):
    e_min, e_max = [5, 7]
    test_range = list(range(1, 10))
    callback = ClosedIntervalLocal(min=e_min, max=e_max)
    for e in test_range:
        solver.local_epoch = e
        assert callback.condition(solver) == (e_min <= e <= e_max)

    callback = ClosedIntervalLocal(min=e_min)
    for e in test_range:
        solver.local_epoch = e
        assert callback.condition(solver) == (e_min <= e)

    callback = ClosedIntervalLocal(max=e_max)
    for e in test_range:
        solver.local_epoch = e
        assert callback.condition(solver) == (e <= e_max)


def test_closed_interval_global(solver):
    e_min, e_max = [5, 7]
    test_range = list(range(1, 10))
    callback = ClosedIntervalGlobal(min=e_min, max=e_max)
    for e in test_range:
        _set_global_epoch(solver, e)
        assert callback.condition(solver) == (e_min <= e <= e_max)

    callback = ClosedIntervalGlobal(min=e_min)
    for e in test_range:
        _set_global_epoch(solver, e)
        assert callback.condition(solver) == (e_min <= e)

    callback = ClosedIntervalGlobal(max=e_max)
    for e in test_range:
        _set_global_epoch(solver, e)
        assert callback.condition(solver) == (e <= e_max)


def test_random(solver):
    callback = Random(0.5)
    callback.condition(solver)
    Random(0.0)
    Random(1.0)
    with pytest.raises(ValueError):
        Random(-0.1)
    with pytest.raises(ValueError):
        Random(1.1)
