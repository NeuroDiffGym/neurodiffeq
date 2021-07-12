from random import random
import torch
from torch.utils.tensorboard import SummaryWriter
import dill
import shutil
from pathlib import Path
import os
import pytest
from neurodiffeq import diff
from neurodiffeq.conditions import NoCondition
from neurodiffeq.solvers import Solver1D
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.callbacks import MonitorCallback, CheckpointCallback, ReportOnFitCallback, \
    BaseCallback, SimpleTensorboardCallback, ReportCallback, ActionCallback
from neurodiffeq.callbacks import TrueCallback, FalseCallback, ConditionCallback
from neurodiffeq.callbacks import OnFirstLocal, OnFirstGlobal, OnLastLocal, PeriodGlobal, PeriodLocal
from neurodiffeq.callbacks import ClosedIntervalGlobal, ClosedIntervalLocal, Random
from neurodiffeq.callbacks import RepeatedMetricDown, RepeatedMetricUp, RepeatedMetricDiverge, RepeatedMetricConverge
from neurodiffeq.callbacks import _RepeatedMetricChange
from neurodiffeq.callbacks import EveCallback, StopCallback


@pytest.fixture
def tmp_dir():
    tmp = Path('.') / 'test-tmp'
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
    class PassCallback(ActionCallback):
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
    monitor = Monitor1D(0, 1, check_every=30)
    with pytest.warns(FutureWarning):
        MonitorCallback(monitor, check_against_local=True)
    with pytest.warns(FutureWarning):
        MonitorCallback(monitor, check_against='local')
    with pytest.warns(FutureWarning):
        MonitorCallback(monitor, check_against='global')
    with pytest.warns(FutureWarning):
        MonitorCallback(monitor, repaint_last=True)

    callback = MonitorCallback(monitor, fig_dir=tmp_dir)
    assert callback.logger
    callback(solver)

    assert tmp_dir.exists() and tmp_dir.is_dir()


def test_checkpoint_callback(solver, tmp_dir):
    callback = CheckpointCallback(ckpt_dir=tmp_dir)
    callback(solver)
    content = os.listdir(tmp_dir)
    assert len(content) == 1 and content[0].endswith('.internals')

    with open(tmp_dir / content[0], 'rb') as f:
        internals = dill.load(f)

    assert isinstance(internals, dict)
    assert isinstance(internals.get('nets'), list)
    for net in internals.get('nets'):
        assert isinstance(net, torch.nn.Module)


def test_report_callback(solver):
    callback = ReportCallback()
    callback(solver)
    with pytest.warns(FutureWarning):
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


def test_repeated_metric_change(solver):
    class Callback(_RepeatedMetricChange):
        def _last_satisfied(self, last, second2last):
            return True

    for phase in ['train', 'valid']:
        use_train = (phase == 'train')
        for metric in ['loss', 'mse', 'something_else']:
            for repetition in range(1, 5):
                callback = Callback(use_train=use_train, metric=metric, repetition=repetition)
                assert callback.key == f'{phase}_{metric}'
                assert callback.times_required == repetition
                assert callback.so_far == 0


def test_repeated_metric_down(solver):
    repetition = 10
    value = 0.0
    at_least_by = 1.0
    callback = RepeatedMetricDown(at_least_by=at_least_by, use_train=True, metric='loss', repetition=repetition)
    for i in range(repetition + 1):
        value -= at_least_by
        solver.metrics_history['train_loss'].append(value)
        if i == repetition:
            assert callback.condition(solver)
        else:
            assert not callback.condition(solver)


def test_repeated_metric_up(solver):
    repetition = 10
    value = 0.0
    at_least_by = 1.0
    callback = RepeatedMetricUp(at_least_by=at_least_by, use_train=True, metric='loss', repetition=repetition)
    for i in range(repetition + 1):
        value += at_least_by
        solver.metrics_history['train_loss'].append(value)
        if i == repetition:
            assert callback.condition(solver)
        else:
            assert not callback.condition(solver)


def test_repeated_converge(solver):
    repetition = 10
    value = 0.0
    epsilon = 0.01
    callback = RepeatedMetricConverge(epsilon=epsilon, use_train=True, metric='loss', repetition=repetition)
    for i in range(repetition + 1):
        value += random() * epsilon * (-1) ** int(random() < 0.5)
        solver.metrics_history['train_loss'].append(value)
        if i == repetition:
            assert callback.condition(solver)
        else:
            assert not callback.condition(solver)


def test_repeated_diverge(solver):
    repetition = 10
    value = 0.0
    gap = 10.0
    callback = RepeatedMetricDiverge(gap=gap, use_train=True, metric='loss', repetition=repetition)
    for i in range(repetition + 1):
        value += (gap / random()) * (-1) ** int(random() < 0.5)
        solver.metrics_history['train_loss'].append(value)
        if i == repetition:
            assert callback.condition(solver)
        else:
            assert not callback.condition(solver)


def test_eve_callback(solver):
    BASE_VALUE = 1000.0
    DOUBLE_AT = 0.5
    N_0 = 3
    callback = EveCallback(base_value=BASE_VALUE, double_at=DOUBLE_AT, n_0=N_0)
    for i in range(5):
        solver.metrics_history['train_loss'] = [BASE_VALUE * (DOUBLE_AT ** i)]
        callback(solver)
        assert solver.n_batches['train'] == (2 ** i) * N_0

    N_MAX = 16
    callback = EveCallback(base_value=BASE_VALUE, double_at=DOUBLE_AT, n_max=N_MAX)
    solver.metrics_history['train_loss'] = [BASE_VALUE * (DOUBLE_AT ** 10)]
    callback(solver)
    assert solver.n_batches['train'] == N_MAX


def test_stop_callback(solver):
    callback = StopCallback()
    callback(solver)
    assert solver._stop_training


def test_tensorboard_callback(solver, tmp_dir):
    writer = SummaryWriter(tmp_dir)
    callback = SimpleTensorboardCallback(writer=writer)
    for i in range(10):
        solver.metrics_history['train_loss'].append(float(i))
        callback(solver)

    callback = SimpleTensorboardCallback()
    for i in range(10):
        solver.metrics_history['train_loss'].append(float(i))
        callback(solver)

    default_path = Path('.') / 'runs'
    assert os.path.isdir(default_path)
    shutil.rmtree(default_path, ignore_errors=True)
