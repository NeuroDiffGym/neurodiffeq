import numpy as np
import pytest
from neurodiffeq.conditions import NoCondition
from neurodiffeq.networks import FCNN
from neurodiffeq.monitors import Monitor2D, MetricsMonitor, Monitor1D, StreamPlotMonitor2D

N_FUNCTIONS = 2
EMPTY_HISTORY = dict(train_loss=[], valid_loss=[])
HISTORY_10 = dict(
    train_loss=list(np.random.random(10)),
    valid_loss=list(np.random.random(10)),
)


@pytest.mark.parametrize(argnames='history', argvalues=[EMPTY_HISTORY, HISTORY_10])
@pytest.mark.parametrize(argnames='solution_style', argvalues=['curves', 'heatmap'])
def test_monitor_2d(solution_style, history):
    monitor = Monitor2D((0, 0), (1, 1), check_every=100, solution_style=solution_style)
    nets = [FCNN(2, 1) for _ in range(N_FUNCTIONS)]
    conditions = [NoCondition() for _ in range(N_FUNCTIONS)]
    monitor.check(nets, conditions, history)


@pytest.mark.parametrize(argnames='history', argvalues=[EMPTY_HISTORY, HISTORY_10])
def test_metrics_mointor(history):
    monitor = MetricsMonitor(check_every=10)
    nets = [FCNN(2, 1) for _ in range(N_FUNCTIONS)]
    conditions = [NoCondition() for _ in range(N_FUNCTIONS)]
    monitor.check(nets, conditions, history)


@pytest.mark.parametrize(argnames='history', argvalues=[EMPTY_HISTORY, HISTORY_10])
def test_monitor_1d(history):
    monitor = Monitor1D(0, 1)
    nets = [FCNN() for _ in range(N_FUNCTIONS)]
    conditions = [NoCondition() for _ in range(N_FUNCTIONS)]
    monitor.check(nets, conditions, history=history)


@pytest.mark.parametrize(argnames='specify_field_names', argvalues=[False, True])
@pytest.mark.parametrize(argnames='mask_fn', argvalues=[None, lambda x, y: (x ** 2 + y ** 2 < 1)])
@pytest.mark.parametrize(argnames=['nx', 'ny'], argvalues=[(30, 50), (40, 20)])
def test_stream_plot_monitor(mask_fn, nx, ny, specify_field_names):
    nets = [FCNN(2, 1, hidden_units=(3,)) for _ in range(5)]
    conditions = [NoCondition() for _ in nets]
    pairs = [(0, 1), (2, 3), (0, 3), 4, 2]

    if specify_field_names:
        field_names = [str(i) for i in range(len(pairs))]
    else:
        field_names = None
    monitor = StreamPlotMonitor2D(
        xy_min=(-1, -1),
        xy_max=(1, 1),
        nx=nx,
        ny=ny,
        pairs=pairs,
        mask_fn=mask_fn,
        equal_aspect=True,
        field_names=field_names,
    )

    if specify_field_names:
        with pytest.raises(ValueError):
            StreamPlotMonitor2D(
                xy_min=(-1, -1),
                xy_max=(1, 1),
                pairs=[0, 1],
                field_names=['a', 'b', 'c'],
            )

    monitor.check(nets, conditions, history=None)
    monitor.check(nets[::-1], conditions, history=None)
