import numpy as np
import pytest
from neurodiffeq.conditions import NoCondition
from neurodiffeq.networks import FCNN
from neurodiffeq.monitors import Monitor2D, MetricsMonitor

N_FUNCTIONS = 2
EMPTY_HISTORY = dict(train_loss=[], valid_loss=[])
HISTORY_10 = dict(
    train_loss=list(np.random.random(10)),
    valid_loss=list(np.random.random(10)),
)


@pytest.mark.parametrize(argnames='history', argvalues=[EMPTY_HISTORY, HISTORY_10])
@pytest.mark.parametrize(argnames='solution_style', argvalues=['curves', 'heatmap'])
def test_monitor2d(solution_style, history):
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
