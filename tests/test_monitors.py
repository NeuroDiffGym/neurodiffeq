import pytest
from neurodiffeq.conditions import NoCondition
from neurodiffeq.networks import FCNN
from neurodiffeq.monitors import Monitor2D

N_FUNCTIONS = 2
EMPTY_HISTORY = dict(train_loss=[], valid_loss=[])


@pytest.mark.parametrize(argnames='solution_style', argvalues=['curves', 'heatmap'])
def test_monitor2d(solution_style):
    monitor = Monitor2D((0, 0), (1, 1), check_every=100, solution_style=solution_style)
    nets = [FCNN(2, 1) for _ in range(N_FUNCTIONS)]
    conditions = [NoCondition() for _ in range(N_FUNCTIONS)]
    monitor.check(nets, conditions, EMPTY_HISTORY)
