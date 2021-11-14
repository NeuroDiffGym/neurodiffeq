import sys
import pytest
import random
import numpy as np
import torch
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.conditions import IVP, NoCondition
from neurodiffeq.generators import Generator1D
from neurodiffeq.solvers import GenericSolver, GenericSolution, BaseSolver
from neurodiffeq.solvers import Solver1D, Solver2D, SolverSpherical
from neurodiffeq.solvers import SolutionSphericalHarmonics

SPECIFIC_SOLVERS = [Solver1D, Solver2D, SolverSpherical]

N_SAMPLES = 64
T_MIN, T_MAX = 0.0, 1.0
DIFF_EQS = lambda u, t: [diff(u, t) + u]
CONDITIONS = [IVP(0, 1)]


@pytest.fixture
def generators():
    return dict(
        train=Generator1D(64, t_min=T_MIN, t_max=T_MAX, method='uniform'),
        valid=Generator1D(64, t_min=T_MIN, t_max=T_MAX, method='equally-spaced'),
    )


@pytest.fixture
def solver(generators):
    return GenericSolver(
        diff_eqs=DIFF_EQS,
        conditions=CONDITIONS,
        train_generator=generators['train'],
        valid_generator=generators['valid'],
        n_input_units=1,
        n_output_units=1,
    )


def test_legacies(solver, generators):
    solver.fit(1)

    assert solver.batch == solver._batch
    with pytest.warns(FutureWarning):
        assert solver._batch_examples == solver._batch

    with pytest.raises(TypeError), pytest.warns(FutureWarning):
        GenericSolver(
            diff_eqs=DIFF_EQS,
            conditions=CONDITIONS,
            train_generator=generators['train'],
            valid_generator=generators['valid'],
            criterion=lambda residuals, zeros: (residuals ** 2).mean(),
            n_input_units=1,
            n_output_units=1,
        ).fit(1)

    class SolverWithLegacyAdditionalLoss(BaseSolver):
        def additional_loss(self, funcs, key):
            return 0

    with pytest.raises(TypeError), pytest.warns(FutureWarning):
        SolverWithLegacyAdditionalLoss(
            diff_eqs=DIFF_EQS,
            conditions=CONDITIONS,
            train_generator=generators['train'],
            valid_generator=generators['valid'],
            n_input_units=1,
            n_output_units=1,
        ).fit(1)

    with pytest.warns(FutureWarning):
        GenericSolver(
            diff_eqs=DIFF_EQS,
            conditions=CONDITIONS,
            train_generator=generators['train'],
            valid_generator=generators['valid'],
            n_input_units=1,
            n_output_units=1,
            shuffle=True,
        )


def test_missing_generator(generators):
    with pytest.raises(ValueError):
        GenericSolver(
            diff_eqs=DIFF_EQS,
            conditions=CONDITIONS,
            train_generator=generators['train'],
            n_input_units=1,
            n_output_units=1,
        )
    with pytest.raises(ValueError):
        GenericSolver(
            diff_eqs=DIFF_EQS,
            conditions=CONDITIONS,
            valid_generator=generators['valid'],
            n_input_units=1,
            n_output_units=1,
        )
    with pytest.raises(ValueError):
        GenericSolver(
            diff_eqs=DIFF_EQS,
            conditions=CONDITIONS,
            n_input_units=1,
            n_output_units=1,
        )


def test_update_nonexistent_key_in_history(solver):
    with pytest.raises(KeyError):
        solver._update_history(1.0, metric_type='bad name', key='train')

    with pytest.raises(KeyError):
        solver._update_history(1.0, metric_type='bad name', key='valid')


@pytest.mark.parametrize(argnames='key', argvalues=['train', 'valid'])
def test_update_train_valid_history(solver, key):
    for _ in range(5):
        r = random.random()
        getattr(solver, f'_update_{key}_history')(value=r, metric_type='loss')
        assert solver.metrics_history[f'{key}_loss'][-1] == r


@pytest.mark.parametrize(argnames='key', argvalues=['train', 'valid'])
def test_generate_train_valid_batch(solver, key):
    batch = getattr(solver, f'_generate_{key}_batch')()
    assert batch == solver._batch[key]


def test_no_validation(solver):
    solver.n_batches['valid'] = 0
    solver.fit(1)


def test_lbfgs(generators):
    nets = [FCNN()]
    lbfgs = torch.optim.LBFGS(params=nets[0].parameters(), lr=1e-3)
    GenericSolver(
        diff_eqs=DIFF_EQS,
        conditions=CONDITIONS,
        train_generator=generators['train'],
        valid_generator=generators['valid'],
        nets=nets,
        optimizer=lbfgs,
        n_input_units=1,
        n_output_units=1,
    ).fit(1)


def test_early_stopping(solver):
    from neurodiffeq.callbacks import StopCallback

    cb = StopCallback()
    solver.fit(max_epochs=10, callbacks=[cb])
    assert solver.global_epoch == 1


def test_invalid_get_internals(solver):
    with pytest.raises(ValueError):
        solver.get_internals(['generator'], return_type='bad type')


def test_tqdm(solver, capfd):
    solver.fit(max_epochs=20, tqdm_file=sys.stdout)
    tqdm_desc = 'Training Progress'
    stdout, stderr = capfd.readouterr()
    assert tqdm_desc in stdout
    assert tqdm_desc not in stderr

    solver.fit(max_epochs=20, tqdm_file=sys.stderr)
    stdout, stderr = capfd.readouterr()
    assert tqdm_desc not in stdout
    assert tqdm_desc in stderr

    solver.fit(max_epochs=20, tqdm_file=None)
    stdout, stderr = capfd.readouterr()
    assert tqdm_desc not in stdout
    assert tqdm_desc not in stderr


@pytest.mark.parametrize(argnames='best', argvalues=[True, False])
@pytest.mark.parametrize(argnames='ts', argvalues=[np.linspace(0, 1, 50), torch.linspace(0, 1, 50)])
@pytest.mark.parametrize(argnames='to_numpy', argvalues=[True, False])
@pytest.mark.parametrize(argnames='first_shape', argvalues=[(-1,), (-1, 1)])
# TODO more scenarios (and best is not tested yet)
def test_get_residual(solver, best, ts, to_numpy, first_shape):
    solver.best_nets = [FCNN(1, 1)]
    ts = ts.reshape(*first_shape)
    rs = solver.get_residuals(ts, to_numpy=to_numpy, best=best)
    if to_numpy:
        assert isinstance(rs, np.ndarray)
    else:
        assert isinstance(rs, torch.Tensor)
    assert rs.shape == rs.reshape(first_shape).shape


def test_generic_solver(solver):
    solver.fit(1)


def test_generic_solution(solver):
    solution = solver.get_solution(best=False)
    t0 = torch.zeros((1, 1))
    assert (solution(t0) == 1).all()


@pytest.mark.parametrize(argnames='SolverClass', argvalues=SPECIFIC_SOLVERS)
def test_missing_domain(SolverClass, generators):
    with pytest.raises(ValueError):
        SolverClass(DIFF_EQS, CONDITIONS)
    with pytest.raises(ValueError):
        SolverClass(DIFF_EQS, CONDITIONS, train_generator=generators['train'])
    with pytest.raises(ValueError):
        SolverClass(DIFF_EQS, CONDITIONS, valid_generator=generators['valid'])


def test_default_generator():
    Solver1D(DIFF_EQS, CONDITIONS, t_min=0, t_max=1)
    Solver2D(DIFF_EQS, CONDITIONS, xy_min=(0, 0), xy_max=(1, 1))
    SolverSpherical(DIFF_EQS, CONDITIONS, r_min=0, r_max=1)


@pytest.mark.parametrize(argnames='SolverClass', argvalues=SPECIFIC_SOLVERS)
def test_get_internals_variables(SolverClass, generators):
    solver = SolverClass(
        DIFF_EQS,
        CONDITIONS,
        train_generator=generators['train'],
        valid_generator=generators['valid'],
    )
    d1 = BaseSolver._get_internal_variables(solver)
    d2 = solver._get_internal_variables()

    for k in d1:
        assert k in d2, f'{k} not in {d2.keys()}'
        assert d2[k] == d1[k], f'd1[{k}] == {d1[k]} != d2[{k}] == {d2[k]}'


def test_legacy_max_degree_in_solution_spherical_harmonics():
    with pytest.warns(FutureWarning):
        SolutionSphericalHarmonics(
            nets=[FCNN(1, 10)],
            conditions=[NoCondition()],
            max_degree=4,
        )
