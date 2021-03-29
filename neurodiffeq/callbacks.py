import os
import dill
import warnings
import random
import numpy as np
from datetime import datetime
import logging
from .utils import safe_mkdir as _safe_mkdir
from ._version_utils import deprecated_alias, warn_deprecate_class
from abc import ABC, abstractmethod


class _LoggerMixin:
    r"""A mix-in class that has a standard Python `logger`.

    :param logger: The logger or its name (str). Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, logger=None):
        if not logger:
            self.logger = logging.getLogger('root')
        elif isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        else:
            self.logger = logger


class BaseCallback(ABC, _LoggerMixin):
    r"""Base class of all callbacks.
    The class should not be directly subclassed. Instead, subclass `ActionCallback` or `ConditionCallback`.

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, logger=None):
        _LoggerMixin.__init__(self, logger=logger)

    @abstractmethod
    def __call__(self, solver):
        pass  # pragma: no cover


class ActionCallback(BaseCallback):
    r"""Base class of action callbacks.
    Custom callbacks that *performs an action* should subclass this class.

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def conditioned_on(self, condition_callback):
        if not isinstance(condition_callback, ConditionCallback):
            raise TypeError(f'{condition_callback} is not an instance of ConditionCallback')
        return condition_callback.set_action_callback(self)


class MonitorCallback(ActionCallback):
    r"""A callback for updating the monitor plots (and optionally saving the fig to disk).

    :param monitor: The underlying monitor responsible for plotting solutions.
    :type monitor: `neurodiffeq.monitors.BaseMonitor`
    :param fig_dir: Directory for saving monitor figs; if not specified, figs will not be saved.
    :type fig_dir: str
    :param format: Format for saving figures: {'jpg', 'png' (default), ...}.
    :type format: str
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, monitor, fig_dir=None, format=None, logger=None, **kwargs):
        super(MonitorCallback, self).__init__(logger=logger)
        self.monitor = monitor
        self.fig_dir = fig_dir
        self.format = format or 'png'

        # deprecation warnings
        for kw in ['check_against_local', 'check_against']:
            if kwargs.pop(kw, None) is not None:
                warnings.warn(
                    f'`Passing {kw}` is deprecated and ignored, '
                    f'use a `PeriodLocal` or `PeriodGlobal` to control how frequently the callback is run',
                    FutureWarning,
                )
        if kwargs.pop('repaint_last', None) is not None:
            warnings.warn(
                f'`Passing repaint_last is deprecated and ignored, '
                f'Use a `OnLastLocal` callback to plot on last epoch',
                FutureWarning,
            )
        if kwargs:
            raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')

        # make dir for figs
        if fig_dir:
            _safe_mkdir(fig_dir)

    def __call__(self, solver):
        self.monitor.check(
            solver.nets,
            solver.conditions,
            history=solver.metrics_history,
        )
        if self.fig_dir:
            pic_path = os.path.join(self.fig_dir, f"epoch-{solver.global_epoch}.{self.format}")
            self.monitor.fig.savefig(pic_path)
            self.logger.info(f'plot saved to {pic_path}')


class StopCallback(ActionCallback):
    r"""A callback that stops the training/validation process and terminates the ``solver.fit()`` call.

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``

    .. note::
        This callback should always be used together with a `ConditionCallback`,
        otherwise the ``solver.fit()`` call will exit after first epoch.
    """

    def __call__(self, solver):
        solver._stop_training = True


class CheckpointCallback(ActionCallback):
    r"""A callback that saves the networks (and their weights) to the disk.

    :param ckpt_dir:
        The directory to save model checkpoints.
        If non-existent, the directory is automatically created at instantiation time.
    :type ckpt_dir: str
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``

    .. note::
        Unless the callback is called twice within the same second, new checkpoints will not overwrite existing ones.
    """

    def __init__(self, ckpt_dir, logger=None):
        super(CheckpointCallback, self).__init__(logger=logger)
        self.ckpt_dir = ckpt_dir
        _safe_mkdir(ckpt_dir)

    def __call__(self, solver):
        now = datetime.now()
        timestr = now.strftime("%Y-%m-%d_%H-%M-%S")
        fname = os.path.join(self.ckpt_dir, timestr + ".internals")
        with open(fname, 'wb') as f:
            dill.dump(solver.get_internals("all"), f)
            self.logger.info(f"Saved checkpoint to {fname} at local epoch = {solver.local_epoch} "
                             f"(global epoch = {solver.global_epoch})")


class ReportCallback(ActionCallback):
    r"""A callback that logs the training/validation information, including

    - number of batches (train/valid)
    - batch size (train/valid)
    - generator to be used (train/valid)

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __call__(self, solver):
        self.logger.info(
            f"Starting from global epoch {solver.global_epoch - 1}\n"
            f"    training with {solver.generator['train']}\n"
            f"    validating with {solver.generator['valid']}"
        )
        tb = solver.generator['train'].size
        ntb = solver.n_batches['train']
        t = tb * ntb
        vb = solver.generator['valid'].size
        nvb = solver.n_batches['valid']
        v = vb * nvb
        self.logger.info(f"train size = {tb} x {ntb} = {t}, valid_size = {vb} x {nvb} = {v}")


ReportOnFitCallback = warn_deprecate_class(ReportCallback)


class EveCallback(ActionCallback):
    r"""A callback that readjusts the number of batches for training based on latest value of a specified metric.
    The number of batches will be :math:`\displaystyle{\left(n_0 \cdot 2^k\right)}`
    or :math:`n_\mathrm{max}` (if specified), whichever is lower,
    where :math:`\displaystyle{k=\max\left(0,\left\lfloor\log_p{\frac{v}{v_0}}\right\rfloor\right)}`
    and :math:`v` is the value of the metric in the last epoch.

    :param base_value:
        Base value of the specified metric (:math:`v_0` in the above equation).
        When the metric value is higher than ``base_value``, number of batches will be :math:`n_0`.
    :type base_value: float
    :param double_at:
        The ratio at which the batch number will be doubled (:math:`p` in the above equation).
        When :math:`\displaystyle{\frac{v}{v_0}=p^k}`,
        the number of batches will be :math:`\displaystyle{\left(n_0 \cdot 2^k\right)}`.
    :type double_at: float
    :param n_0: Minimum number of batches (:math:`n_0`). Defaults to 1.
    :type n_0: int
    :param n_max: Maximum number of batches (:math:`n_\mathrm{max}`). Defaults to infinity.
    :type n_max: int
    :param use_train: Whether to use the training (instead of validation) phase value of the metric. Defaults to True.
    :type use_train: bool
    :param metric:
        Name of which metric to use. Must be 'loss' or present in ``solver.metrics_fn.keys()``. Defaults to 'loss'.
    :type metric: str
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """
    EPS = 1e-4

    def __init__(self, base_value=1.0, double_at=0.1, n_0=1, n_max=None, use_train=True, metric='loss', logger=None):
        super(EveCallback, self).__init__(logger=logger)
        self.base_value = base_value
        self.double_at = double_at
        self.n_0 = n_0
        self.n_max = n_max or np.inf
        key = 'train' if use_train else 'valid'
        self.key = f'{key}_{metric}'

    def __call__(self, solver):
        value = solver.metrics_history[self.key][-1]
        double_times = int(self.__class__.EPS + (np.log(value) - np.log(self.base_value)) / np.log(self.double_at))
        double_times = max(double_times, 0)
        solver.n_batches['train'] = min(self.n_0 * 2 ** double_times, self.n_max)


class SimpleTensorboardCallback(ActionCallback):
    r"""A callback that writes all metric values to the disk for TensorBoard to plot.
    Tensorboard must be installed for this callback to work.

    :param writer:
        The summary writer for writing values to disk.
        Defaults to a new ``SummaryWriter`` instance created with default kwargs.
    :type writer: ``torch.utils.tensorboard.SummaryWriter``
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, writer=None, logger=None):
        super(SimpleTensorboardCallback, self).__init__(logger=logger)
        if not writer:
            self.logger.info('No writer specified, creating a SummaryWriter automatically.')
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(f"TensorBoard doesn't seem to be installed. See the following\n{e}")

        self.writer = writer or SummaryWriter()

    def __call__(self, solver):
        for name, values in solver.metrics_history.items():
            self.writer.add_scalar(
                tag=name,
                scalar_value=values[-1] if values else np.nan,
                global_step=solver.global_epoch,
            )


class ConditionCallback(BaseCallback):
    r"""Base class of condition callbacks.
    Custom callbacks that *determines whether some action shall be performed* should subclass this class and overwrite
    the ``.condition`` method.

    Instances of ``ConditionCallback`` (and its children classes) support (short-circuit) evaluation of
    common boolean operations: ``&`` (and), ``|`` (or), ``~`` (not), and ``^`` (xor).

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, logger=None):
        super(ConditionCallback, self).__init__(logger=logger)
        self.action_callback = None

    def set_action_callback(self, action_callback):
        if not isinstance(action_callback, ActionCallback):
            raise TypeError(f'{action_callback} is not an instance of ActionCallback')
        self.action_callback = action_callback
        return self

    @abstractmethod
    def condition(self, solver) -> bool:
        pass  # pragma: no cover

    def __call__(self, solver):
        if self.condition(solver):
            if self.action_callback:
                self.logger.debug(f"condition of {self} met, running the underlying callback {self.action_callback}")
                self.action_callback(solver)
            else:
                self.logger.warning(f"condition of {self} met, but no underlying action callback is set; skipping")
        else:
            self.logger.debug(f"condition of {self} not met")

    def __and__(self, other):
        return AndCallback(condition_callbacks=[self, other], logger=self.logger)

    def __or__(self, other):
        return OrCallback(condition_callbacks=[self, other], logger=self.logger)

    def __invert__(self):
        return NotCallback(condition_callback=self, logger=self.logger)

    def __xor__(self, other):
        return XorCallback(condition_callbacks=[self, other], logger=self.logger)


class AndCallback(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True iff none of its sub-``ConditionCallback`` s evaluates to False.

    :param condition_callbacks: List of sub-``ConditionCallback`` s.
    :type condition_callbacks: list[``ConditionCallback``]
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``

    .. note::
        ``c = AndCallback([c1, c2, c3])`` can be simplified as ``c = c1 & c2 & c3``.
    """

    def __init__(self, condition_callbacks, logger=None):
        super(AndCallback, self).__init__(logger=logger)
        self.condition_callbacks = condition_callbacks

    def condition(self, solver) -> bool:
        for cond_cb in self.condition_callbacks:
            c = cond_cb.condition(solver)
            if not c:
                return False
        return True


class OrCallback(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to False iff none of its sub-``ConditionCallback`` s evaluates to True.

    :param condition_callbacks: List of sub-``ConditionCallback`` s.
    :type condition_callbacks: list[``ConditionCallback``]
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``

    .. note::
        ``c = OrCallback([c1, c2, c3])`` can be simplified as ``c = c1 | c2 | c3``.
    """

    def __init__(self, condition_callbacks, logger=None):
        super(OrCallback, self).__init__(logger=logger)
        self.condition_callbacks = condition_callbacks

    def condition(self, solver) -> bool:
        for cond_cb in self.condition_callbacks:
            if cond_cb.condition(solver):
                return True
        return False


class NotCallback(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True iff its sub-``ConditionCallback`` evaluates to False.

    :param condition_callback: The sub-``ConditionCallback`` .
    :type condition_callback: ConditionCallback
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``

    .. note::
        ``c = NotCallback(c1)`` can be simplified as ``c = ~c1``.
    """

    def __init__(self, condition_callback, logger=None):
        super(NotCallback, self).__init__(logger=logger)
        self.condition_callback = condition_callback

    def condition(self, solver) -> bool:
        return not self.condition_callback.condition(solver)


class XorCallback(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to False iff
    evenly many of its sub-``ConditionCallback`` s evaluates to True.

    :param condition_callbacks: List of sub-``ConditionCallback`` s.
    :type condition_callbacks: list[``ConditionCallback``]
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``

    .. note::
        ``c = XorCallback([c1, c2, c3])`` can be simplified as ``c = c1 ^ c2 ^ c3``.
    """

    def __init__(self, condition_callbacks, logger=None):
        super(XorCallback, self).__init__(logger=logger)
        self.condition_callbacks = condition_callbacks

    def condition(self, solver) -> bool:
        return sum(1 for cond_cb in self.condition_callbacks if cond_cb.condition(solver)) % 2 == 1


class TrueCallback(ConditionCallback):
    r"""A ``ConditionCallback`` which always evaluates to True.

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def condition(self, solver) -> bool:
        return True


class FalseCallback(ConditionCallback):
    r"""A ``ConditionCallback`` which always evaluates to False.

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def condition(self, solver) -> bool:
        return False


class OnFirstLocal(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True only on the first local epoch.

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def condition(self, solver) -> bool:
        return solver.local_epoch == 1


class OnFirstGlobal(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True only on the first global epoch.

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def condition(self, solver) -> bool:
        return solver.global_epoch == 1


class OnLastLocal(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True only on the last local epoch.

    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def condition(self, solver) -> bool:
        return solver.local_epoch == solver._max_local_epoch


class PeriodLocal(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True only when the local epoch count equals
    :math:`\mathrm{period}\times n + \mathrm{offset}`.

    :param period: Period of the callback.
    :type period: int
    :param offset: Offset of the period. Defaults to 0.
    :type offset: int
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, period, offset=0, logger=None):
        super(PeriodLocal, self).__init__(logger=logger)
        self.period = period
        self.offset = offset % period

    def condition(self, solver) -> bool:
        return solver.local_epoch % self.period == self.offset


class PeriodGlobal(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True only when the global epoch count equals
    :math:`\mathrm{period}\times n + \mathrm{offset}`.

    :param period: Period of the callback.
    :type period: int
    :param offset: Offset of the period. Defaults to 0.
    :type offset: int
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, period, offset=0, logger=None):
        super(PeriodGlobal, self).__init__(logger=logger)
        self.period = period
        self.offset = offset % period

    def condition(self, solver) -> bool:
        return solver.global_epoch % self.period == self.offset


class ClosedIntervalLocal(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True only when
    :math:`l_0 \leq l \leq l_1`, where :math:`l` is the local epoch count.

    :param min: Lower bound of the closed interval (:math:`l_0` in the above inequality). Defaults to None.
    :type min: int
    :param max: Upper bound of the closed interval (:math:`l_1` in the above inequality). Defaults to None.
    :type max: int
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, min=None, max=None, logger=None):
        super(ClosedIntervalLocal, self).__init__(logger=logger)
        self.min = - np.inf if min is None else min
        self.max = np.inf if max is None else max

    def condition(self, solver) -> bool:
        return self.min <= solver.local_epoch <= self.max


class ClosedIntervalGlobal(ConditionCallback):
    r"""A ``ConditionCallback`` which evaluates to True only when
    :math:`g_0 \leq g \leq g_1`, where :math:`g` is the global epoch count.

    :param min: Lower bound of the closed interval (:math:`g_0` in the above inequality). Defaults to None.
    :type min: int
    :param max: Upper bound of the closed interval (:math:`g_1` in the above inequality). Defaults to None.
    :type max: int
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, min=None, max=None, logger=None):
        super(ClosedIntervalGlobal, self).__init__(logger=logger)
        self.min = - np.inf if min is None else min
        self.max = np.inf if max is None else max

    def condition(self, solver) -> bool:
        return self.min <= solver.global_epoch <= self.max


class Random(ConditionCallback):
    r"""A ``ConditionCallback`` which has a certain probability of evaluating to True.

    :param probability: The probability of this callback evaluating to True (between 0 and 1).
    :type probability: float
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, probability, logger=None):
        super(Random, self).__init__(logger=logger)
        if probability < 0 or probability > 1:
            raise ValueError('probability must lie in [0, 1]')
        self.probability = probability

    def condition(self, solver) -> bool:
        return random.random() < self.probability


class _RepeatedMetricChange(ConditionCallback):
    def __init__(self, use_train=True, metric='loss', repetition=1, logger=None):
        super(_RepeatedMetricChange, self).__init__(logger=logger)
        key = 'train' if use_train else 'valid'
        self.key = f'{key}_{metric}'
        self.times_required = repetition
        self.so_far = 0

    @abstractmethod
    def _last_satisfied(self, last, second2last):
        return last > second2last

    def condition(self, solver) -> bool:
        history = solver.metrics_history[self.key]
        if len(history) >= 2 and self._last_satisfied(last=history[-1], second2last=history[-2]):
            self.so_far += 1
        else:
            self.so_far = 0
        return self.so_far >= self.times_required


class RepeatedMetricUp(_RepeatedMetricChange):
    r"""A ``ConditionCallback`` which evaluates to True if a certain metric for the latest :math:`n` epochs
    kept increasing by at least some margin.

    :param at_least_by: The said margin.
    :type at_least_by: float
    :param use_train: Whether to use the metric value in the training (rather than validation) phase.
    :type use_train: bool
    :param metric:
        Name of which metric to use. Must be 'loss' or present in ``solver.metrics_fn.keys()``. Defaults to 'loss'.
    :type metric: str
    :param repetition: Number of times the metric should increase by the said margin (the :math:`n`).
    :type repetition: int
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, at_least_by=0.0, use_train=True, metric='loss', repetition=1, logger=None):
        super(RepeatedMetricUp, self).__init__(
            use_train=use_train, metric=metric, repetition=repetition, logger=logger
        )
        self.at_least_by = at_least_by

    def _last_satisfied(self, last, second2last):
        return last >= second2last + self.at_least_by


class RepeatedMetricDown(_RepeatedMetricChange):
    r"""A ``ConditionCallback`` which evaluates to True if a certain metric for the latest :math:`n` epochs
    kept decreasing by at least some margin.

    :param at_least_by: The said margin.
    :type at_least_by: float
    :param use_train: Whether to use the metric value in the training (rather than validation) phase.
    :type use_train: bool
    :param metric:
        Name of which metric to use. Must be 'loss' or present in ``solver.metrics_fn.keys()``. Defaults to 'loss'.
    :type metric: str
    :param repetition: Number of times the metric should decrease by the said margin (the :math:`n`).
    :type repetition: int
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, at_least_by=0.0, use_train=True, metric='loss', repetition=1, logger=None):
        super(RepeatedMetricDown, self).__init__(
            use_train=use_train, metric=metric, repetition=repetition, logger=logger
        )
        self.at_least_by = at_least_by

    def _last_satisfied(self, last, second2last):
        return last <= second2last - self.at_least_by


class RepeatedMetricConverge(_RepeatedMetricChange):
    r"""A ``ConditionCallback`` which evaluates to True if a certain metric for the latest :math:`n` epochs
    kept converging within some tolerance :math:`\varepsilon`.

    :param epsilon: The said tolerance.
    :type epsilon: float
    :param use_train: Whether to use the metric value in the training (rather than validation) phase.
    :type use_train: bool
    :param metric:
        Name of which metric to use. Must be 'loss' or present in ``solver.metrics_fn.keys()``. Defaults to 'loss'.
    :type metric: str
    :param repetition: Number of times the metric should converge within said tolerance.
    :type repetition: int
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, epsilon, use_train=True, metric='loss', repetition=1, logger=None):
        super(RepeatedMetricConverge, self).__init__(
            use_train=use_train, metric=metric, repetition=repetition, logger=logger
        )
        self.epsilon = abs(epsilon)

    def _last_satisfied(self, last, second2last):
        return abs(last - second2last) < self.epsilon


class RepeatedMetricDiverge(_RepeatedMetricChange):
    r"""A ``ConditionCallback`` which evaluates to True if a certain metric for the latest :math:`n` epochs
    kept diverging beyond some gap.

    :param gap: The said gap.
    :type gap: float
    :param use_train: Whether to use the metric value in the training (rather than validation) phase.
    :type use_train: bool
    :param metric:
        Name of which metric to use. Must be 'loss' or present in ``solver.metrics_fn.keys()``. Defaults to 'loss'.
    :type metric: str
    :param repetition: Number of times the metric should diverge beyond said gap.
    :type repetition: int
    :param logger: The logger (or its name) to be used for this callback. Defaults to the 'root' logger.
    :type logger: str or ``logging.Logger``
    """

    def __init__(self, gap, use_train=True, metric='loss', repetition=1, logger=None):
        super(RepeatedMetricDiverge, self).__init__(
            use_train=use_train, metric=metric, repetition=repetition, logger=logger
        )
        self.gap = abs(gap)

    def _last_satisfied(self, last, second2last):
        return abs(last - second2last) > self.gap
