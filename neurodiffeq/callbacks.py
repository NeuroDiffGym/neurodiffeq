import os
import dill
import warnings
import random
import numpy as np
from datetime import datetime
import logging
from .utils import safe_mkdir as _safe_mkdir
from ._version_utils import deprecated_alias
from abc import ABC, abstractmethod


class _LoggerMixin:
    def __init__(self, logger=None):
        if not logger:
            self.logger = logging.getLogger('root')
        elif isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        else:
            self.logger = logger


class BaseCallback(ABC, _LoggerMixin):
    def __init__(self, logger=None):
        _LoggerMixin.__init__(self, logger=logger)

    @abstractmethod
    def __call__(self, solver):
        pass  # pragma: no cover


class ActionCallback(BaseCallback):
    def conditioned_on(self, condition_callback):
        if not isinstance(condition_callback, ConditionMetaCallback):
            raise TypeError(f'{condition_callback} is not an instance of ConditionMetaCallback')
        return condition_callback.set_action_callback(self)


class MonitorCallback(ActionCallback):
    """A callback for updating the monitor plots (and optionally saving the fig to disk).

    :param monitor: The underlying monitor responsible for plotting solutions.
    :type monitor: `neurodiffeq.monitors.BaseMonitor`
    :param fig_dir: Directory for saving monitor figs; if not specified, figs will not be saved.
    :type fig_dir: str
    :param format: Format for saving figures: {'jpg', 'png' (default), ...}.
    :type format: str
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
    def __call__(self, solver):
        solver._stop_training = True


class CheckpointCallback(ActionCallback):
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


class ReportOnFitCallback(ActionCallback):
    def __call__(self, solver):
        self.logger.info(
            f"Starting from global epoch {solver.global_epoch - 1}\n"
            f"training with {solver.generator['train']}\n"
            f"validating with {solver.generator['valid']}\n"
        )
        tb = solver.generator['train'].size
        ntb = solver.n_batches['train']
        t = tb * ntb
        vb = solver.generator['valid'].size
        nvb = solver.n_batches['valid']
        v = vb * nvb
        self.logger.info(f"train size = {tb} x {ntb} = {t}, valid_size = {vb} x {nvb} = {v}")


class ConditionMetaCallback(BaseCallback):
    def __init__(self, logger=None):
        super(ConditionMetaCallback, self).__init__(logger=logger)
        self.action_callback = None

    def set_action_callback(self, action_callback):
        if not isinstance(action_callback, ActionCallback):
            raise TypeError(f'{action_callback} if not an instance of ActionCallback')
        self.action_callback = action_callback
        return self

    @abstractmethod
    def condition(self, solver) -> bool:
        pass  # pragma: no cover

    def __call__(self, solver):
        if self.condition(solver):
            if self.action_callback:
                self.logger.info(f"condition of {self} met, running the underlying callback {self.action_callback}")
                self.action_callback(solver)
            else:
                self.logger.warning(f"condition of {self} met, but no underlying action callback is set; skipping")
        else:
            self.logger.info(f"condition of {self} not met")

    def __and__(self, other):
        return AndCallback(condition_callbacks=[self, other], logger=self.logger)

    def __or__(self, other):
        return OrCallback(condition_callbacks=[self, other], logger=self.logger)

    def __invert__(self):
        return NotCallback(condition_callback=self, logger=self.logger)

    def __xor__(self, other):
        return XorCallback(condition_callbacks=[self, other], logger=self.logger)


class AndCallback(ConditionMetaCallback):
    def __init__(self, condition_callbacks, logger=None):
        super(AndCallback, self).__init__(logger=logger)
        self.condition_callbacks = condition_callbacks

    def condition(self, solver) -> bool:
        for cond_cb in self.condition_callbacks:
            c = cond_cb.condition(solver)
            if not c:
                return False
        return True


class OrCallback(ConditionMetaCallback):
    def __init__(self, condition_callbacks, logger=None):
        super(OrCallback, self).__init__(logger=logger)
        self.condition_callbacks = condition_callbacks

    def condition(self, solver) -> bool:
        for cond_cb in self.condition_callbacks:
            if cond_cb.condition(solver):
                return True
        return False


class NotCallback(ConditionMetaCallback):
    def __init__(self, condition_callback, logger=None):
        super(NotCallback, self).__init__(logger=logger)
        self.condition_callback = condition_callback

    def condition(self, solver) -> bool:
        return not self.condition_callback.condition(solver)


class XorCallback(ConditionMetaCallback):
    def __init__(self, condition_callbacks, logger=None):
        super(XorCallback, self).__init__(logger=logger)
        self.condition_callbacks = condition_callbacks

    def condition(self, solver) -> bool:
        return sum(1 for cond_cb in self.condition_callbacks if cond_cb.condition(solver)) % 2 == 1


class TrueCallback(ConditionMetaCallback):
    def condition(self, solver) -> bool:
        return True


class FalseCallback(ConditionMetaCallback):
    def condition(self, solver) -> bool:
        return False


class OnFirstLocal(ConditionMetaCallback):
    def condition(self, solver) -> bool:
        return solver.local_epoch == 1


class OnFirstGlobal(ConditionMetaCallback):
    def condition(self, solver) -> bool:
        return solver.global_epoch == 1


class OnLastLocal(ConditionMetaCallback):
    def condition(self, solver) -> bool:
        return solver.local_epoch == solver._max_local_epoch


class PeriodLocal(ConditionMetaCallback):
    def __init__(self, period, logger=None):
        super(PeriodLocal, self).__init__(logger=logger)
        self.period = period

    def condition(self, solver) -> bool:
        return solver.local_epoch % self.period == 0


class PeriodGlobal(ConditionMetaCallback):
    def __init__(self, period, logger=None):
        super(PeriodGlobal, self).__init__(logger=logger)
        self.period = period

    def condition(self, solver) -> bool:
        return solver.global_epoch % self.period == 0


class ClosedIntervalLocal(ConditionMetaCallback):
    def __init__(self, min=None, max=None, logger=None):
        super(ClosedIntervalLocal, self).__init__(logger=logger)
        self.min = - np.inf if min is None else min
        self.max = np.inf if max is None else max

    def condition(self, solver) -> bool:
        return self.min <= solver.local_epoch <= self.max


class ClosedIntervalGlobal(ConditionMetaCallback):
    def __init__(self, min=None, max=None, logger=None):
        super(ClosedIntervalGlobal, self).__init__(logger=logger)
        self.min = - np.inf if min is None else min
        self.max = np.inf if max is None else max

    def condition(self, solver) -> bool:
        return self.min <= solver.global_epoch <= self.max


class Random(ConditionMetaCallback):
    def __init__(self, probability, logger=None):
        super(Random, self).__init__(logger=logger)
        if probability < 0 or probability > 1:
            raise ValueError('probability must lie in [0, 1]')
        self.probability = probability

    def condition(self, solver) -> bool:
        return random.random() < self.probability


class _RepeatedMetricChange(ConditionMetaCallback):
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
    def __init__(self, at_least_by=0.0, use_train=True, metric='loss', repetition=1, logger=None):
        super(RepeatedMetricUp, self).__init__(
            use_train=use_train, metric=metric, repetition=repetition, logger=logger
        )
        self.at_least_by = at_least_by

    def _last_satisfied(self, last, second2last):
        return last >= second2last + self.at_least_by


class RepeatedMetricDown(_RepeatedMetricChange):
    def __init__(self, at_least_by=0.0, use_train=True, metric='loss', repetition=1, logger=None):
        super(RepeatedMetricDown, self).__init__(
            use_train=use_train, metric=metric, repetition=repetition, logger=logger
        )
        self.at_least_by = at_least_by

    def _last_satisfied(self, last, second2last):
        return last <= second2last - self.at_least_by


class RepeatedMetricConverge(_RepeatedMetricChange):
    def __init__(self, epsilon, use_train=True, metric='loss', repetition=1, logger=None):
        super(RepeatedMetricConverge, self).__init__(
            use_train=use_train, metric=metric, repetition=repetition, logger=logger
        )
        self.epsilon = abs(epsilon)

    def _last_satisfied(self, last, second2last):
        return abs(last - second2last) < self.epsilon


class RepeatedMetricDiverge(_RepeatedMetricChange):
    def __init__(self, gap, use_train=True, metric='loss', repetition=1, logger=None):
        super(RepeatedMetricDiverge, self).__init__(
            use_train=use_train, metric=metric, repetition=repetition, logger=logger
        )
        self.gap = abs(gap)

    def _last_satisfied(self, last, second2last):
        return abs(last - second2last) > self.gap
