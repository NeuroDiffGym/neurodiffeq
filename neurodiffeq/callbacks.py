import os
import dill
import warnings
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


class MonitorCallback(BaseCallback):
    """A callback for updating the monitor plots (and optionally saving the fig to disk).

    :param monitor: The underlying monitor responsible for plotting solutions.
    :type monitor: `neurodiffeq.monitors.BaseMonitor`
    :param fig_dir: Directory for saving monitor figs; if not specified, figs will not be saved.
    :type fig_dir: str
    :param check_against: Which epoch count to check against; either 'local' (default) or 'global'.
    :type check_against: str
    :param repaint_last: Whether to update the plot on the last local epoch, defaults to True.
    :type repaint_last: bool
    """

    @deprecated_alias(check_against='check_against_local')
    def __init__(self, monitor, fig_dir=None, check_against_local=True, repaint_last=True, logger=None):
        super(MonitorCallback, self).__init__(logger=logger)
        self.monitor = monitor
        self.fig_dir = fig_dir
        if fig_dir:
            _safe_mkdir(fig_dir)
        self.repaint_last = repaint_last

        if isinstance(check_against_local, bool):
            self.check_against_local = check_against_local
        elif check_against_local in ['local', 'global']:
            warnings.warn(
                '`check_against` is deprecated use `check_against_local={True, False}` in stead',
                FutureWarning
            )
            self.check_against_local = (check_against_local == 'local')
        else:
            raise TypeError(f"pass `check_against_local={{True, False}}` instead of {check_against_local}")

    def to_repaint(self, solver):
        if self.check_against_local:
            epoch_now = solver.local_epoch + 1
        else:
            epoch_now = solver.global_epoch + 1

        if epoch_now % self.monitor.check_every == 0:
            return True
        if self.repaint_last and solver.local_epoch == solver._max_local_epoch - 1:
            return True

        return False

    def __call__(self, solver):
        if not self.to_repaint(solver):
            return

        self.monitor.check(
            solver.nets,
            solver.conditions,
            history=solver.metrics_history,
        )
        if self.fig_dir:
            pic_path = os.path.join(self.fig_dir, f"epoch-{solver.global_epoch}.png")
            self.monitor.fig.savefig(pic_path)
            self.logger.info(f'plot saved to {pic_path}')


class CheckpointCallback(BaseCallback):
    def __init__(self, ckpt_dir, logger=None):
        super(CheckpointCallback, self).__init__(logger=logger)
        self.ckpt_dir = ckpt_dir
        _safe_mkdir(ckpt_dir)

    def __call__(self, solver):
        if solver.local_epoch == solver._max_local_epoch - 1:
            now = datetime.now()
            timestr = now.strftime("%Y-%m-%d_%H-%M-%S")
            fname = os.path.join(self.ckpt_dir, timestr + ".internals")
            with open(fname, 'wb') as f:
                dill.dump(solver.get_internals("all"), f)
                self.logger.info(f"Saved checkpoint to {fname} at local epoch = {solver.local_epoch} "
                                 f"(global epoch = {solver.global_epoch})")


class ReportOnFitCallback(BaseCallback):
    def __call__(self, solver):
        if solver.local_epoch == 0:
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

    def set_action_callback(self, callback):
        self.action_callback = callback

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
