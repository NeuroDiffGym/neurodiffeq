import os
import dill
from datetime import datetime
import logging


class MonitorCallback:
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

    def __init__(self, monitor, fig_dir=None, check_against='local', repaint_last=True):
        self.monitor = monitor
        self.fig_dir = fig_dir
        self.repaint_last = repaint_last
        if check_against not in ['local', 'global']:
            raise ValueError(f'unknown check_against type = {check_against}')
        self.check_against = check_against

    def to_repaint(self, solver):
        if self.check_against == 'local':
            epoch_now = solver.local_epoch + 1
        elif self.check_against == 'global':
            epoch_now = solver.global_epoch + 1
        else:
            raise ValueError(f'unknown check_against type = {self.check_against}')

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


class CheckpointCallback:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def __call__(self, solver):
        if solver.local_epoch == solver._max_local_epoch - 1:
            now = datetime.now()
            timestr = now.strftime("%Y-%m-%d_%H-%M-%S")
            fname = os.path.join(self.ckpt_dir, timestr + ".internals")
            with open(fname, 'wb') as f:
                dill.dump(solver.get_internals("all"), f)
                logging.info(f"Saved checkpoint to {fname} at local epoch = {solver.local_epoch} "
                             f"(global epoch = {solver.global_epoch})")


class ReportOnFitCallback:
    def __call__(self, solver):
        if solver.local_epoch == 0:
            logging.info(
                f"Starting from global epoch {solver.global_epoch - 1}, training on {(solver.r_min, solver.r_max)}")
            tb = solver.generator['train'].size
            ntb = solver.n_batches['train']
            t = tb * ntb
            vb = solver.generator['valid'].size
            nvb = solver.n_batches['valid']
            v = vb * nvb
            logging.info(f"train size = {tb} x {ntb} = {t}, valid_size = {vb} x {nvb} = {v}")
