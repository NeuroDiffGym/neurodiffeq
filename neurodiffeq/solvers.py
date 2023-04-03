import sys
import warnings
import inspect
from inspect import signature
from abc import ABC, abstractmethod
from itertools import chain
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from .solvers_utils import PretrainedSolver
from .networks import FCNN
from ._version_utils import deprecated_alias
from .generators import GeneratorSpherical
from .generators import SamplerGenerator
from .generators import Generator1D
from .generators import Generator2D
from .generators import GeneratorND
from .function_basis import RealSphericalHarmonics
from .conditions import BaseCondition
from .neurodiffeq import safe_diff as diff
from .losses import _losses


def _requires_closure(optimizer):
    # starting from torch v1.13, simple optimizers no longer have a `closure` argument
    closure_param = inspect.signature(optimizer.step).parameters.get('closure')
    return closure_param and closure_param.default == inspect._empty


class BaseSolver(ABC, PretrainedSolver):
    r"""A class for solving ODE/PDE systems.

    :param diff_eqs:
        The differential equation system to solve, which maps a tuple of coordinates to a tuple of ODE/PDE residuals.
        Both the coordinates and ODE/PDE residuals must have shape (-1, 1).
    :type diff_eqs: callable
    :param conditions:
        List of boundary conditions for each target function.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param nets:
        List of neural networks for parameterized solution.
        If provided, length must equal that of conditions.
    :type nets: list[`torch.nn.Module`], optional
    :param train_generator:
        A generator for sampling training points.
        It must provide a `.get_examples()` method and a `.size` field.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, required
    :param valid_generator:
        A generator for sampling validation points.
        It must provide a `.get_examples()` method and a `.size` field.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, required
    :param analytic_solutions:
        **[DEPRECATED]** Pass ``metrics`` instead.
        The analytical solutions to be compared with neural net solutions.
        It maps a tuple of coordinates to a tuple of function values.
        The output shape should match that of networks.
    :type analytic_solutions: callable, optional
    :param optimizer:
        The optimizer to be used for training.
    :type optimizer: `torch.nn.optim.Optimizer`, optional
    :param loss_fn:
        The loss function used for training.

        - If a str, must be present in the keys of `neurodiffeq.losses._losses`.
        - If a `torch.nn.modules.loss._Loss` instance, just pass the instance.
        - If any other callable, it must map
          A) a residual tensor (shape `(n_points, n_equations)`),
          B) a function values tuple (length `n_funcs`, each element a tensor of shape `(n_points, 1)`), and
          C) a coordinate values tuple (length `n_coords`, each element a tensor of shape `(n_coords, 1)`
          to a tensor of empty shape (i.e. a scalar). The returned tensor must be connected to the computational graph,
          so that backpropagation can be performed.

    :type loss_fn:
        str or `torch.nn.moduesl.loss._Loss` or callable
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param metrics:
        Additional metrics to be logged (besides loss). ``metrics`` should be a dict where

        - Keys are metric names (e.g. 'analytic_mse');
        - Values are functions (callables) that computes the metric value.
          These functions must accept the same input as the differential equation ``diff_eq``.

    :type metrics: dict, optional
    :param n_input_units:
        Number of input units for each neural network. Ignored if ``nets`` is specified.
    :type n_input_units: int, required
    :param n_output_units:
        Number of output units for each neural network. Ignored if ``nets`` is specified.
    :type n_output_units: int, required
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    """

    @deprecated_alias(criterion='loss_fn')
    def __init__(self, diff_eqs, conditions,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None,
                 optimizer=None, loss_fn=None, n_batches_train=1, n_batches_valid=4,
                 metrics=None, n_input_units=None, n_output_units=None,
                 # deprecated arguments are listed below
                 shuffle=None, batch_size=None):
        # deprecate argument `shuffle`
        if shuffle:
            warnings.warn(
                "param `shuffle` is deprecated and ignored; shuffling should be performed by generators",
                FutureWarning,
            )
        # deprecate argument `batch_size`
        if batch_size is not None:
            warnings.warn(
                "param `batch_size` is deprecated and ignored; specify n_batches_train and n_batches_valid instead",
                FutureWarning,
            )

        self.diff_eqs = diff_eqs
        self.conditions = conditions
        self.n_funcs = len(conditions)
        if nets is None:
            self.nets = [
                FCNN(n_input_units=n_input_units, n_output_units=n_output_units, hidden_units=(32, 32), actv=nn.Tanh)
                for _ in range(self.n_funcs)
            ]
        else:
            self.nets = nets

        if train_generator is None:
            raise ValueError("train_generator must be specified")

        if valid_generator is None:
            raise ValueError("valid_generator must be specified")

        self.metrics_fn = metrics if metrics else {}
        # For backward compatibility with the legacy `analytic_solutions` argument
        if analytic_solutions:
            warnings.warn(
                'The `analytic_solutions` argument is deprecated and could lead to unstable behavior. '
                'Pass a `metrics` dict instead.',
                FutureWarning,
            )

            def analytic_mse(*args):
                x = args[-n_input_units:]
                u_hat = analytic_solutions(*x)
                u = args[:-n_input_units]
                u, u_hat = torch.stack(u), torch.stack(u_hat)
                return ((u - u_hat) ** 2).mean()

            if 'analytic_mse' in self.metrics_fn:
                warnings.warn(
                    "Ignoring `analytic_solutions` in presence of key 'analytic_mse' in `metrics`",
                    FutureWarning,
                )
            else:
                self.metrics_fn['analytic_mse'] = analytic_mse

        # metric history, keys will be train_loss, valid_loss, train__<metric_name>, valid__<metric_name>.
        # For compatibility with ode.py and pde.py,
        # double underscore are used between 'train'/'valid' and custom metric names.
        self.metrics_history = {}
        self.metrics_history.update({'train_loss': [], 'valid_loss': []})
        self.metrics_history.update({'train__' + name: [] for name in self.metrics_fn})
        self.metrics_history.update({'valid__' + name: [] for name in self.metrics_fn})

        self.optimizer = optimizer if optimizer else Adam(set(chain.from_iterable(n.parameters() for n in self.nets)))
        self._set_loss_fn(loss_fn)

        def make_pair_dict(train=None, valid=None):
            return {'train': train, 'valid': valid}

        self.generator = make_pair_dict(
            train=SamplerGenerator(train_generator),
            valid=SamplerGenerator(valid_generator),
        )
        # number of batches for training / validation;
        self.n_batches = make_pair_dict(train=n_batches_train, valid=n_batches_valid)
        # current batch of samples, kept for additional_loss term to use
        self._batch = make_pair_dict()
        if self.n_batches['valid'] == 0 and _requires_closure(self.optimizer):
            warnings.warn(f"Setting n_batches_valid=0 will update lowest_loss and best_net with training loss "
                          f"instead of validation loss. "
                          f"This is a problem for {self.optimizer.__class__} optimizer "
                          f"because it updates the parameters before the training loss computed. "
                          f"This leads to potentially worse solution in `best_net`!", RuntimeWarning)
        # current network with lowest loss
        self.best_nets = None
        # current lowest loss
        self.lowest_loss = None
        # local epoch in a `.fit` call, should only be modified inside self.fit()
        self.local_epoch = 0
        # maximum local epochs to run in a `.fit()` call, should only set by inside self.fit()
        self._max_local_epoch = 0
        # controls early stopping, should be set to False at the beginning of a `.fit()` call
        # and optionally set to False by `callbacks` in `.fit()` to support early stopping
        self._stop_training = False
        # the _phase variable is registered for callback functions to access
        self._phase = None

    def _set_loss_fn(self, criterion):
        if criterion is None:
            self.loss_fn = lambda r, f, x: (r ** 2).mean()
        elif isinstance(criterion, nn.modules.loss._Loss):
            self.loss_fn = lambda r, f, x: criterion(r, torch.zeros_like(r))
        elif isinstance(criterion, str):
            self.loss_fn = _losses[criterion.lower()]
        elif callable(criterion):
            self.loss_fn = criterion
        else:
            raise TypeError(f"Unknown type of criterion {type(criterion)}")

    @property
    def global_epoch(self):
        r"""Global epoch count, always equal to the length of train loss history.

        :return: Number of training epochs that have been run.
        :rtype: int
        """
        return len(self.metrics_history['train_loss'])

    @property
    def batch(self):
        return self._batch

    @property
    def _batch_examples(self):
        warnings.warn(
            '`._batch_examples` has been deprecated in favor of `._batch` and will be removed in a future version',
            FutureWarning,
        )
        return self._batch

    @property
    def criterion(self):
        warnings.warn(
            f'`{self.__class__.__name__}`.criterion is a deprecated alias for `{self.__class__.__name__}.loss_fn`.'
            f'The alias is only meant to be accessed by certain functions in `neurodiffeq.solver_utils` '
            f'until proper fixes are made; by which time this alias will be removed.'
        )
        return self.loss_fn

    @criterion.setter
    def criterion(self, loss_fn):
        warnings.warn(
            f'`{self.__class__.__name__}`.criterion is a deprecated alias for `{self.__class__.__name__}.loss_fn`.'
            f'The alias is only meant to be accessed by certain functions in `neurodiffeq.solver_utils` '
            f'until proper fixes are made; by which time this alias will be removed.'
        )
        self.loss_fn = loss_fn

    def compute_func_val(self, net, cond, *coordinates):
        r"""Compute the function value evaluated on the points specified by ``coordinates``.

        :param net: The network to be parameterized and evaluated.
        :type net: torch.nn.Module
        :param cond: The condition (a.k.a. parameterization) for the network.
        :type cond: `neurodiffeq.conditions.BaseCondition`
        :param coordinates: A tuple of coordinate components, each with shape = (-1, 1).
        :type coordinates: tuple[torch.Tensor]
        :return: Function values at the sampled points.
        :rtype: torch.Tensor
        """
        return cond.enforce(net, *coordinates)

    def _update_history(self, value, metric_type, key):
        r"""Append a value to corresponding history list.

        :param value: Value to be appended.
        :type value: float
        :param metric_type: Name of the metric. Must be 'loss' or present in ``self.metrics``.
        :type metric_type: str
        :param key: {'train', 'valid'}. Phase of the process.
        :type key: str
        """
        self._phase = key
        if metric_type == 'loss':
            self.metrics_history[f'{key}_{metric_type}'].append(value)
        elif metric_type in self.metrics_fn:
            self.metrics_history[f'{key}__{metric_type}'].append(value)
        else:
            raise KeyError(f"metric '{metric_type}' not specified")

    def _update_train_history(self, value, metric_type):
        r"""Append a value to corresponding training history list."""
        self._update_history(value, metric_type, key='train')

    def _update_valid_history(self, value, metric_type):
        r"""Append a value to corresponding validation history list."""
        self._update_history(value, metric_type, key='valid')

    def _generate_batch(self, key):
        r"""Generate the next batch, register in self._batch and return the batch.

        :param key:
            {'train', 'valid'};
            Dict key in ``self._examples``, ``self._batch``, or ``self._batch_start``
        :type key: str
        :return: The generated batch of points.
        :type: List[`torch.Tensor`]
        """
        # the following side effects are helpful for future extension,
        # especially for additional loss term that depends on the coordinates
        self._phase = key
        self._batch[key] = [v.reshape(-1, 1) for v in self.generator[key].get_examples()]
        return self._batch[key]

    def _generate_train_batch(self):
        r"""Generate the next training batch, register in ``self._batch`` and return."""
        return self._generate_batch('train')

    def _generate_valid_batch(self):
        r"""Generate the next validation batch, register in ``self._batch`` and return."""
        return self._generate_batch('valid')

    def _do_optimizer_step(self, closure=None):
        r"""Optimization procedures after gradients have been computed. Usually ``self.optimizer.step()`` is sufficient.
        At times, users can overwrite this method to perform gradient clipping, etc. Here is an example::

            import itertools
            class MySolver(Solver)
                def _do_optimizer_step(self, closure=None):
                    nn.utils.clip_grad_norm_(itertools.chain([net.parameters() for net in self.nets]), 1.0, 'inf')
                    self.optimizer.step(closure=closure)
        """
        self.optimizer.step(closure=closure)

    def _run_epoch(self, key):
        r"""Run an epoch on train/valid points, update history, and perform an optimization step if key=='train'.

        :param key: {'train', 'valid'}; phase of the epoch
        :type key: str

        .. note::
            The optimization step is only performed after all batches are run.
        """
        if self.n_batches[key] <= 0:
            # XXX maybe we should append NaN to metric history?
            return
        self._phase = key
        epoch_loss = 0.0
        batch_loss = 0.0
        metric_values = {name: 0.0 for name in self.metrics_fn}

        # Zero the gradient only once, before running the batches. Gradients of different batches are accumulated.
        if key == 'train' and not _requires_closure(self.optimizer):
            self.optimizer.zero_grad()

        # perform forward pass for all batches: a single graph is created and release in every iteration
        # see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/17
        for batch_id in range(self.n_batches[key]):
            batch = self._generate_batch(key)

            def closure(zero_grad=True):
                nonlocal batch_loss
                if key == 'train' and zero_grad:
                    self.optimizer.zero_grad()
                funcs = [
                    self.compute_func_val(n, c, *batch) for n, c in zip(self.nets, self.conditions)
                ]

                for name in self.metrics_fn:
                    value = self.metrics_fn[name](*funcs, *batch).item()
                    metric_values[name] += value
                residuals = self.diff_eqs(*funcs, *batch)
                residuals = torch.cat(residuals, dim=1)
                try:
                    loss = self.loss_fn(residuals, funcs, batch) + self.additional_loss(residuals, funcs, batch)
                except TypeError as e:
                    warnings.warn(
                        "You might need to update your code. "
                        "Since v0.4.0; both `criterion` and `additional_loss` requires three inputs: "
                        "`residual`, `funcs`, and `coords`. See documentation for more.", FutureWarning)
                    raise e

                # accumulate gradients before the current graph is collected as garbage
                if key == 'train':
                    loss.backward()
                    batch_loss = loss.item()
                return loss

            if key == 'train':
                if _requires_closure(self.optimizer):
                    # If `closure` is required by `optimizer.step()`, perform a step for every batch
                    self._do_optimizer_step(closure=closure)
                else:
                    # Otherwise, only perform backward propagation.
                    # Optimizer step will be performed only once outside the for-loop (i.e. after all batches).
                    closure(zero_grad=False)
                epoch_loss += batch_loss
            else:
                epoch_loss += closure().item()

        # calculate mean loss of all batches and register to history
        self._update_history(epoch_loss / self.n_batches[key], 'loss', key)

        # If validation is performed, update the best network with the validation loss
        # Otherwise, try to update the best network with the training loss
        if key == 'valid' or self.n_batches['valid'] == 0:
            self._update_best(key)

        # perform the optimizer step after all batches are run (if optimizer.step doesn't require `closure`)
        if key == 'train' and not _requires_closure(self.optimizer):
            self._do_optimizer_step()

        # calculate average metrics across batches and register to history
        for name in self.metrics_fn:
            self._update_history(
                metric_values[name] / self.n_batches[key], name, key)

    def run_train_epoch(self):
        r"""Run a training epoch, update history, and perform gradient descent."""
        self._run_epoch('train')

    def run_valid_epoch(self):
        r"""Run a validation epoch and update history."""
        self._run_epoch('valid')

    def _update_best(self, key):
        r"""Update ``self.lowest_loss`` and ``self.best_nets``
        if current training/validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.metrics_history[key + '_loss'][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.best_nets = deepcopy(self.nets)

    def fit(self, max_epochs, callbacks=(), tqdm_file=sys.stderr, **kwargs):
        r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.

        If ``callbacks`` is passed, callbacks are run, one at a time,
        after training, validating and updating best model.

        :param max_epochs: Number of epochs to run.
        :type max_epochs: int
        :param callbacks:
            A list of callback functions.
            Each function should accept the ``solver`` instance itself as its **only** argument.
        :rtype callbacks: list[callable]
        :param tqdm_file:
            File to write tqdm progress bar. If set to None, tqdm is not used at all.
            Defaults to ``sys.stderr``.
        :type tqdm_file: io.StringIO or _io.TextIOWrapper

        .. note::
            1. This method does not return solution, which is done in the ``.get_solution()`` method.
            2. A callback ``cb(solver)`` can set ``solver._stop_training`` to True to perform early stopping.
        """
        self._stop_training = False
        self._max_local_epoch = max_epochs

        monitor = kwargs.pop('monitor', None)
        if monitor:
            warnings.warn("Passing `monitor` is deprecated, "
                          "use a MonitorCallback and pass a list of callbacks instead")
            callbacks = [monitor.to_callback()] + list(callbacks)
        if kwargs:
            raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')  # pragma: no cover

        if tqdm_file is None:
            loop = range(max_epochs)
        else:
            loop = tqdm(
                range(max_epochs),
                desc='Training Progress',
                colour='blue',
                file=tqdm_file,
                dynamic_ncols=True,
            )

        for local_epoch in loop:
            # stop training if self._stop_training is set to True by a callback
            if self._stop_training:
                break

            # register local epoch (starting from 1 instead of 0) so it can be accessed by callbacks
            self.local_epoch = local_epoch + 1
            self.run_train_epoch()
            self.run_valid_epoch()

            for cb in callbacks:
                cb(self)

    @abstractmethod
    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        pass  # pragma: no cover

    def _get_internal_variables(self):
        r"""Get a dict of all available internal variables.

        :return:
            All available internal variables,
            where keys are variable names and values are the corresponding variables.
        :rtype: dict

        .. note::
            Children classes should inherit all items and optionally include new ones.
        """

        return {
            "metrics": self.metrics_fn,
            "n_batches": self.n_batches,
            "best_nets": self.best_nets,
            "criterion": self.loss_fn,
            "loss_fn": self.loss_fn,
            "conditions": self.conditions,
            "global_epoch": self.global_epoch,
            "lowest_loss": self.lowest_loss,
            "n_funcs": self.n_funcs,
            "nets": self.nets,
            "optimizer": self.optimizer,
            "diff_eqs": self.diff_eqs,
            "generator": self.generator,
            "train_generator": self.generator['train'],
            "valid_generator": self.generator['valid'],
        }

    @deprecated_alias(param_names='var_names')
    def get_internals(self, var_names=None, return_type='list'):
        r"""Return internal variable(s) of the solver

        - If var_names == 'all', return all internal variables as a dict.
        - If var_names is single str, return the corresponding variables.
        - If var_names is a list and return_type == 'list', return corresponding internal variables as a list.
        - If var_names is a list and return_type == 'dict', return a dict with keys in var_names.

        :param var_names: An internal variable name or a list of internal variable names.
        :type var_names: str or list[str]
        :param return_type: {'list', 'dict'}; Ignored if ``var_names`` is a string.
        :type return_type: str
        :return: A single variable, or a list/dict of internal variables as indicated above.
        :rtype: list or dict or any
        """

        available_variables = self._get_internal_variables()

        if var_names == "all" or var_names is None:
            return available_variables

        if isinstance(var_names, str):
            return available_variables[var_names]

        if return_type == 'list':
            return [available_variables[name] for name in var_names]
        elif return_type == "dict":
            return {name: available_variables[name] for name in var_names}
        else:
            raise ValueError(f"unrecognized return_type = {return_type}")

    def additional_loss(self, residual, funcs, coords):
        r"""Additional loss terms for training. This method is to be overridden by subclasses.
        This method can use any of the internal variables: self.nets, self.conditions, self.global_epoch, etc.

        :param residual: Residual tensor of differential equation. It has shape (N_SAMPLES, N_EQUATIONS)
        :type residual: torch.Tensor
        :param funcs:
            Outputs of the networks after parameterization.
            There are ``len(nets)`` entries in total. Each entry is a tensor of shape (N_SAMPLES, N_OUTPUT_UNITS).
        :type funcs: List[torch.Tensor]
        :param coords:
            Inputs to the networks; a.k.a. the spatio-temporal coordinates of the system.
            There are ``N_COORDS`` entries in total. Each entry is a tensor of shape (N_SAMPLES, 1).
        :type coords: List[torch.Tensor]
        :return: Additional loss. Must be a ``torch.Tensor`` of empty shape (scalar).
        :rtype: torch.Tensor
        """
        return 0.0

    def get_residuals(self, *coords, to_numpy=False, best=True, no_reshape=False):
        r"""Get the residuals of the differential equation (or system of differential equations)
        evaluated at given points.

        :param coords:
            The coordinate values where the residual(s) shall be evaluated.
            If numpy arrays are passed, the method implicitly creates torch tensors with corresponding values.
        :type coords: tuple[torch.Tensor] or tuple[np.ndarray]
        :param to_numpy: Whether to return numpy arrays. Defaults to False.
        :type to_numpy: bool
        :param best:
            If set to False, the network from the most recent epoch will be used to evaluate the residuals.
            If set to True, the network from the epoch with the lowest validation loss
            (or training loss, if there's no validation) will be used to evaluate the residuals.
            Defaults to True.
        :type best: bool
        :param no_reshape: If set to True, no reshaping will be performed on output. Defaults to False.
        :type no_reshape: bool
        :return:
            The residuals evaluated at given points.
            If there is only one equation in the differential equation system,
            a single torch tensor (or numpy array) will be returned.
            If there are multiple equations, a list of torch tensors (or numpy arrays) will be returned.
            The returned shape will be the same as the first input coordinate, unless `no_reshape` is set to True.
            Note that the return value will always be torch tensors (even if ``coords`` are numpy arrays)
            unless `to_numpy` is explicitly set to True.
        :rtype: list[`torch.Tensor` or `numpy.array`] or `torch.Tensor` or `numpy.array`
        """
        coords = [c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in coords]
        original_shape = coords[0].shape
        coords = [c.reshape(-1, 1).requires_grad_() for c in coords]
        solution = self.get_solution(copy=False, best=best)
        funcs = solution(*coords, to_numpy=False, no_reshape=no_reshape)
        if isinstance(funcs, torch.Tensor):
            funcs = [funcs]
        residuals = self.diff_eqs(*funcs, *coords)
        if not no_reshape:
            residuals = [r.reshape(*original_shape) for r in residuals]
        if to_numpy:
            residuals = [r.detach().cpu().numpy() for r in residuals]
        return residuals if len(residuals) > 1 else residuals[0]


class BaseSolution(ABC):
    r"""A solution to a PDE/ODE (system).

    :param nets:
        The neural networks that approximate the PDE/ODE solution.

        - If ``nets`` is a list of ``torch.nn.Module``, it should have the same length with ``conditions``
        - If ``nets`` is a single ``torch.nn.Module``, it should have as many output units as length of ``conditions``

    :type nets: list[`torch.nn.Module`] or `torch.nn.Module`
    :param conditions:
        A list of conditions that should be enforced on the PDE/ODE solution.
        ``conditions`` should have a length equal to the number of dependent variables in the ODE/PDE system.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    """

    def __init__(self, nets, conditions):
        if nets is None:
            raise RuntimeError("The nets cannot be None, check if you disabled validation "
                               "and used `best`=True with `get_solution` / `get_residual`")
        if isinstance(nets, nn.Module):
            # This is for backward compatibility with the `single_net` option
            # The same torch.nn.Module instance is repeated to form a list of the same length as `conditions`
            self.nets = [nets] * len(conditions)
        else:
            self.nets = nets
        self.conditions = conditions

    @abstractmethod
    def _compute_u(self, net, condition, *coords):
        pass  # pragma: no cover

    @deprecated_alias(as_type='to_numpy')
    def __call__(self, *coords, to_numpy=False, no_reshape=False):
        r"""Evaluate the solution at certain points.

        :param coords: tuple of coordinate tensors, each of shape (n_samples, 1)
        :type coords: Tuple[`torch.Tensor`]
        :param to_numpy:
            If set to True, the call returns a ``numpy.ndarray`` instead of ``torch.Tensor``.
            Defaults to False.
        :type to_numpy: bool
        :param no_reshape: If set to True, no reshaping will be performed on output. Defaults to False.
        :type no_reshape: bool
        :return:
            Dependent variables evaluated at given points.
            The shape of output will be that of the first input coordinate, unless `no_reshape` is set to True.
        :rtype: list[`torch.Tensor` or `numpy.array`] or `torch.Tensor` or `numpy.array`
        """
        coords = [c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in coords]
        original_shape = coords[0].shape
        coords = [c.reshape(-1, 1) for c in coords]
        if isinstance(to_numpy, str):
            # Why did we allow `tf` as an option >_<
            # We should phase this out as soon as possible
            if to_numpy == 'tf' or to_numpy == 'torch':
                to_numpy = False
            elif to_numpy == 'np':
                to_numpy = True
            else:
                raise ValueError(f"Unrecognized `as_type` option: '{to_numpy}'")

        us = [
            self._compute_u(net, con, *coords)
            for con, net in zip(self.conditions, self.nets)
        ]
        if not no_reshape:
            us = [u.reshape(*original_shape) for u in us]
        if to_numpy:
            us = [u.detach().cpu().numpy() for u in us]

        return us if len(self.nets) > 1 else us[0]


class GenericSolution(BaseSolution):
    def _compute_u(self, net, condition, *coords):
        return condition.enforce(net, *coords)


class GenericSolver(BaseSolver):
    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return GenericSolution(nets, conditions)


class SolverSpherical(BaseSolver):
    r"""A solver class for solving PDEs in spherical coordinates

    :param pde_system:
        The PDE system to solve, which maps a tuple of three coordinates to a tuple of PDE residuals,
        both the coordinates and PDE residuals must have shape (n_samples, 1).
    :type pde_system: callable
    :param conditions:
        List of boundary conditions for each target function.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param r_min:
        Radius for inner boundary (:math:`r_0>0`).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type r_min: float, optional
    :param r_max:
        Radius for outer boundary (:math:`r_1>r_0`).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type r_max: float, optional
    :param nets:
        List of neural networks for parameterized solution.
        If provided, length of ``nets`` must equal that of ``conditions``
    :type nets: list[torch.nn.Module], optional
    :param train_generator:
        Generator for sampling training points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``train_generator`` must be specified if ``r_min`` and ``r_max`` are not set.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param valid_generator:
        Generator for sampling validation points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``valid_generator`` must be specified if ``r_min`` and ``r_max`` are not set.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param analytic_solutions:
        Analytical solutions to be compared with neural net solutions.
        It maps a tuple of three coordinates to a tuple of function values.
        Output shape should match that of ``nets``.
    :type analytic_solutions: callable, optional
    :param optimizer:
        Optimizer to be used for training.
        Defaults to a ``torch.optim.Adam`` instance that trains on all parameters of ``nets``.
    :type optimizer: ``torch.nn.optim.Optimizer``, optional
    :param loss_fn:
        The loss function used for training.

        - If a str, must be present in the keys of `neurodiffeq.losses._losses`.
        - If a `torch.nn.modules.loss._Loss` instance, just pass the instance.
        - If any other callable, it must map
          A) a residual tensor (shape `(n_points, n_equations)`),
          B) a function values tuple (length `n_funcs`, each element a tensor of shape `(n_points, 1)`), and
          C) a coordinate values tuple (length `n_coords`, each element a tensor of shape `(n_coords, 1)`
          to a tensor of empty shape (i.e. a scalar). The returned tensor must be connected to the computational graph,
          so that backpropagation can be performed.

    :type loss_fn:
        str or `torch.nn.moduesl.loss._Loss` or callable
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param metrics:
        Additional metrics to be logged (besides loss). ``metrics`` should be a dict where

        - Keys are metric names (e.g. 'analytic_mse');
        - Values are functions (callables) that computes the metric value.
          These functions must accept the same input as the differential equation ``diff_eq``.

    :type metrics: dict, optional
    :param enforcer:
        A function of signature
        ``enforcer(net: nn.Module, cond: neurodiffeq.conditions.BaseCondition,
        coords: Tuple[torch.Tensor]) -> torch.Tensor``
        that returns the dependent variable value evaluated on the batch.
    :type enforcer: callable
    :param n_output_units:
        Number of output units for each neural network.
        Ignored if ``nets`` is specified.
        Defaults to 1.
    :type n_output_units: int, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    """

    def __init__(self, pde_system, conditions, r_min=None, r_max=None,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None,
                 optimizer=None, loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, enforcer=None,
                 n_output_units=1,
                 # deprecated arguments are listed below
                 shuffle=None, batch_size=None):

        if train_generator is None or valid_generator is None:
            if r_min is None or r_max is None:
                raise ValueError(f"Either generator is not provided, r_min and r_max should be both provided: "
                                 f"got r_min={r_min}, r_max={r_max}, train_generator={train_generator}, "
                                 f"valid_generator={valid_generator}")

        if train_generator is None:
            train_generator = GeneratorSpherical(512, r_min, r_max, method='equally-spaced-noisy')

        if valid_generator is None:
            valid_generator = GeneratorSpherical(512, r_min, r_max, method='equally-spaced-noisy')

        self.r_min, self.r_max = r_min, r_max
        self.enforcer = enforcer

        super(SolverSpherical, self).__init__(
            diff_eqs=pde_system,
            conditions=conditions,
            nets=nets,
            train_generator=train_generator,
            valid_generator=valid_generator,
            analytic_solutions=analytic_solutions,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_batches_train=n_batches_train,
            n_batches_valid=n_batches_valid,
            metrics=metrics,
            n_input_units=3,
            n_output_units=n_output_units,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def _auto_enforce(self, net, cond, *coordinates):
        r"""Enforce condition on network with inputs. If self.enforcer is set, use it.
        Otherwise, fill cond.enforce() with as many arguments as needed.

        :param net: Network for parameterized solution.
        :type net: torch.nn.Module
        :param cond: Condition (a.k.a. parameterization) for the network.
        :type cond: `neurodiffeq.conditions.BaseCondition`
        :param coordinates: A tuple of vectors, each with shape = (-1, 1).
        :type coordinates: tuple[torch.Tensor]
        :return: Function values at sampled points.
        :rtype: torch.Tensor
        """
        if self.enforcer:
            return self.enforcer(net, cond, coordinates)

        # Base .enforce takes a variable length *arg; n_params should be deduced from .parameterize
        if cond.__class__.enforce == BaseCondition.enforce:
            n_params = len(signature(cond.parameterize).parameters)
        else:
            n_params = len(signature(cond.enforce).parameters)
        coordinates = coordinates[:n_params - 1]
        return cond.enforce(net, *coordinates)

    def compute_func_val(self, net, cond, *coordinates):
        r"""Enforce condition on network with inputs. If self.enforcer is set, use it.
        Otherwise, fill cond.enforce() with as many arguments as needed.

        :param net: Network for parameterized solution.
        :type net: torch.nn.Module
        :param cond: Condition (a.k.a. parameterization) for the network.
        :type cond: `neurodiffeq.conditions.BaseCondition`
        :param coordinates: A tuple of vectors, each with shape = (-1, 1).
        :type coordinates: tuple[torch.Tensor]
        :return: Function values at sampled points.
        :rtype: torch.Tensor
        """
        return self._auto_enforce(net, cond, *coordinates)

    def get_solution(self, copy=True, best=True, harmonics_fn=None):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :param harmonics_fn:
            If set, use it as function basis for returned solution.
        :type harmonics_fn: callable
        :return: The solution after training.
        :rtype: ``neurodiffeq.solvers.BaseSolution``
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        if harmonics_fn:
            return SolutionSphericalHarmonics(nets, conditions, harmonics_fn=harmonics_fn)
        else:
            return SolutionSpherical(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(SolverSpherical, self)._get_internal_variables()
        available_variables.update({
            'r_min': self.r_min,
            'r_max': self.r_max,
            'enforcer': self.enforcer,
        })
        return available_variables


class SolutionSpherical(BaseSolution):
    def _compute_u(self, net, condition, rs, thetas, phis):
        return condition.enforce(net, rs, thetas, phis)


class SolutionSphericalHarmonics(SolutionSpherical):
    r"""A solution to a PDE (system) in spherical coordinates.

    :param nets: List of networks that takes in radius tensor and outputs the coefficients of spherical harmonics.
    :type nets: list[`torch.nn.Module`]
    :param conditions: List of conditions to be enforced on each nets; must be of the same length as nets.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param harmonics_fn: Mapping from :math:`\theta` and :math:`\phi` to basis functions, e.g., spherical harmonics.
    :type harmonics_fn: callable
    :param max_degree: **DEPRECATED and SUPERSEDED** by ``harmonics_fn``. Highest used for the harmonic basis.
    :type max_degree: int
    """

    def __init__(self, nets, conditions, max_degree=None, harmonics_fn=None):
        super(SolutionSphericalHarmonics, self).__init__(nets, conditions)
        if (harmonics_fn is None) and (max_degree is None):
            raise ValueError("harmonics_fn should be specified")

        if max_degree is not None:
            warnings.warn(
                "`max_degree` is DEPRECATED; pass `harmonics_fn` instead, which takes precedence",
                FutureWarning,
            )
            self.harmonics_fn = RealSphericalHarmonics(max_degree=max_degree)

        if harmonics_fn is not None:
            self.harmonics_fn = harmonics_fn

    def _compute_u(self, net, condition, rs, thetas, phis):
        products = condition.enforce(net, rs) * self.harmonics_fn(thetas, phis)
        return torch.sum(products, dim=1)


class Solution1D(BaseSolution):
    def _compute_u(self, net, condition, ts):
        return condition.enforce(net, ts)


class Solver1D(BaseSolver):
    r"""A solver class for solving ODEs (single-input differential equations)

    :param ode_system:
        The ODE system to solve, which maps a torch.Tensor to a tuple of ODE residuals,
        both the input and output must have shape (n_samples, 1).
    :type ode_system: callable
    :param conditions:
        List of conditions for each target function.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param t_min:
        Lower bound of input (start time).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type t_min: float, optional
    :param t_max:
        Upper bound of input (start time).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type t_max: float, optional
    :param nets:
        List of neural networks for parameterized solution.
        If provided, length of ``nets`` must equal that of ``conditions``
    :type nets: list[torch.nn.Module], optional
    :param train_generator:
        Generator for sampling training points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``train_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param valid_generator:
        Generator for sampling validation points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``valid_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param analytic_solutions:
        Analytical solutions to be compared with neural net solutions.
        It maps a torch.Tensor to a tuple of function values.
        Output shape should match that of ``nets``.
    :type analytic_solutions: callable, optional
    :param optimizer:
        Optimizer to be used for training.
        Defaults to a ``torch.optim.Adam`` instance that trains on all parameters of ``nets``.
    :type optimizer: ``torch.nn.optim.Optimizer``, optional
    :param loss_fn:
        The loss function used for training.

        - If a str, must be present in the keys of `neurodiffeq.losses._losses`.
        - If a `torch.nn.modules.loss._Loss` instance, just pass the instance.
        - If any other callable, it must map
          A) a residual tensor (shape `(n_points, n_equations)`),
          B) a function values tuple (length `n_funcs`, each element a tensor of shape `(n_points, 1)`), and
          C) a coordinate values tuple (length `n_coords`, each element a tensor of shape `(n_coords, 1)`
          to a tensor of empty shape (i.e. a scalar). The returned tensor must be connected to the computational graph,
          so that backpropagation can be performed.

    :type loss_fn:
        str or `torch.nn.moduesl.loss._Loss` or callable
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param metrics:
        Additional metrics to be logged (besides loss). ``metrics`` should be a dict where

        - Keys are metric names (e.g. 'analytic_mse');
        - Values are functions (callables) that computes the metric value.
          These functions must accept the same input as the differential equation ``ode_system``.

    :type metrics: dict[str, callable], optional
    :param n_output_units:
        Number of output units for each neural network.
        Ignored if ``nets`` is specified.
        Defaults to 1.
    :type n_output_units: int, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    """

    def __init__(self, ode_system, conditions, t_min=None, t_max=None,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None, optimizer=None,
                 loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, n_output_units=1,
                 # deprecated arguments are listed below
                 batch_size=None, shuffle=None):

        if train_generator is None or valid_generator is None:
            if t_min is None or t_max is None:
                raise ValueError(f"Either generator is not provided, t_min and t_max should be both provided: \n"
                                 f"got t_min={t_min}, t_max={t_max}, "
                                 f"train_generator={train_generator}, valid_generator={valid_generator}")

        if train_generator is None:
            train_generator = Generator1D(32, t_min=t_min, t_max=t_max, method='equally-spaced-noisy')
        if valid_generator is None:
            valid_generator = Generator1D(32, t_min=t_min, t_max=t_max, method='equally-spaced')

        self.t_min, self.t_max = t_min, t_max

        super(Solver1D, self).__init__(
            diff_eqs=ode_system,
            conditions=conditions,
            nets=nets,
            train_generator=train_generator,
            valid_generator=valid_generator,
            analytic_solutions=analytic_solutions,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_batches_train=n_batches_train,
            n_batches_valid=n_batches_valid,
            metrics=metrics,
            n_input_units=1,
            n_output_units=n_output_units,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return Solution1D(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(Solver1D, self)._get_internal_variables()
        available_variables.update({
            't_min': self.t_min,
            't_max': self.t_max,
        })
        return available_variables


class BundleSolution1D(BaseSolution):
    def _compute_u(self, net, condition, *ts):
        return condition.enforce(net, *ts)


class BundleSolver1D(BaseSolver):
    r"""A solver class for solving ODEs (single-input differential equations)
    , or a bundle of ODEs for different values of its parameters and/or conditions

    :param ode_system:
        The ODE system to solve, which maps a torch.Tensor or a tuple of torch.Tensors, to a tuple of ODE residuals,
        both the input and output must have shape (n_samples, 1).
    :type ode_system: callable
    :param conditions:
        List of conditions for each target function.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param t_min:
        Lower bound of input (start time).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type t_min: float, optional
    :param t_max:
        Upper bound of input (start time).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type t_max: float, optional
    :param theta_min:
        Lower bound of input (parameters and/or conditions). If conditions are included in the bundle,
        the order should match the one inferred by the values of the ``bundle_param_lookup`` input
        in the ``neurodiffeq.conditions.BundleIVP``.
        Defaults to None.
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type theta_min: float or tuple, optional
    :param theta_max:
        Upper bound of input (parameters and/or conditions). If conditions are included in the bundle,
        the order should match the one inferred by the values of the ``bundle_param_lookup`` input
        in the ``neurodiffeq.conditions.BundleIVP``.
        Defaults to None.
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type theta_max: float or tuple, optional
    :param eq_param_index:
        Index (or indices) of bundle parameter that appears in the equation.
        E.g., if there are 5 bundle parameters generated and the first (index 0) and last (index 4) parameters are used
        in the equation, then eq_param_index should be (0, 4).
        The signature of the original equation should have, in addition to functions to solve for and coordinates,
        2 more parameters, corresponding to bundle parameters indexed at 0 and 4, in that order.
    :type eq_param_index: int or tuple[int], optional
    :param nets:
        List of neural networks for parameterized solution.
        If provided, length of ``nets`` must equal that of ``conditions``
    :type nets: list[torch.nn.Module], optional
    :param train_generator:
        Generator for sampling training points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``train_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
        If provided, the generator must generate bundle parameters too.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param valid_generator:
        Generator for sampling validation points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``valid_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
        If provided, the generator must generate bundle parameters too.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param analytic_solutions:
        Analytical solutions to be compared with neural net solutions.
        It maps a torch.Tensor to a tuple of function values.
        Output shape should match that of ``nets``.
    :type analytic_solutions: callable, optional
    :param optimizer:
        Optimizer to be used for training.
        Defaults to a ``torch.optim.Adam`` instance that trains on all parameters of ``nets``.
    :type optimizer: ``torch.nn.optim.Optimizer``, optional
    :param loss_fn:
        The loss function used for training.

        - If a str, must be present in the keys of `neurodiffeq.losses._losses`.
        - If a `torch.nn.modules.loss._Loss` instance, just pass the instance.
        - If any other callable, it must map
          A) a residual tensor (shape `(n_points, n_equations)`),
          B) a function values tuple (length `n_funcs`, each element a tensor of shape `(n_points, 1)`), and
          C) a coordinate values tuple (length `n_coords`, each element a tensor of shape `(n_coords, 1)`
          to a tensor of empty shape (i.e. a scalar). The returned tensor must be connected to the computational graph,
          so that backpropagation can be performed.

    :type loss_fn:
        str or `torch.nn.moduesl.loss._Loss` or callable
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param metrics:
        Additional metrics to be logged (besides loss). ``metrics`` should be a dict where

        - Keys are metric names (e.g. 'analytic_mse');
        - Values are functions (callables) that computes the metric value.
          These functions must accept the same input as the differential equation ``ode_system``.

    :type metrics: dict[str, callable], optional
    :param n_output_units:
        Number of output units for each neural network.
        Ignored if ``nets`` is specified.
        Defaults to 1.
    :type n_output_units: int, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    """

    def __init__(self, ode_system, conditions, t_min, t_max,
                 theta_min=None, theta_max=None, eq_param_index=(),
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None, optimizer=None,
                 loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, n_output_units=1,
                 # deprecated arguments are listed below
                 batch_size=None, shuffle=None):

        if train_generator is None or valid_generator is None:
            if t_min is None or t_max is None:
                raise ValueError(f"Either generator is not provided, t_min and t_max should be both provided: \n"
                                 f"got t_min={t_min}, t_max={t_max}, "
                                 f"train_generator={train_generator}, valid_generator={valid_generator}")

        if isinstance(theta_min, (float, int)):
            theta_min = (theta_min,)
        elif theta_min is None:
            theta_min = ()

        if isinstance(theta_max, (float, int)):
            theta_max = (theta_max,)
        elif theta_max is None:
            theta_max = ()

        if len(theta_min) != len(theta_max):
            raise ValueError(
                f"length of theta_min and theta_max must be equal, " f"got {len(theta_min)} != {len(theta_max)}"
            )

        r_min = (t_min,) + tuple(theta_min)
        r_max = (t_max,) + tuple(theta_max)

        n_input_units = len(r_min)

        if train_generator is None:
            train_generator = Generator1D(32, t_min=t_min, t_max=t_max, method='equally-spaced-noisy')
            for i in range(n_input_units - 1):
                train_generator ^= Generator1D(32, t_min=r_min[i + 1], t_max=r_max[i + 1],
                                               method='equally-spaced-noisy')
        if valid_generator is None:
            valid_generator = Generator1D(32, t_min=t_min, t_max=t_max, method='equally-spaced')
            for i in range(n_input_units - 1):
                valid_generator ^= Generator1D(32, t_min=r_min[i + 1], t_max=r_max[i + 1], method='equally-spaced')

        self.r_min, self.r_max = r_min, r_max

        # number of functions equals number of conditions supplied
        N_FUNCTIONS = len(conditions)
        # there is only 1 coordinate (usually time `t`) for ODEs
        N_COORDS = 1

        # Note: It is intentionally design in this way where `eq_param_index` and `self.eq_param_index`
        # both contain values offset by `N_FUNCTIONS + N_COORDS`. The first one (not bound to `self`) is used for the
        # `eq_param_filter` closure, while the second one (bound to `self`) is used for `_get_internal_variables()`.
        eq_param_index = tuple(N_FUNCTIONS + N_COORDS + idx for idx in eq_param_index)
        self.eq_param_index = eq_param_index

        # TODO: check and warn if there are variables neither used by conditions nor by equations

        def _diff_eqs_wrapper(*variables):
            funcs_and_coords = variables[:N_FUNCTIONS + N_COORDS]
            eq_params = tuple(variables[idx] for idx in eq_param_index)
            return ode_system(*funcs_and_coords, *eq_params)

        super(BundleSolver1D, self).__init__(
            diff_eqs=_diff_eqs_wrapper,
            conditions=conditions,
            nets=nets,
            train_generator=train_generator,
            valid_generator=valid_generator,
            analytic_solutions=analytic_solutions,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_batches_train=n_batches_train,
            n_batches_valid=n_batches_valid,
            metrics=metrics,
            n_input_units=n_input_units,
            n_output_units=n_output_units,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return BundleSolution1D(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(BundleSolver1D, self)._get_internal_variables()
        available_variables.update({
            'r_min': self.r_min,
            'r_max': self.r_max,
            'eq_param_index': self.eq_param_index,
        })
        return available_variables


class Solution2D(BaseSolution):
    def _compute_u(self, net, condition, xs, ys):
        return condition.enforce(net, xs, ys)


class Solver2D(BaseSolver):
    r"""A solver class for solving PDEs in 2 dimensions.

    :param pde_system:
        The PDE system to solve, which maps two ``torch.Tensor``s to PDE residuals (``tuple[torch.Tensor]``),
        both the input and output must have shape (n_samples, 1).
    :type pde_system: callable
    :param conditions:
        List of conditions for each target function.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param xy_min:
        The lower bound of 2 dimensions.
        If we only care about :math:`x \geq x_0` and :math:`y \geq y_0`,
        then `xy_min` is `(x_0, y_0)`.
        Only needed when train_generator or valid_generator are not specified.
        Defaults to None
    :type xy_min: tuple[float, float], optional
    :param xy_max:
        The upper bound of 2 dimensions.
        If we only care about :math:`x \leq x_1` and :math:`y \leq y_1`, then `xy_min` is `(x_1, y_1)`.
        Only needed when train_generator or valid_generator are not specified.
        Defaults to None
    :type xy_max: tuple[float, float], optional
    :param nets:
        List of neural networks for parameterized solution.
        If provided, length of ``nets`` must equal that of ``conditions``
    :type nets: list[torch.nn.Module], optional
    :param train_generator:
        Generator for sampling training points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``train_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param valid_generator:
        Generator for sampling validation points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``valid_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param analytic_solutions:
        Analytical solutions to be compared with neural net solutions.
        It maps a torch.Tensor to a tuple of function values.
        Output shape should match that of ``nets``.
    :type analytic_solutions: callable, optional
    :param optimizer:
        Optimizer to be used for training.
        Defaults to a ``torch.optim.Adam`` instance that trains on all parameters of ``nets``.
    :type optimizer: ``torch.nn.optim.Optimizer``, optional
    :param loss_fn:
        The loss function used for training.

        - If a str, must be present in the keys of `neurodiffeq.losses._losses`.
        - If a `torch.nn.modules.loss._Loss` instance, just pass the instance.
        - If any other callable, it must map
          A) a residual tensor (shape `(n_points, n_equations)`),
          B) a function values tuple (length `n_funcs`, each element a tensor of shape `(n_points, 1)`), and
          C) a coordinate values tuple (length `n_coords`, each element a tensor of shape `(n_coords, 1)`
          to a tensor of empty shape (i.e. a scalar). The returned tensor must be connected to the computational graph,
          so that backpropagation can be performed.

    :type loss_fn:
        str or `torch.nn.moduesl.loss._Loss` or callable
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param metrics:
        Additional metrics to be logged (besides loss). ``metrics`` should be a dict where

        - Keys are metric names (e.g. 'analytic_mse');
        - Values are functions (callables) that computes the metric value.
          These functions must accept the same input as the differential equation ``ode_system``.

    :type metrics: dict[str, callable], optional
    :param n_output_units:
        Number of output units for each neural network.
        Ignored if ``nets`` is specified.
        Defaults to 1.
    :type n_output_units: int, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    """

    def __init__(self, pde_system, conditions, xy_min=None, xy_max=None,
                 nets=None, train_generator=None, valid_generator=None, analytic_solutions=None, optimizer=None,
                 loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, n_output_units=1,
                 # deprecated arguments are listed below
                 batch_size=None, shuffle=None):

        if train_generator is None or valid_generator is None:
            if xy_min is None or xy_max is None:
                raise ValueError(f"Either generator is not provided, xy_min and xy_max should be both provided: \n"
                                 f"got xy_min={xy_min}, xy_max={xy_max}, "
                                 f"train_generator={train_generator}, valid_generator={valid_generator}")

        if train_generator is None:
            train_generator = Generator2D((32, 32), xy_min=xy_min, xy_max=xy_max, method='equally-spaced-noisy')
        if valid_generator is None:
            valid_generator = Generator2D((32, 32), xy_min=xy_min, xy_max=xy_max, method='equally-spaced')

        self.xy_min, self.xy_max = xy_min, xy_max

        super(Solver2D, self).__init__(
            diff_eqs=pde_system,
            conditions=conditions,
            nets=nets,
            train_generator=train_generator,
            valid_generator=valid_generator,
            analytic_solutions=analytic_solutions,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_batches_train=n_batches_train,
            n_batches_valid=n_batches_valid,
            metrics=metrics,
            n_input_units=2,
            n_output_units=n_output_units,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return Solution2D(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(Solver2D, self)._get_internal_variables()
        available_variables.update({
            'xy_min': self.xy_min,
            'xy_max': self.xy_max,
        })
        return available_variables
