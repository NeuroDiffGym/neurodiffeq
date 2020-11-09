import torch
import torch.autograd as autograd
import warnings


def unsafe_diff(x, t, order=1):
    r"""The derivative of a variable with respect to another.
    While there's no requirement for shapes, errors could occur in some cases.
    See `this issue <https://github.com/odegym/neurodiffeq/issues/63#issue-719436650>`_ for details

    :param x: The :math:`x` in :math:`\displaystyle\frac{\partial x}{\partial t}`.
    :type x: `torch.Tensor`
    :param t: The :math:`t` in :math:`\displaystyle\frac{\partial x}{\partial t}`.
    :type t: `torch.Tensor`
    :param order: The order of the derivative, defaults to 1.
    :type order: int
    :returns: The derivative.
    :rtype: `torch.Tensor`
    """
    ones = torch.ones_like(x)
    der, = autograd.grad(x, t, create_graph=True, grad_outputs=ones, allow_unused=True)
    if der is None:
        return torch.zeros_like(t).requires_grad_(True)
    for i in range(1, order):
        ones = torch.ones_like(der)
        der, = autograd.grad(der, t, create_graph=True, grad_outputs=ones, allow_unused=True)
        if der is None:
            return torch.zeros_like(t).requires_grad_(True)
    return der


def safe_diff(x, t, order=1):
    r"""The derivative of a variable with respect to another.
    Both tensors must have a shape of (n_samples, 1)
    See `this issue comment <https://github.com/odegym/neurodiffeq/issues/63#issuecomment-718007133>`_ for details

    :param x: The :math:`x` in :math:`\displaystyle\frac{\partial x}{\partial t}`.
    :type x: `torch.Tensor`
    :param t: The :math:`t` in :math:`\displaystyle\frac{\partial x}{\partial t}`.
    :type t: `torch.Tensor`
    :param order: The order of the derivative, defaults to 1.
    :type order: int
    :returns: The derivative.
    :rtype: `torch.Tensor`
    """
    if len(x.shape) != 2 or len(t.shape) != 2 or x.shape[1] != 1 or t.shape[1] != 1:
        raise ValueError(f"Input shapes must both be (n_samples, 1) starting from neurodiffeq v0.2.0; \n"
                         f"got {x.shape} (for dependent variable) and {t.shape} (for independent variable)"
                         f"In most scenarios, consider reshaping inputs by `x = x.view(-1, 1)`\n"
                         f"For legacy usage, try `from neurodiffeq.neurodiffeq import unsafe_diff as diff`")
    if x.shape != t.shape:
        raise ValueError(f"Input shapes must be the same shape starting from v0.2.0; got {x.shape} != {t.shape}"
                         f"For legacy usage, try `from neurodiffeq.neurodiffeq import unsafe_diff as diff`")
    return unsafe_diff(x, t, order=order)


def diff(x, t, order=1, shape_check=True):
    r"""The derivative of a variable with respect to another.
    Currently, ``diff`` defaults to ``unsafe_diff``, but in a future release, it will default to ``safe_diff``

    :param x: The :math:`x` in :math:`\displaystyle\frac{\partial x}{\partial t}`.
    :type x: `torch.Tensor`
    :param t: The :math:`t` in :math:`\displaystyle\frac{\partial x}{\partial t}`.
    :type t: `torch.Tensor`
    :param order: The order of the derivative, defaults to 1.
    :type order: int
    :param shape_check: Whether to perform shape checking or not, defaults to True (since v0.2.0).
    :type shape_check: bool
    :returns: The derivative.
    :rtype: `torch.Tensor`
    """

    if shape_check:
        return safe_diff(x, t, order=order)
    else:
        if shape_check is None:
            warnings.warn("Currently, `diff` doesn't enforce any restrictions on shapes, "
                          "which will be enforced in v0.2.0\n"
                          "To perform shape checking before v0.2.0, please set shape_check=True\n"
                          "See https://github.com/odegym/neurodiffeq/issues/63#issue-719436650 for more details")
        return unsafe_diff(x, t, order=order)
