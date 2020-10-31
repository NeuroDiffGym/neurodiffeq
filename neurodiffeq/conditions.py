import numpy as np
import torch
import warnings


class BaseCondition:
    r"""Base class for all conditions.

    A condition is a tool to `re-parameterize` the output(s) of a neural network.
    such that the re-parameterized output(s) will automatically satisfy initial conditions (ICs)
    and boundary conditions (BCs) of the PDEs/ODEs that are being solved.

    .. note::
        - The nouns *(re-)parameterization* and *condition* are used interchangeably in the documentation and library.
        - The verbs *(re-)parameterize* and *enforce* are different in that:

          - *(re)parameterize* is said of network outputs;
          - *enforce* is said of networks themselves.
    """

    def __init__(self):
        self.ith_unit = None

    def parameterize(self, output_tensor, *input_tensors):
        r"""Re-parameterize output(s) of a network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param input_tensors: Inputs to the neural network; i.e., sampled coordinates; i.e., independent variables.
        :type input_tensors: `torch.Tensor`
        :return: the re-parameterized output of the network
        :rtype: `torch.Tensor`

        .. note:: 
            This method is **abstract** for BaseCondition
        """
        raise ValueError(f"Abstract {self.__class__.__name__} cannot be parameterized")

    def enforce(self, net, *coordinates):
        r"""Enforces this condition on a network.

        :param net: The network whose output is to be re-parameterized.
        :type net: `torch.nn.Module`
        :param coordinates: Inputs of the neural network.
        :type coordinates: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`
        """
        # concatenate the coordinates and pass to network
        network_output = net(torch.cat(coordinates, dim=1))
        # if `ith_unit` is set, the condition will only be enforced on the i-th output unit
        if self.ith_unit is not None:
            network_output = network_output[:, self.ith_unit].view(-1, 1)
        # parameterize the raw output and return
        return self.parameterize(network_output, *coordinates)

    def set_impose_on(self, ith_unit):
        r"""**[DEPRECATED]** When training several functions with a single, multi-output network, this method is called
        (by a `Solver` class or a `solve` function) to keep track of which output is being parameterized.

        :param ith_unit: The index of network output to be parameterized.
        :type ith_unit: int

        .. note::
            This method is deprecated and retained for backward compatibility only. Users interested in enforcing
            conditions on multi-output networks should consider using a ``neurodiffeq.neurodiffeq.EnsembleCondition``.
        """

        warnings.warn(f"`{self.__class__.__name__}.set_impose_on` is deprecated and will be removed in the future")
        self.ith_unit = ith_unit


class IrregularBoundaryCondition(BaseCondition):
    # Is there a more elegant solution?
    def in_domain(self, *coordinates):
        """Given the coordinates (numpy.ndarray), the methods returns an boolean array indicating
        whether the points lie within the domain.

        :param coordinates: Input tensors, each with shape (n_samples, 1).
        :type coordinates: `numpy.ndarray`
        :return: Whether each point lies within the domain.
        :rtype: `numpy.ndarray`

        .. note::
            - This method is meant to be used by monitors for irregular domain visualization.
        """

        # returns straight `True`-s by default; i.e., all points are considered within domain
        return np.ones_like(coordinates[0], dtype=np.bool)


class EnsembleCondition(BaseCondition):
    r"""An ensemble condition that enforces sub-conditions on individual output units of the networks.
    """

    def __init__(self, *sub_conditions):
        super(EnsembleCondition, self).__init__()
        self.conditions = sub_conditions

    def parameterize(self, output_tensor, *input_tensors):
        r"""Re-parameterizes each column in output_tensor individually, using its corresponding sub-condition.
        This is useful when solving differential equations with a single, multi-output network.

        :param output_tensor: Output of the neural network. Number of units (.shape[1]) must equal number of sub-conditions.
        :type output_tensor: `torch.Tensor`
        :param input_tensors: Inputs to the neural network; i.e., sampled coordinates; i.e., independent variables.
        :type input_tensors: `torch.Tensor`
        :return: The column-wise re-parameterized network output, concatenated across columns so that it's still one tensor.
        :rtype: `torch.Tensor`
        """
        if output_tensor.shape[1] != len(self.conditions):
            raise ValueError(f"number of output units ({output_tensor.shape[1]}) "
                             f"differs from number of conditions ({len(self.conditions)})")
        return torch.cat([
            con.parameterize(output_tensor[:, i].view(-1, 1), *input_tensors) for i, con in enumerate(self.conditions)
        ], dim=1)


class NoCondition(BaseCondition):
    r"""A polymorphic condition where no re-parameterization will be performed.

    .. note::
        This condition is called *polymorphic* because it can be enforced on networks of arbitrary input/output sizes.
    """

    def parameterize(self, output_tensor, *input_tensors):
        f"""Performs no re-parameterization, or identity parameterization, in this case.
        """
        return output_tensor


class IVP(BaseCondition):
    r"""An initial value problem of one of the following forms:

    - Dirichlet condition: :math:`x(t)\bigg|_{t = t_0} = x_0`.
    - Neumann condition: :math:`\displaystyle\frac{\partial x}{\partial t}\bigg|_{t = t_0} = x_0'`.

    :param t_0: The initial time.
    :type t_0: float
    :param x_0: The initial value of :math:`x`. :math:`x(t)\bigg|_{t = t_0} = x_0`.
    :type x_0: float
    :param x_0_prime: The initial derivative of :math:`x` w.r.t. :math:`t`. :math:`\displaystyle\frac{\partial x}{\partial t}\bigg|_{t = t_0} = x_0'`, defaults to None.
    :type x_0_prime: float, optional
    """

    def __init__(self, t_0, x_0, x_0_prime=None):
        super().__init__()
        self.t_0, self.x_0, self.x_0_prime = t_0, x_0, x_0_prime

    def parameterize(self, output_tensor, t):
        r"""Re-parameterize outputs such that the Dirichlet/Neumann condition is satisfied.

        - For Dirichlet condition, the re-parameterization is
          :math:`\displaystyle x(t) = x_0 + \left(1 - e^{-(t-t_0)}\right) \mathrm{ANN}(t)`
          where :math:`\mathrm{ANN}` is the neural network.
        - For Neumann condition, the re-parameterization is
          :math:`\displaystyle x(t) = x_0 + (t-t_0) x'_0 + \left(1 - e^{-(t-t_0)}\right)^2 \mathrm{ANN}(t)`
          where :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param t: Input to the neural network; i.e., sampled time-points; i.e., independent variables.
        :type t: `torch.Tensor`
        :return: the re-parameterized output of the network
        :rtype: `torch.Tensor`
        """
        if self.x_0_prime is None:
            return self.x_0 + (1 - torch.exp(-t + self.t_0)) * output_tensor
        else:
            return self.x_0 + (t - self.t_0) * self.x_0_prime + ((1 - torch.exp(-t + self.t_0)) ** 2) * output_tensor


class DirichletBVP(BaseCondition):
    r"""A double-ended Dirichlet boundary condition:
    :math:`x(t)\bigg|_{t = t_0} = x_0` and :math:`x(t)\bigg|_{t = t_1} = x_1`.

    :param t_0: The initial time.
    :type t_0: float
    :param t_1: The final time.
    :type t_1: float
    :param x_0: The initial value of :math:`x`. :math:`x(t)\bigg|_{t = t_0} = x_0`.
    :type x_0: float
    :param x_1: The initial value of :math:`x`. :math:`x(t)\bigg|_{t = t_1} = x_1`.
    :type x_1: float
    """

    def __init__(self, t_0, x_0, t_1, x_1):
        super().__init__()
        self.t_0, self.x_0, self.t_1, self.x_1 = t_0, x_0, t_1, x_1

    def parameterize(self, output_tensor, t):
        r"""Re-parameterize outputs such that the Dirichlet condition is satisfied on both ends of the domain.

        The re-parameterization is
        :math:`\displaystyle x(t)=(1-\tilde{t})x_0+\tilde{t}x_1+\left(1-e^{(1-\tilde{t})\tilde{t}}\right)\mathrm{ANN}(t)`,
        where :math:`\displaystyle \tilde{t} = \frac{t-t_0}{t_1-t_0}` and :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param t: Input to the neural network; i.e., sampled time-points or another independent variable.
        :type t: `torch.Tensor`
        """

        t_tilde = (t - self.t_0) / (self.t_1 - self.t_0)
        return self.x_0 * (1 - t_tilde) \
               + self.x_1 * t_tilde \
               + (1 - torch.exp((1 - t_tilde) * t_tilde)) * output_tensor
