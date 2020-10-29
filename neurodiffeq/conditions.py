import torch
import warnings


class BaseCondition:
    r"""Base class for all conditions.
    A condition is a tool to `re-parameterize` the output(s) of a neural network.
        such that the re-parameterized output(s) will automatically satisfy initial conditions (ICs)
        and boundary conditions (BCs) of the PDEs/ODEs that are being solved.

    .. note::
        The nouns "(re)parameterization" and "condition" are used interchangeably in the documentation and the library.
        The verbs "(re)parameterize" and "enforce" are different in that *(re)parameterize* is said of network outputs
            whereas *enforce* is said of networks themselves.
    """

    def __init__(self):
        self.ith_unit = None

    def parameterize(self, output_tensor, *input_tensors):
        f"""[ABSTRACT METHOD] Re-parameterize output(s) of a network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.nn.Tensor`
        :param input_tensors: Inputs to the neural network; i.e., sampled coordinates; i.e., independent variables.
        :type input_tensors: tuple[`torch.nn.Tensor`]
        :return: the re-parameterized output of the network
        :rtype: `torch.Tensor`

        .. note:: 
            This method is abstract for {self.__class__.__name__}
        """
        raise ValueError(f"Abstract {self.__class__.__name__} cannot be parameterized")

    def enforce(self, net, *coordinates):
        f"""Enforces this condition on a network.

        :param net: The network whose output is to be re-parameterized.
        :type net: `torch.nn.Module`
        :param coordinates: Inputs of the neural network.
        :type coordinates: tuple[`torch.Tensor`]
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`
        """
        return self.parameterize(net(*coordinates), *coordinates)

    def set_impose_on(self, ith_unit):
        r"""[DEPRECATED] When training several functions with a single (multi-output network), this method is called
            (by a `Solver` class or a `solve` function) to keep track of which output is being parameterized.

        :param ith_unit: The index of network output to be parameterized.
        :type ith_unit: int

        .. note::
            This method is deprecated and retained for backward compatibility only.
        """

        warnings.warn(f"`{self.__class__.__name__}.set_impose_on` is deprecated and will be removed in the future")
        self.ith_unit = ith_unit


class NoCondition(BaseCondition):
    r"""A polymorphic condition where no re-parameterization will be performed.

    .. note::
        This condition is called *polymorphic* because it can be enforced on networks of arbitrary input/output sizes.
    """

    def parameterize(self, output_tensor, *input_tensors):
        return output_tensor


        :type net: `torch.nn.Module`
        :param coordinates: Inputs of the neural network.
        :type coordinates: tuple[`torch.Tensor`]
        :return: The re-parameterized output, where actually no condition is enforced.
        :rtype: `torch.Tensor`
        """
        return net(*coordinates)
