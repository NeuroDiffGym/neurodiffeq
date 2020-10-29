import torch
import warnings


class BaseCondition:
    r"""Base class for all conditions.
    A condition is a tool to `re-parameterize` the output(s) of a neural network.
        such that the re-parameterized output(s) will automatically satisfy initial conditions (ICs)
        and boundary conditions (BCs) of the PDEs/ODEs that are being solved.

    .. note::
        The nouns "(re)parameterization" and "condition" are used interchangeably in the documentation and the library.
        The verbs "(re)parameterize" and "enforce" are used interchangeably in the documentation and the library.
    """

    def __init__(self):
        self.ith_unit = None

    def enforce(self, net, *coordinates):
        f"""[ABSTRACT METHOD] re-parameterize output(s) of a network

        :param net: the network whose output is to be re-parameterized
        :type net: `torch.nn.Module`
        :param coordinates: inputs of the neural network
        :type coordinates: tuple[`torch.Tensor`]
        :return: the re-parameterized output, where the condition is automatically satisfied
        :rtype: `torch.Tensor`

        .. note:: 
            This method is abstract for {self.__class__.__name__}
        """
        raise ValueError(f"Abstract {self.__class__.__name__} cannot be enforced")

    def set_impose_on(self, ith_unit):
        r"""[DEPRECATED] When training several functions with a single (multi-output network), this method is called
            (by a `Solver` class or a `solve` function) to keep track of which output is being parameterized.

        :param ith_unit: the index of network output to be parameterized
        :type ith_unit: int

        .. note::
            This method is deprecated and retained for backward compatibility only.
        """

        warnings.warn(f"`{self.__class__.__name__}.set_impose_on` is deprecated and will be removed in the future")
        self.ith_unit = ith_unit


class NoCondition(BaseCondition):
    r"""A polymorphic condition where no re-parameterization will be performed

    .. note::
        This condition is called *polymorphic* because it can be enforced on networks of arbitrary input/output sizes
    """

    def enforce(self, net, *coordinates):
        """enforce no condition (or an identity re-parameterization) on network output(s)

        :param net: the network whose output is to be re-parameterized
        :type net: `torch.nn.Module`
        :param coordinates: inputs of the neural network
        :type coordinates: tuple[`torch.Tensor`]
        :return: the re-parameterized output, where actually no condition is enforced
        :rtype: `torch.Tensor`
        """
        return net(*coordinates)
