import torch
import warnings
from .neurodiffeq import diff
from .utils import set_tensor_type as _set_tensor_type  # Don't export this function

from . import conditions
from . import function_basis
from . import generators
from . import networks
from . import neurodiffeq
from . import operators
from . import pde
from . import ode
from . import pde_spherical
from . import temporal

# Set default float type to 64 bits
_set_tensor_type(float_bits=64)

# Turn on deprecation warning by default
warnings.simplefilter('always', DeprecationWarning)
