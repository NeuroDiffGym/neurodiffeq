import torch
import warnings
from .neurodiffeq import diff
from .utils import set_tensor_type as _set_tensor_type  # Don't export this function

# Set default float type to 64 bits
_set_tensor_type(float_bits=64)

# Turn on deprecation warning by default
warnings.simplefilter('always', DeprecationWarning)
