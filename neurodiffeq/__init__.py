import torch
from .neurodiffeq import diff
from .utils import set_tensor_type as _set_tensor_type  # Don't export this function

_set_tensor_type(float_bits=64)