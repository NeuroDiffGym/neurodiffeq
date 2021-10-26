import os
import random
from pathlib import Path
import numpy as np
import torch


def set_tensor_type(device=None, float_bits=32):
    """Set the default torch tensor type to be used with neurodiffeq.

    :param device: Either "cpu" or "cuda" ("gpu"); defaults to "cuda" if available.
    :type device: str
    :param float_bits: Length of float numbers. Either 32 (float) or 64 (double); defaults to 32.
    :type float_bits: int

    .. note:
        The function calls ``torch.set_default_tensor_type`` under the hood.
        Therefore the ``device`` and ``float_bits`` also becomes default tensor type for PyTorch.
    """
    if not isinstance(float_bits, int):
        raise ValueError(f"float_bits must be int, got {type(float_bits)}")
    if float_bits == 32:
        tensor_str = "FloatTensor"
    elif float_bits == 64:
        tensor_str = "DoubleTensor"
    else:
        raise ValueError(f"float_bits must be 32 or 64, got {float_bits}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        type_string = f"torch.{tensor_str}"
    elif device in ["cuda", "gpu"]:
        type_string = f"torch.cuda.{tensor_str}"
    else:
        raise ValueError(f"Unknown device '{device}'; device must be either 'cuda' or 'cpu'")

    torch.set_default_tensor_type(type_string)


def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed_value, ignore_numpy=False, ignore_torch=False, ignore_random=False):
    """
    Set the random seed for the numpy, torch, and random packages.

    :param seed_value: The value of seed.
    :type seed_value: int
    :param ignore_numpy: If True, the seed for numpy.random will not be set. Defaults to False.
    :type ignore_numpy: bool
    :param ignore_torch: If True, the seed for torch will not be set. Defaults to False.
    :type ignore_torch: bool
    :param ignore_random: If True, the seed for `random` will not be set. Defaults to False.
    :type ignore_random: bool
    """
    if not ignore_numpy:
        np.random.seed(seed_value)
    if not ignore_torch:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if not ignore_random:
        random.seed(seed_value)


def get_residual_info(solution, data, diff_eqs, highest_order=0, detach=True):
    # XXX this function is not tested
    from .neurodiffeq import diff

    funcs = solution(data)
    residuals = diff_eqs(*funcs, *data)

    ret = [residuals]
    for order in range(1, highest_order + 1):
        ret.append([[diff(pdr, x) for pdr, x in zip(prev_drs, data)] for prev_drs in ret[-1]])

    if detach:
        def recurse(l):
            for i, entry in enumerate(l):
                if isinstance(entry, torch.Tensor):
                    l[i] = entry.detach()
                else:
                    recurse(entry)

        recurse(ret)

    return ret


def split_columns(mat):
    if len(mat.shape) != 2:
        raise ValueError(f'matrix must have 2 dimensions, but matrix has shape {mat.shape}.')
    return [mat[:, j] for j in range(mat.shape[1])]


def hstack(tensors):
    return torch.stack(tensors, dim=1)


def vstack(tensors):
    return torch.stack(tensors, dim=0)
