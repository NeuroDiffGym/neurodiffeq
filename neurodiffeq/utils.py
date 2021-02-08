from pathlib import Path
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
