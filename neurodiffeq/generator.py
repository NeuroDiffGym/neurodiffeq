"""This module name (`neurodiffeq.generator`) is deprecated, please use `neurodiffeq.generators` (with 's') instead
"""
import warnings
from .generators import Generator1D
from .generators import Generator2D
from .generators import Generator3D
from .generators import GeneratorSpherical
from .generators import BatchGenerator
from .generators import ConcatGenerator
from .generators import EnsembleGenerator
from .generators import FilterGenerator
from .generators import PredefinedGenerator
from .generators import ResampleGenerator
from .generators import StaticGenerator
from .generators import TransformGenerator

warnings.warn(
    f"The module name `neurodiffeq.generator` is deprecated, please use `neurodiffeq.generators` (with 's') instead",
    category=FutureWarning,
    stacklevel=2,
)
