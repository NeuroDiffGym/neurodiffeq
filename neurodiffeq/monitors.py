import math
import torch
import warnings
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

from ._version_utils import deprecated_alias
from .function_basis import RealSphericalHarmonics
from .generators import Generator3D


class BaseMonitor(ABC):
    @abstractmethod
    def check(self, nets, conditions, history):
        pass


