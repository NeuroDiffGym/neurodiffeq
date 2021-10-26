import numpy as np
import torch
import random
from neurodiffeq.utils import set_seed


def test_set_seed():
    set_seed(10)
    x1 = np.random.rand(5)
    y1 = torch.rand(5)
    z1 = random.random()

    set_seed(10)
    x2 = np.random.rand(5)
    y2 = torch.rand(5)
    z2 = random.random()

    assert (x1 == x2).all()
    assert (y1 == y2).all()
    assert z1 == z2
