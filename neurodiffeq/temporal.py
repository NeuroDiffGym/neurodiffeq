from abc import ABC, abstractmethod
import torch


class Approximator(ABC):
    @abstractmethod
    def __call__(self, time_dimension, *spatial_dimensions):
        pass

    @abstractmethod
    def parameters(self):
        pass

    def loss(self):
        pass


class FirstOrderInitialCondition:
    def __init__(self, u0):
        pass


class BoundaryCondition:
    def __init__(self, form, points):
        pass


def generator_1dspatial(size, x_min, x_max, random=True):
    seg_len = (x_max-x_min) / size
    linspace_lo = x_min + seg_len*0.5
    linspace_hi = x_max - seg_len*0.5
    center = torch.linspace(linspace_lo, linspace_hi, size)
    noise_lo = -seg_len*0.5
    while True:
        if random:
            noise = seg_len*torch.rand(size) + noise_lo
            yield center + noise
        else:
            yield center


def generator_temporal(size, t_min, t_max, random=True):
    seg_len = (t_max - t_min) / size
    linspace_lo = t_min + seg_len * 0.5
    linspace_hi = t_max - seg_len * 0.5
    center = torch.linspace(linspace_lo, linspace_hi, size)
    noise_lo = -seg_len * 0.5
    while True:
        if random:
            noise = seg_len * torch.rand(size) + noise_lo
            yield center + noise
        else:
            yield center


class Monitor1DSpatialTemporal:
    def __init__(self, check_on_x, check_on_t, check_every):
        pass


def solve_1dspatial_temporal(
        pde, initial_condition, boundary_conditions,
        train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
        batch_size, max_epochs, monitor
):
    pass


def _train(train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle):
    pass


def _valid(valid_generator_spatial, valid_generator_temporal, approximator, metrics):
    pass
