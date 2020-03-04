# Consider these convention:
# For user facing part, use 1-d tensor (when representing a collection of 2-d points, use 2 1-d tensors).
# For non-user facing part, avoid reshaping and use 1-d tensor as much as possible.
# In function signatures, let u comes before x, let x comes before t
# Use x and t for distinguish values for x or t; Use xx and tt when corresponding entries of xx and tt are
# supposed to be paired to represent a point.
from abc import ABC, abstractmethod
import torch


class Approximator(ABC):
    @abstractmethod
    def __call__(self, time_dimension, *spatial_dimensions):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def loss(self):
        pass


class SingleNetworkApproximator1DSpatialTemporal(Approximator):
    def __init__(self, single_network, pde, initial_condition, boundary_conditions):
        self.single_network = single_network
        self.pde = pde
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions

    def __call__(self, xx, tt):
        xx = torch.unsqueeze(xx, dim=1)
        tt = torch.unsqueeze(tt, dim=1)
        xt = torch.cat((xx, tt), dim=1)
        uu = self.initial_condition.u0(xx) + (1 - torch.exp(-tt)) * self.single_network(xt)
        return torch.squeeze(uu)

    def parameters(self):
        return self.single_network.parameters()

    def loss(self, x, t):
        xt = torch.cartesian_prod(x, t)
        xx = torch.squeeze(xt[:, 0])
        xx.requires_grad = True
        tt = torch.squeeze(xt[:, 1])
        tt.requires_grad = True
        uu = self.__call__(xx, tt)

        equation_mse = torch.mean(self.pde(uu, xx, tt)**2)

        boundary_mse = sum(self._boundary_mse(t, bc) for bc in self.boundary_conditions)

        return equation_mse + boundary_mse

    def _boundary_mse(self, t, bc):
        x = next(bc.points_generator)

        xt = torch.cartesian_prod(x, t)
        xx = torch.squeeze(xt[:, 0])
        xx.requires_grad = True
        tt = torch.squeeze(xt[:, 1])
        uu = self.__call__(xx, tt)
        return torch.mean(bc.form(uu, xx, tt)**2)


class FirstOrderInitialCondition:
    def __init__(self, u0):
        self.u0 = u0


class BoundaryCondition:
    def __init__(self, form, points_generator):
        self.form = form
        self.points_generator = points_generator


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
