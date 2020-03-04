# Consider these convention:
# For user facing part, use 1-d tensor (when representing a collection of 2-d points, use 2 1-d tensors).
# For non-user facing part, avoid reshaping and use 1-d tensor as much as possible.
# In function signatures, let u comes before x, let x comes before t
# Use x and t for distinguish values for x or t; Use xx and tt when corresponding entries of xx and tt are
# supposed to be paired to represent a point.
from abc import ABC, abstractmethod
import torch
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class Approximator(ABC):
    @abstractmethod
    def __call__(self, time_dimension, *spatial_dimensions):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def parameters(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def loss(self):
        raise NotImplementedError  # pragma: no cover


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
        self.using_non_gui_backend = matplotlib.get_backend() is 'agg'

        check_on_xt_tensor = torch.cartesian_prod(check_on_x, check_on_t)
        self.xx_tensor = torch.squeeze(check_on_xt_tensor[:, 0])
        self.xx_tensor.requires_grad = True
        self.tt_tensor = torch.squeeze(check_on_xt_tensor[:, 1])
        self.tt_tensor.requires_grad = True
        self.x_array = check_on_x.clone().detach().numpy()
        self.t_array = check_on_t.clone().detach().numpy()
        self.check_every = check_every
        self.t_color = torch.linspace(0, 1, len(check_on_t)).detach().numpy()

        self.fig = plt.figure(figsize=(15, 4))
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

    def check(self, approximator, history):
        uu_array = approximator(self.xx_tensor, self.tt_tensor).detach().numpy()

        self.ax1.clear()
        for i, t_c in enumerate(zip(self.t_array, self.t_color)):
            u_t = uu_array[i::len(self.t_array)]
            t, c = t_c
            t = float(t)
            c = cm.viridis(c)

            self.ax1.plot(self.x_array, u_t, color=c, label=f't = {t:.2E}')
        self.ax1.legend()
        self.ax1.set_title('approximation')

        self.ax2.clear()
        self.ax2.plot(history['train_loss'], label='training loss')
        self.ax2.plot(history['valid_loss'], label='validation loss')
        self.ax2.set_title('loss during training')
        self.ax2.set_ylabel('loss')
        self.ax2.set_xlabel('epochs')
        self.ax2.set_yscale('log')
        self.ax2.legend()

        self.ax3.clear()
        for metric_name, metric_values in history.items():
            if metric_name == 'train_loss' or metric_name == 'valid_loss':
                continue
            self.ax3.plot(metric_values, label=metric_name)
        self.ax3.set_title('metrics during training')
        self.ax3.set_ylabel('metrics')
        self.ax3.set_xlabel('epochs')
        self.ax3.set_yscale('log')
        # if there's not custom metrics, then there won't be any labels in this axis
        if len(history) > 2:
            self.ax3.legend()

        self.fig.canvas.draw()
        if not self.using_non_gui_backend:
            plt.pause(0.05)   # pragma: no cover (we are not using gui backend for testing)



def solve_1dspatial_temporal(
        pde, initial_condition, boundary_conditions,
        train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
        batch_size, max_epochs, monitor
):
    raise NotImplementedError  # pragma: no cover


def _train(train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle):
    raise NotImplementedError  # pragma: no cover


def _valid(valid_generator_spatial, valid_generator_temporal, approximator, metrics):
    raise NotImplementedError  # pragma: no cover
