# Consider these convention:
# For user facing part, use 1-d tensor (when representing a collection of 2-d points, use 2 1-d tensors).
# For non-user facing part, avoid reshaping and use 1-d tensor as much as possible.
# In function signatures, let u comes before x, let x comes before t
# Use x and t for distinguish values for x or t; Use xx and tt when corresponding entries of xx and tt are
# supposed to be paired to represent a point. ([xx, tt] is often the Cartesian product of x and t)
from abc import ABC, abstractmethod
import torch
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def _cartesian_prod_dims(x, t, x_grad=True, t_grad=True):
    xt = torch.cartesian_prod(x, t)
    xx = torch.squeeze(xt[:, 0])
    xx.requires_grad = x_grad
    tt = torch.squeeze(xt[:, 1])
    tt.requires_grad = t_grad
    return xx, tt

class Approximator(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def parameters(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def calculate_loss(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def calculate_metrics(self):
        raise NotImplementedError  # pragma: no cover


class SingleNetworkApproximator1DSpatialTemporal(Approximator):
    def __init__(self, single_network, pde, initial_condition, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, tt):
        xx = torch.unsqueeze(xx, dim=1)
        tt = torch.unsqueeze(tt, dim=1)
        xt = torch.cat((xx, tt), dim=1)
        uu = self.initial_condition.u0(xx) + (1 - torch.exp(-tt)) * self.single_network(xt)
        return torch.squeeze(uu)

    def parameters(self):
        return self.single_network.parameters()

    # AHHHHHHHHHHHHHHHH WHY IS THIS FUNCTION SIGNATURE SO UGLY
    # Perhaps ugliness is an essential part of human condition
    def calculate_loss(self, xx, tt, x, t):
        uu = self.__call__(xx, tt)

        equation_mse = torch.mean(self.pde(uu, xx, tt)**2)

        boundary_mse = self.boundary_strictness * sum(self._boundary_mse(t, bc) for bc in self.boundary_conditions)

        return equation_mse + boundary_mse

    def _boundary_mse(self, t, bc):
        x = next(bc.points_generator)

        xx, tt = _cartesian_prod_dims(x, t, x_grad=True, t_grad=False)
        uu = self.__call__(xx, tt)
        return torch.mean(bc.form(uu, xx, tt)**2)

    def calculate_metrics(self, xx, tt, x, t, metrics):
        uu = self.__call__(xx, tt)

        return {
            metric_name: metric_func(uu, xx, tt)
            for metric_name, metric_func in metrics.items()
        }


class SingleNetworkApproximator2DSpatialTemporal(Approximator):
    def __init__(self, single_network, pde, initial_condition, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, yy, tt):
        xx = torch.unsqueeze(xx, dim=1)
        yy = torch.unsqueeze(yy, dim=1)
        tt = torch.unsqueeze(tt, dim=1)
        xyt = torch.cat((xx, yy, tt), dim=1)
        uu = self.initial_condition.u0(xx, yy) + (1 - torch.exp(-tt)) * self.single_network(xyt)
        return torch.squeeze(uu)

    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy, tt, x, y, t):
        uu = self.__call__(xx, yy, tt)

        equation_mse = torch.mean(self.pde(uu, xx, yy, tt)**2)

        boundary_mse = self.boundary_strictness * sum(self._boundary_mse(t, bc) for bc in self.boundary_conditions)

        return equation_mse + boundary_mse

    def _boundary_mse(self, t, bc):
        x, y = next(bc.points_generator)

        xx, tt = _cartesian_prod_dims(x, t, x_grad=True, t_grad=False)
        yy, tt = _cartesian_prod_dims(y, t, x_grad=True, t_grad=False)
        uu = self.__call__(xx, yy, tt)
        return torch.mean(bc.form(uu, xx, yy, tt)**2)

    def calculate_metrics(self, xx, yy, tt, x, y, t, metrics):
        uu = self.__call__(xx, yy, tt)

        return {
            metric_name: metric_func(uu, xx, yy, tt)
            for metric_name, metric_func in metrics.items()
        }


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


def generator_2dspatial_segment(size, start, end, random=True):
    x1, y1 = start
    x2, y2 = end
    step = 1./size
    center = torch.linspace(0. + 0.5*step, 1. - 0.5*step, size)
    noise_lo = -step*0.5
    while True:
        if random:
            noise = step*torch.rand(size) + noise_lo
            center = center + noise
        yield x1 + (x2-x1)*center, y1 + (y2-y1)*center


def generator_2dspatial_rectangle(size, x_min, x_max, y_min, y_max, random=True):
    x_size, y_size = size
    x_generator = generator_1dspatial(x_size, x_min, x_max, random)
    y_generator = generator_1dspatial(y_size, y_min, y_max, random)
    while True:
        x = next(x_generator)
        y = next(y_generator)
        xy = torch.cartesian_prod(x, y)
        xx = torch.squeeze(xy[:, 0])
        yy = torch.squeeze(xy[:, 1])
        yield xx, yy


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

        self.xx_tensor, self.tt_tensor = _cartesian_prod_dims(check_on_x, check_on_t)
        self.x_array = check_on_x.clone().detach().numpy()
        self.t_array = check_on_t.clone().detach().numpy()
        self.check_every = check_every
        self.t_color = torch.linspace(0, 1, len(check_on_t)).detach().numpy()

        self.fig = plt.figure(figsize=(30, 8))
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


class Monitor2DSpatialTemporal:
    pass


def _solve_1dspatial_temporal(
    train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor
):
    return _solve_spatial_temporal(
        train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
        approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,
        train_routine=_train_1dspatial_temporal, valid_routine=_valid_1dspatial_temporal
    )


def _solve_spatial_temporal(
    train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,
    train_routine, valid_routine
):
    history = {'train_loss': [], 'valid_loss': []}
    for metric_name, _ in metrics.items():
        history['train_' + metric_name] = []
        history['valid_' + metric_name] = []

    for epoch in range(max_epochs):
        train_epoch_loss, train_epoch_metrics = train_routine(
            train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size
        )
        history['train_loss'].append(train_epoch_loss)
        for metric_name, metric_value in train_epoch_metrics.items():
            history['train_' + metric_name].append(metric_value)

        valid_epoch_loss, valid_epoch_metrics = valid_routine(
            valid_generator_spatial, valid_generator_temporal, approximator, metrics
        )
        history['valid_loss'].append(valid_epoch_loss)
        for metric_name, metric_value in valid_epoch_metrics.items():
            history['valid_' + metric_name].append(metric_value)

        if monitor and epoch % monitor.check_every == 0:
            monitor.check(approximator, history)

    return approximator, history


def _train_1dspatial_temporal(train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size):
    x = next(train_generator_spatial)
    t = next(train_generator_temporal)
    xx, tt = _cartesian_prod_dims(x, t)
    training_set_size = len(xx)
    idx = torch.randperm(training_set_size) if shuffle else torch.arange(training_set_size)

    batch_start, batch_end = 0, batch_size
    while batch_start < training_set_size:
        if batch_end > training_set_size:
            batch_end = training_set_size
        batch_idx = idx[batch_start:batch_end]
        batch_xx = xx[batch_idx]
        batch_tt = tt[batch_idx]

        batch_loss = approximator.calculate_loss(batch_xx, batch_tt, x, t)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_start += batch_size
        batch_end += batch_size

    epoch_loss = approximator.calculate_loss(xx, tt, x, t).item()

    epoch_metrics = approximator.calculate_metrics(xx, tt, x, t, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


def _valid_1dspatial_temporal(valid_generator_spatial, valid_generator_temporal, approximator, metrics):
    x = next(valid_generator_spatial)
    t = next(valid_generator_temporal)
    xx, tt = _cartesian_prod_dims(x, t)

    epoch_loss = approximator.calculate_loss(xx, tt, x, t).item()

    epoch_metrics = approximator.calculate_metrics(xx, tt, x, t, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics
