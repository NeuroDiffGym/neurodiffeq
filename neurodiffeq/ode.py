import abc

import torch
import torch.optim as optim
import torch.nn    as nn

import numpy             as np
import matplotlib.pyplot as plt

from .networks import FCNN

class IVP:
    """
    A initial value problem: 
    x (t=t_0) = x_0
    x'(t=t_0) = x_0_prime
    """
    def __init__(self, t_0, x_0, x_0_prime=None):
        self.t_0, self.x_0, self.x_0_prime = t_0, x_0, x_0_prime
    def enforce(self, t, x):
        if self.x_0_prime:
            return self.x_0 + (t-self.t_0)*self.x_0_prime + ( (1-torch.exp(-t+self.t_0))**2 )*x
        else:
            return self.x_0 + (1-torch.exp(-t+self.t_0))*x

class DirichletBVP:
    """
    A two point Dirichlet boundary condition: 
    x(t=t_0) = x_0
    x(t=t_0) = x_1
    """
    def __init__(self, t_0, x_0, t_1, x_1):
        self.t_0, self.x_0, self.t_1, self.x_1 = t_0, x_0, t_1, x_1
    def enforce(self, t, x):
        t_tilde = (t-self.t_0) / (self.t_1-self.t_0)
        return self.x_0*(1-t_tilde) + self.x_1*t_tilde + (1-torch.exp((1-t_tilde)*t_tilde))*x

class ExampleGenerator:
    def __init__(self, size, t_min=0.0, t_max=1.0, method='uniform'):
        self.size = size
        self.t_min, self.t_max = t_min, t_max
        if   method=='uniform':
            self.examples = torch.zeros(self.size, requires_grad=True)
            self.get_examples = lambda: self.examples + torch.rand(self.size)*(self.t_max-self.t_min) + self.t_min
        elif method=='equally-spaced':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.get_examples = lambda: self.examples
        elif method=='equally-spaced-noisy':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.noise_mean = torch.zeros(self.size)
            self.noise_std  = torch.ones(self.size) * ( (t_max-t_min)/size ) / 4.0
            self.get_examples = lambda: self.examples + torch.normal(mean=self.noise_mean, std=self.noise_std)
        else:
            raise ValueError(f'Unknown method: {method}')

class Monitor:
    def __init__(self, t_min, t_max, check_every=100):
        self.check_every = check_every
        self.fig = plt.figure(figsize=(20, 8))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        # input for plotting
        self.ts_plt =    np.linspace(t_min, t_max, 100)
        # input for neural network
        self.ts_ann = torch.linspace(t_min, t_max, 100, requires_grad=True).reshape((-1, 1, 1))

    def check(self, nets, ode_system, conditions, loss_history):
        n_dependent = len(conditions)

        vs = []
        for i in range(n_dependent):
            v_i = nets[i](self.ts_ann)
            if conditions[i]: v_i = conditions[i].enforce(self.ts_ann, v_i)
            vs.append(v_i.detach().numpy().flatten())

        self.ax1.clear()
        for i in range(n_dependent):
            self.ax1.plot(self.ts_plt, vs[i], label=f'variable {i}')
        self.ax1.legend()
        self.ax1.set_title('solutions')

        self.ax2.clear()
        self.ax2.plot(loss_history['train'], label='training loss')
        self.ax2.plot(loss_history['valid'], label='validation loss')
        self.ax2.set_title('loss during training')
        self.ax2.set_ylabel('loss')
        self.ax2.set_xlabel('epochs')
        self.ax2.set_yscale('log')
        self.ax2.legend()

        self.fig.canvas.draw()

def solve(ode, condition, t_min, t_max,
          net=None, train_generator=None, shuffle=True, valid_generator=None,
          optimizer=None, criterion=None, batch_size=16,
          max_epochs=1000,
          monitor=None, return_internal=False):
    """
    Train a neural network to solve an ODE.
    
    :param ode: 
    The ODE to solve. If the ODE is F(x, t) = 0 where x is the dependent 
    variable and t is the independent variable,It should be a function the maps
    (x, t) to F(x, t).
    :param condition: 
    the initial value/boundary condition as Condition instance.
    :param net: 
    the networks to be used as a torch.nn.Module instance. 
    :param t_min: lower bound of the domain (t) on which the ODE is solved
    :param t_max: upper bound of the domain (t) on which the ODE is solved
    :param train_generator: an ExampleGenerator instance for training purpose
    :param valid_generator: an ExampleGenerator instance for validation purpose
    :param optimizer: an optimizer from torch.optim
    :param criterion: a loss function from torch.nn
    :param batch_size: the size of the minibatch
    :param max_epochs: the maximum number of epochs
    :param monitor: a Monitor instance
    """
    nets = None if not net else [net]
    returned_tuple = solve_system(
        ode_system=lambda x, t: [ode(x, t)], conditions=[condition],
        t_min=t_min, t_max=t_max, nets=nets,
        train_generator=train_generator, shuffle=shuffle, valid_generator=valid_generator,
        optimizer=optimizer, criterion=criterion, batch_size=batch_size,
        max_epochs=max_epochs, monitor=monitor, return_internal=return_internal
    )

    def solution_wrapped(ts, as_type='tf'):
        return solution(ts, as_type)[0]
    if return_internal:
        solution, loss_history, internal = returned_tuple
        return solution_wrapped, loss_history, internal
    else:
        solution, loss_history = returned_tuple
        return solution_wrapped, loss_history


def solve_system(ode_system, conditions, t_min, t_max,
          nets=None, train_generator=None, shuffle=True, valid_generator=None,
          optimizer=None, criterion=None, batch_size=16,
          max_epochs=1000,
          monitor=None, return_internal=False):
    """
    Train a neural network to solve an ODE.
    
    :param ode_system: 
    ODE system as a list of functions. If the ODE system is F_i(x, y, ... t) = 0
    for i = 0, 1, ..., n-1 where x, y, ... are dependent variables and t is the 
    independent variable, then ode_system should map (x, y, ... t) to a list where 
    the ith entry is F_i(x, y, ... t).
    :param conditions: 
    the initial value/boundary conditions as a list of Condition instance. They
    should be in an order such that the first condition constraints the first 
    variable in F_i's (see above) signature. The second the second, and so on.
    :param nets: 
    the networks to be used as a list of torch.nn.Module instances. They should
    be ina na order such that the first network will be used to solve the first
    variable in F_i's (see above) signature. The second the second, and so on.
    :param t_min: lower bound of the domain (t) on which the ODE system is solved
    :param t_max: upper bound of the domain (t) on which the ODE system is solved
    :param train_generator: an ExampleGenerator instance for training purpose
    :param valid_generator: an ExampleGenerator instance for validation purpose
    :param optimizer: an optimizer from torch.optim
    :param criterion: a loss function from torch.nn
    :param batch_size: the size of the minibatch
    :param max_epochs: the maximum number of epochs
    :param monitor: a Monitor instance
    """

    # default values
    n_dependent_vars = len(conditions)
    if not nets:
        nets = [FCNN() for _ in range(n_dependent_vars)]
    if not train_generator:
        train_generator = ExampleGenerator(32, t_min, t_max, method='equally-spaced-noisy')
    if not valid_generator:
        valid_generator = ExampleGenerator(32, t_min, t_max, method='equally-spaced')
    if not optimizer:
        all_parameters = []
        for net in nets: all_parameters += list(net.parameters())
        optimizer = optim.Adam(all_parameters, lr=0.001)
    if not criterion:
        criterion = nn.MSELoss()

    if return_internal:
        internal = {
            'nets': nets,
            'conditions': conditions,
            'train_generator': train_generator,
            'valid_generator': valid_generator,
            'optimizer': optimizer,
            'criterion': criterion
        }

    n_examples_train = train_generator.size
    n_examples_valid = valid_generator.size
    train_zeros = torch.zeros(batch_size)
    valid_zeros = torch.zeros(n_examples_valid)

    loss_history = {'train': [], 'valid': []}

    for epoch in range(max_epochs):
        train_loss_epoch = 0.0

        train_examples = train_generator.get_examples()
        train_examples = train_examples.reshape(n_examples_train, 1)
        idx = np.random.permutation(n_examples_train) if shuffle else np.arange(n_examples_train)
        batch_start, batch_end = 0, batch_size
        while batch_start < n_examples_train:

            if batch_end >= n_examples_train: batch_end = n_examples_train
            batch_idx = idx[batch_start:batch_end]
            ts = train_examples[batch_idx]

            # the dependent variables
            vs = []
            for i in range(n_dependent_vars):
                v_i = nets[i](ts)
                if conditions[i]: v_i = conditions[i].enforce(ts, v_i)
                vs.append(v_i)

            Fvts = ode_system(*vs, ts)
            loss = 0.0
            for Fvt in Fvts: loss += criterion(Fvt, train_zeros)
            train_loss_epoch += loss.item() * (batch_end-batch_start)/n_examples_train # assume the loss is a mean over all examples

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_start += batch_size
            batch_end   += batch_size

        loss_history['train'].append(train_loss_epoch)

        # calculate the validation loss
        ts = valid_generator.get_examples().reshape(n_examples_valid, 1)
        vs = []
        for i in range(n_dependent_vars):
            v_i = nets[i](ts)
            if conditions[i]: v_i = conditions[i].enforce(ts, v_i)
            vs.append(v_i)
        Fvts = ode_system(*vs, ts)
        valid_loss_epoch = 0.0
        for Fvt in Fvts: valid_loss_epoch += criterion(Fvt, valid_zeros)
        valid_loss_epoch = valid_loss_epoch.item()

        loss_history['valid'].append(valid_loss_epoch)

        if monitor and epoch%monitor.check_every == 0:
            monitor.check(nets, ode_system, conditions, loss_history)

        def solution(ts, as_type='tf'):
            if not isinstance(ts, torch.Tensor): ts = torch.tensor([ts], dtype=torch.float32)
            ts = ts.reshape(-1, 1)
            results = []
            for i in range(len(conditions)):
                xs = nets[i](ts)
                xs = conditions[i].enforce(ts, xs)
                if   as_type == 'tf': results.append(xs)
                elif as_type == 'np': results.append(xs.detach().numpy().flatten())
                else:
                    raise ValueError("The valid return types are 'tf' and 'np'.")
            return results

    if return_internal:
        return solution, loss_history, internal
    else:
        return solution, loss_history
