import abc

import torch
import torch.optim as optim
import torch.nn    as nn

import numpy             as np
import matplotlib.pyplot as plt

from .networks import FCNN

class Condition:
    def __init__(self):
        pass
    @abc.abstractmethod
    def enforce(self, t, x):
        pass

class InitialValueProblem(Condition):
    """
    A initial value problem: 
    x (t=t_0) = x_0
    x'(t=t_0) = x_0_prime
    """
    def __init__(self, t_0, x_0, x_0_prime=None):
        self.t_0, self.x_0, self.x_0_prime = t_0, x_0, x_0_prime
    def enforce(self, t, x):
        if self.x_0_prime: 
            return self.x_0 + (1-torch.exp(-t+self.t_0))*self.x_0_prime + ( (1-torch.exp(-t+self.t_0))**2 )*x
        else:
            return torch.exp(-t+self.t_0)*self.x_0 + (1-torch.exp(-t+self.t_0))*x
        
class TwoPointDirichletBoundaryCondition(Condition):
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
        self.ax2.plot(loss_history)
        self.ax2.set_title('loss during training')
        self.ax2.set_ylabel('loss')
        self.ax2.set_xlabel('epochs')
        self.ax2.set_yscale('log')

        self.fig.canvas.draw()

def solve_system(ode_system, conditions, t_min, t_max,
          nets=None, example_generator=None, optimizer=None, criterion=None, batch_size=16, 
          max_epochs=100000, tol=1e-4,
          monitor=None):
    """
    Train a neural network to solve an ODE.
    
    :param ode_system: 
    ODE system as a list of functions. If the ODE system is F_i(x, y, ... t) = 0
    for i = 0, 1, ..., n-1 where x, y, ... are dependent variables and t is the 
    independent variable, then ode_system should be a list where the ith element
    is a function the maps (x, y, ... t) to F_i(x, y, ... t).
    :param conditions: 
    the initial value/boundary conditions as a list of Condition instance. They
    should be in an order such that the first condition constraints the first 
    variable in F_i's (see above) signature. The second the second, and so on.
    :param nets: 
    the networds to be used as a list of torch.nn.Module instances. They should
    be ina na order such that the first network will be used to solve the first
    variable in F_i's (see above) signature. The second the second, and so on.
    :param t_min: lower bound of the domain (t) on which the ode is solved
    :param t_max: upper bound of the domain (t) on which the ode is solved
    :param example_generator: a ExampleGenerator instance
    :param optimizer: a optimizer from torch.optim
    :param criterion: a loss function from torch.nn
    :param batch_size: the size of the minibatch
    :param max_epochs: the maximum number of epochs
    :param tol: the training stops if the loss is lower than this value
    :param monitor: a Monitor instance
    """
    
    # default values
    n_dependent_vars = len(conditions)
    if not nets: 
        nets = [FCNN() for _ in range(n_dependent_vars)]
    if not example_generator: 
        example_generator = ExampleGenerator(32, t_min, t_max, method='equally-spaced-noisy')
    if not optimizer:
        all_parameters = []
        for net in nets: all_parameters += list(net.parameters())
        optimizer = optim.Adam(all_parameters, lr=0.001)
    if not criterion:
        criterion = nn.MSELoss()
    
    n_examples = example_generator.size
    if n_examples%batch_size != 0:
        raise RuntimeError('Please choose a batch_size such that it is a factor of the size of the training set.')
    n_batches = n_examples//batch_size
    zeros = torch.zeros(batch_size)
    
    loss_history = []
    
    for epoch in range(max_epochs):
        loss_epoch = 0.0

        examples = example_generator.get_examples()
        ts_batches = examples.reshape((n_batches, batch_size, 1))
        for ts in ts_batches:     

            # the dependent variables
            vs = []
            for i in range(n_dependent_vars):
                v_i = nets[i](ts)
                if conditions[i]: v_i = conditions[i].enforce(ts, v_i)
                vs.append(v_i)
            
            Fvts = ode_system(*vs, ts)
            loss = 0.0
            for Fvt in Fvts: loss += criterion(Fvt, zeros)
            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_history.append(loss_epoch/n_batches)
        if loss_history[-1] < tol: break
        
        if monitor and epoch%monitor.check_every == 0:
            monitor.check(nets, ode_system, conditions, loss_history)
            
        def solution(ts):
            if not isinstance(ts, torch.Tensor): ts = torch.tensor([ts], dtype=torch.float32)
            ts = ts.reshape(-1, 1)
            results = []
            for i in range(len(conditions)):
                xs = nets[i](ts)
                xs = conditions[i].enforce(ts, xs)
                results.append( xs.detach().numpy().flatten() )
            return results
    if loss_history[-1] > tol:
        print('The solution has not converged.')
        
    return solution, loss_history