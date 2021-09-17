import torch
from ..networks import FCNN
from ..utils import vstack, hstack, split_columns


class DiscreteSolution1D:
    def __init__(self, ts, *us):
        self.ts = ts
        self.us_tuple = torch.stack(us, dim=1)

    def __call__(self, ts):
        ret_u = []
        for t in ts:
            for i in range(len(self.ts) - 1):
                if (self.ts[i] <= t) and (t <= self.ts[i + 1]):
                    u = (self.us_tuple[i + 1] * (t - self.ts[i]) + self.us_tuple[i] * (self.ts[i + 1] - t)) \
                        / (self.ts[i + 1] - self.ts[i])
                    ret_u.append(u)
                    break

        ret_u = torch.stack(ret_u, dim=0)
        return [ret_u[:, j] for j in range(ret_u.shape[1])]


class Hypersolver:
    def __init__(self, func, u0, t0, tn, n_steps, sol, numerical_solver, net=None, optimizer=None):
        self.func = func
        if isinstance(u0, (int, float)):
            self.u0 = torch.tensor([float(u0)])
        elif isinstance(u0, (list, tuple)):
            self.u0 = torch.tensor(u0)
        else:
            raise TypeError(f"u0 must be int, float, list, or tuple, not {type(u0)}")
        self.t0 = t0
        self.tn = tn
        self.n_steps = n_steps
        self.h = (tn - t0) / n_steps
        self.ts = torch.linspace(t0, tn, n_steps + 1)
        self.solution = sol
        self.numerical_solver = numerical_solver
        self.us = torch.stack(self.solution(self.ts), dim=1)
        self.local_epoch = 0
        self._max_local_epoch = 1

        us_no_head = self.us[1:, :]
        us_no_tail = self.us[:-1, :]
        R = us_no_head - us_no_tail - \
            self.h * hstack(self.numerical_solver.step(self.func, split_columns(us_no_tail), self.ts[:-1], self.h))
        self.residual = R / self.h ** (self.numerical_solver.order + 1)

        if net is None:
            self.net = FCNN(n_input_units=len(self.u0) + 1, n_output_units=len(self.u0), hidden_units=(32, 32))
        else:
            self.net = net
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        self.loss_fcn = torch.nn.MSELoss()

        self.metrics_history = {}
        self.metrics_history['train_loss'] = []
        self.metrics_history['valid_loss'] = []

    def fit(self, max_epochs):
        self._max_local_epoch = max_epochs
        for epoch in range(max_epochs):
            self.local_epoch += 1
            input = torch.cat((self.ts.reshape(-1, 1), self.us), dim=1)
            output = self.net(input)
            loss = self.loss_fcn(self.residual, output[1:])

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.metrics_history['train_loss'].append(loss.item())

    @property
    def global_epoch(self):
        return len(self.metrics_history['train_loss'])

    def get_solution(self):
        ret = self.numerical_solver.solve(self.func, self.u0, self.t0, self.tn, self.n_steps, hypernet=self.net)
        return DiscreteSolution1D(*ret)
