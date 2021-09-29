from abc import ABC, abstractmethod
import torch


class NumericalSolver(ABC):
    @abstractmethod
    def solve(self, func, u0, t0, tn, n_steps):
        pass

    @abstractmethod
    def step(self, func, u, t, h):
        pass


class Euler(NumericalSolver):
    order = 1

    def solve(self, func, u0, t0, tn, n_steps, hypernet=None):
        ts = torch.linspace(t0, tn, n_steps + 1)
        if isinstance(u0, (float, int)):
            u0 = (u0,)
        if isinstance(u0, (list, tuple)):
            u0 = torch.tensor(u0)
        us = [u0]
        h = (tn - t0) / n_steps
        for t in ts[:-1]:
            u_old = us[-1]
            u_new = u_old + h * torch.tensor(self.step(func, u_old, t, h))
            if hypernet is not None:
                u_new += h ** 2 * hypernet(torch.cat([t.reshape(1, 1), u_old.reshape(1, -1)], dim=1)).flatten()
            us.append(u_new)

        us = torch.stack(us, dim=0)
        ans = [ts]
        for j in range(us.shape[1]):
            ans.append(us[:, j])

        return ans

    def step(self, func, u, t, h):
        return func(*u, t)
