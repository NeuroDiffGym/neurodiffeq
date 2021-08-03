import torch


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
    pass

