import torch
from .operators import grad


def _l1_norm(residual, funcs, coords):
    return torch.abs(residual).mean()


def _l2_norm(residual, funcs, coords):
    return (residual ** 2).mean()


def _infinity_norm(residual, funcs, coords):
    return residual.abs().max(dim=1)[0].mean()


def _h1_norm(residual, funcs, coords):
    g = grad(residual, *coords)
    rg = torch.cat([residual, *g], dim=1)
    return (rg ** 2).mean()


def _h1_semi_norm(residual, funcs, coords):
    g = grad(residual, *coords)
    g = torch.cat(g, dim=1)
    return (g ** 2).mean()


def _hybrid(residuals, funcs, coords):
    l2_loss = _l2_norm(residuals, funcs, coords)
    h1_semi_loss = _h1_semi_norm(residuals, funcs, coords)
    mu = torch.tanh(100*l2_loss)
    return (1-mu)*l2_loss + mu*h1_semi_loss

_losses = {
    'l1': _l1_norm,
    'l2': _l2_norm,
    'infinity': _infinity_norm,
    'h1': _h1_norm,
    'h1 semi': _h1_semi_norm,
    'hybrid': _hybrid,
}
