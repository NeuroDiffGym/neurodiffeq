import torch
import logging
from .operators import grad

losses_logger = logging.getLogger('neurodiffeq.losses')


def _l1_norm(residual, funcs, coords):
    loss = torch.abs(residual).mean()
    if losses_logger.isEnabledFor(logging.DEBUG):
        losses_logger.debug(f"L1 loss: {loss.item():.6f}, residual stats: mean={residual.mean().item():.6f}, std={residual.std().item():.6f}")
    return loss


def _l2_norm(residual, funcs, coords):
    loss = (residual ** 2).mean()
    if losses_logger.isEnabledFor(logging.DEBUG):
        losses_logger.debug(f"L2 loss: {loss.item():.6f}, residual stats: mean={residual.mean().item():.6f}, std={residual.std().item():.6f}")
    return loss


def _infinity_norm(residual, funcs, coords):
    loss = residual.abs().max(dim=1)[0].mean()
    if losses_logger.isEnabledFor(logging.DEBUG):
        max_residual = residual.abs().max().item()
        losses_logger.debug(f"Infinity loss: {loss.item():.6f}, max residual: {max_residual:.6f}")
    return loss


def _h1_norm(residual, funcs, coords):
    try:
        g = grad(residual, *coords)
        rg = torch.cat([residual, *g], dim=1)
        loss = (rg ** 2).mean()
        if losses_logger.isEnabledFor(logging.DEBUG):
            grad_norm = torch.cat(g, dim=1).norm().item()
            losses_logger.debug(f"H1 loss: {loss.item():.6f}, gradient norm: {grad_norm:.6f}")
        return loss
    except Exception as e:
        losses_logger.error(f"Error computing H1 norm: {e}")
        raise


def _h1_semi_norm(residual, funcs, coords):
    try:
        g = grad(residual, *coords)
        g = torch.cat(g, dim=1)
        loss = (g ** 2).mean()
        if losses_logger.isEnabledFor(logging.DEBUG):
            grad_norm = g.norm().item()
            losses_logger.debug(f"H1 semi loss: {loss.item():.6f}, gradient norm: {grad_norm:.6f}")
        return loss
    except Exception as e:
        losses_logger.error(f"Error computing H1 semi-norm: {e}")
        raise


_losses = {
    'l1': _l1_norm,
    'l2': _l2_norm,
    'infinity': _infinity_norm,
    'h1': _h1_norm,
    'h1 semi': _h1_semi_norm,
}
