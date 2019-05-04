import torch
import torch.autograd as autograd

def diff(x, t, order=1):
    ones = torch.ones_like(t)
    der, = autograd.grad(x, t, create_graph=True, grad_outputs=ones)
    for i in range(1, order):
        der, = autograd.grad(der, t, create_graph=True, grad_outputs=ones)
    return der