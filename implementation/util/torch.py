import importlib

from torch import optim

def get_optim(params, name, optimizer_params):
    optimizer = getattr(optim, name)
    return optimizer(params=params, **optimizer_params)