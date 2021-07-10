import importlib

from torch import optim

def get_optim(params, optimizer_config):
    name = optimizer_config.pop('name', 'Adam')
    optimizer = getattr(optim, name)
    return optimizer(params=params, **optimizer_config)