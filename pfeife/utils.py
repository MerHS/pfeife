import logging
import torch
from torch.utils._pytree import tree_map

_logger = logging.getLogger("pfeife")


def get_logger():
    return _logger


def to_device(value, device):
    def _map_device(value):
        if torch.is_tensor(value):
            return value.to(device)
        else:
            return value

    return tree_map(_map_device, value)
