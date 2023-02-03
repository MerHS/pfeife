import logging
import torch
from torch.utils._pytree import tree_map

logger = None


def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger("pfeife")
        logging.basicConfig()
    return logger


def set_logger_level(level):
    logger = get_logger()
    logger.level = level


def to_device(value, device):
    def _map_device(value):
        if torch.is_tensor(value):
            return value.to(device)
        else:
            return value

    return tree_map(_map_device, value)
