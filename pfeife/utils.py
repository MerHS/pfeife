import logging
import math
import numpy as np
import random

import torch
import torch.fx as fx
import torch.backends.cudnn as cudnn
from torch.utils._pytree import tree_map, tree_flatten

logger = None


def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger("pfeife")
        logging.basicConfig(
            format="(%(asctime)s.%(msecs)03d) %(message)s", datefmt="%I:%M:%S"
        )
    return logger


def fmem(mem):
    return f"{mem / 1024 / 1024:.2f}MB"


def set_logger_level(level):
    logger = get_logger()
    logger.level = level


def to_device(value, device):
    def _map_device(value):
        if torch.is_tensor(value) or isinstance(value, torch.nn.Module):
            value = value.to(device)
        return value

    return tree_map(_map_device, value)


def tree_trav(fn, pytree):
    flat_args, _ = tree_flatten(pytree)
    for i in flat_args:
        fn(i)


def tree_filter(fn, pytree):
    value = []
    flat_args, _ = tree_flatten(pytree)
    for i in flat_args:
        v = fn(i)
        if v is not None:
            value.append(v)
    return value


def tree_filter_tensor(pytree):
    value = []
    flat_args, _ = tree_flatten(pytree)
    for i in flat_args:
        if isinstance(i, torch.Tensor):
            value.append(i)
    return value


def fetch_attr(module, target: str):
    target_atoms = target.split(".")
    attr_itr = module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def get_submodules(gm: fx.GraphModule):
    mods = []
    for node in gm.graph.nodes:
        if node.op == "call_module":
            mods.append(fetch_attr(gm, node.target))
    return mods


def list_params(attr):
    if isinstance(attr, torch.nn.Module):
        return list(attr.parameters())

    value = []
    flat_values, _ = tree_flatten(attr)
    for v in flat_values:
        if isinstance(v, torch.nn.Parameter):
            value.append(v)
    return value


def move_param_to_callee(
    root, node, callee_name, param_val, qualname_map, use_idx, is_buffer
):
    assert isinstance(param_val, torch.Tensor)
    callee = root.get_submodule(callee_name)
    new_param_name = f"moved_{node.target.replace('.', '_')}"

    assert not hasattr(callee, new_param_name)
    if is_buffer:
        callee.register_buffer(new_param_name, param_val)
    else:
        setattr(callee, new_param_name, param_val)

    # Update qualname mapping
    # New qualname will have submodule prefix
    new_qualname = f"{callee_name}.{new_param_name}"
    if node.target in qualname_map:
        # Just in case the target name is already in the qualname_map
        # returned by split_module() -- we update the mapping using the
        # new name as a new key
        qualname_map[new_qualname] = qualname_map.pop(node.target)
    else:
        qualname_map[new_qualname] = node.target

    ph_counter = 0
    for sn in callee.graph.nodes:
        if sn.op == "placeholder":
            if ph_counter == use_idx:
                with callee.graph.inserting_before(sn):
                    get_attr = callee.graph.get_attr(new_param_name)
                    sn.replace_all_uses_with(get_attr)
                    callee.graph.erase_node(sn)
            ph_counter += 1
    callee.graph.lint()
    callee.recompile()

    return get_attr


def module_set_attr(module, target: str, attr):
    target_atoms = target.split(".")
    attr_itr = module
    for atom in target_atoms[:-1]:
        attr_itr = getattr(attr_itr, atom)
    setattr(attr_itr, target_atoms[-1], attr)


def to_device_graphmodule(gm: fx.GraphModule, device):
    device = torch.device(device)
    gm = gm.to(device)

    for node in gm.graph.nodes:
        if len(node.kwargs) > 0:
            new_kwargs = {}
            for k, v in node.kwargs.items():
                new_kwargs[k] = device if isinstance(v, torch.device) else v
            node.kwargs = new_kwargs

    gm.recompile()
    return gm


class Recorder:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sq_sum = 0

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def stddev(self):
        mean = self.mean
        return math.sqrt(
            (self.sq_sum - 2 * mean * self.sum + self.count * self.mean * self.mean)
            / self.count
        )

    def append(self, x):
        self.count += 1
        self.sum += x
        self.sq_sum += x * x

    def update(self, lst):
        self.count += len(lst)
        for x in lst:
            self.sum += x
            self.sq_sum += x * x


class MinMaxRecorder:
    def __init__(self):
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")

    def append(self, x):
        self.count += 1
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def update(self, lst):
        self.count += len(lst)
        for x in lst:
            self.min = min(self.min, x)
            self.max = max(self.max, x)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
