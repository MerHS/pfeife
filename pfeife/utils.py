import logging

import torch
import torch.fx as fx
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
        if torch.is_tensor(value) or isinstance(value, torch.nn.Module):
            return value.to(device)
        else:
            return value

    return tree_map(_map_device, value)


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


def move_param_to_callee(
    root, node, callee_name, param_val, qualname_map, use_idx, is_buffer
):
    assert isinstance(
        param_val, torch.Tensor
    ), f"Expected '{node.target}' to be {torch.Tensor} but got {type(param_val)}." + (
        f" It might happen if module '{node.target}' was passed to some 'leaf function'"
        f"(see https://pytorch.org/docs/stable/fx.html#pippy.fx.wrap). Please inspect "
        f"usages of '{node.target}' in the traced graph."
        if isinstance(param_val, torch.nn.Module)
        else ""
    )
    callee = root.get_submodule(callee_name)
    new_param_name = f"moved_{node.target.replace('.', '_')}"
    assert not hasattr(
        callee, new_param_name
    ), f"Module {callee_name} already has a parameter named {new_param_name}"
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
