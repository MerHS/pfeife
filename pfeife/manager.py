from typing import List, Any, Union
import copy

import torch
import torch.nn as nn
import torch.fx as fx
import torch.optim as optim
from torch.fx import Node
import torch.fx.traceback as fx_traceback
import torch.distributed.rpc as rpc

import torch._dynamo as dynamo
from torch._subclasses import UnsupportedFakeTensorException
from torch.utils._pytree import tree_flatten
from torch.nn.modules.loss import _Loss


from .graph_split import ParamSplit
from .utils import get_logger
from .batch import send_value, split_args_kwargs
from .trace import graph_break, StepTrace, RPCStage, TraceInterpreter
from .scheduler import SchedGPipe
from .loss import LossWrapper
from .pipe_graph import PipeGraph
from .option import PipeOption

log = get_logger()


class PipeGraphRunner(nn.Module):
    def __init__(self, manager: "PipeManager", gm: fx.GraphModule):
        super().__init__()
        self.manager = manager
        self.gm = gm
        self.graph = PipeGraph(self.manager.rpc_workers)


class PipeManager:
    def __init__(
        self,
        model,
        loss_fn: Union[_Loss, LossWrapper],
        option: PipeOption = PipeOption(),
    ):
        super().__init__()
        self.orig_model = model
        self.loss_fn = loss_fn
        self.set_option(option)

        self.rpc_workers = []
        self.last_batch_no = 0

        self._compile()

    def set_option(self, option: PipeOption):
        self.stage_cnt = option.stage_cnt
        self.batch_cnt = option.batch_cnt
        self.device_cnt = option.device_cnt

        # TODO: split by other strategies
        self.splitter = ParamSplit(self.stage_cnt)

        # TODO: check this will remove (GC-out) worker objects
        self.rpc_workers = []
        for rank in range(1, self.device_cnt + 1):
            device = f"cuda:{rank-1}"
            worker = rpc.remote(
                f"worker_{rank}",
                RPCStage,
                args=(rank, device, option),
            )
            self.rpc_workers.append(worker)

    def _compile(self):
        def compile_fn(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            qualname_map = {}
            split_gm = self.splitter.split(gm, qualname_map)

            def delete_user_reference(node, user, delete_node=True):
                assert len(user.kwargs) == 0
                use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
                assert len(use_idxs) == 1
                args_copy = list(user.args)
                args_copy.pop(use_idxs[0])
                user.args = tuple(args_copy)
                if delete_node:
                    node.graph.erase_node(node)

                return use_idxs[0]

            def move_param_to_callee(
                root, node, callee_name, param_val, use_idx, is_buffer
            ):
                assert isinstance(param_val, torch.Tensor), (
                    f"Expected '{node.target}' to be {torch.Tensor} but got {type(param_val)}."
                    + (
                        f" It might happen if module '{node.target}' was passed to some 'leaf function'"
                        f"(see https://pytorch.org/docs/stable/fx.html#pippy.fx.wrap). Please inspect "
                        f"usages of '{node.target}' in the traced graph."
                        if isinstance(param_val, torch.nn.Module)
                        else ""
                    )
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

            to_delete = list()  # a list of nodes for deferral deletion

            for node in split_gm.graph.nodes:
                if node.op == "get_attr" and len(node.users) == 1:
                    user = list(node.users)[0]
                    assert user.op == "call_module"
                    use_idx = delete_user_reference(node, user)

                    # Move parameter into submodule and replace PH with a get_attr
                    atoms = node.target.split(".")
                    mod_itr = split_gm
                    for atom in atoms[:-1]:
                        mod_itr = getattr(mod_itr, atom)
                    param_val = getattr(mod_itr, atoms[-1])
                    is_buffer = atoms[-1] in mod_itr._buffers

                    move_param_to_callee(
                        split_gm, node, user.target, param_val, use_idx, is_buffer
                    )

                    to_delete.append((mod_itr, atoms))

            # deferral deletion
            for (mod_itr, atoms) in to_delete:
                delattr(mod_itr, atoms[-1])

            split_gm.recompile()

            log.info(
                "\n---final graph---\n" + str(split_gm.graph) + "\n---------------\n"
            )
            print(split_gm.graph)

            runner = PipeGraphRunner(self, split_gm)

            return runner

        dynamo_ctx = dynamo.optimize(compile_fn)
        self.opt_model = dynamo_ctx(self.orig_model)

    def train(self):
        self.opt_model.train()

    def eval(self):
        self.opt_model.eval()
        # TDOO: rpc send eval

    def optimizer_zero_grad(self):
        for stage in self.rpc_stages:
            stage.rpc_sync().optimizer_zero_grad()

    def optimizer_step(self):
        for stage in self.rpc_stages:
            stage.rpc_sync().optimizer_step()

    def run(self, target, *args, **kwargs):
        """
        1. optimizer.zero_grad()
        2. forward & backward & optimizer step
        3. return loss
        """

        for stage in self.rpc_stages:
            stage.rpc_sync().reset_cache()

        self.reset_batch_no()
        self.optimizer_zero_grad()

        # TODO: should we move args to gpu 0 before splitting?
        args = send_value(args, "cuda:0")
        kwargs = send_value(kwargs, "cuda:0")

        # TODO: check last device
        target = send_value(target, f"cuda:{self.pipe_split - 1}")

        args, kwargs = split_args_kwargs(args, kwargs, self.batch_split)
        targets, _ = split_args_kwargs([target], dict(), self.batch_split)

        traces: List[StepTrace] = []
        for micro_args, micro_kwargs in zip(args, kwargs):
            traces.append(self.opt_model(*micro_args, **micro_kwargs))

        # returns losses
        losses = self.sched.exec(traces, targets, self.loss_fn)
        loss = sum([l for l in losses if torch.is_tensor(l)])

        self.optimizer_step()

        return loss
