from typing import List, Union

import torch
import torch._dynamo as dynamo
import torch.distributed.rpc as rpc
import torch.fx as fx
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.distributed.rpc import PyRRef

from .batch import split_args_kwargs
from .graph_split import ParamSplit
from .loss import LossWrapper
from .option import PipeOption
from .pipe_graph import PipeGraph
from .rpc_worker import RPCWorker
from .scheduler import get_scheduler
from .utils import get_logger

log = get_logger()


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


# TODO: make it a real tensor
class PipeTensor:
    def __init__(self, batch_no, output_token, backward_token=None):
        self.batch_no = batch_no
        self.output_token = output_token
        self.backward_token = backward_token


class PipeGraphRunner(nn.Module):
    def __init__(
        self,
        manager: "PipeManager",
        gm: fx.GraphModule,
    ):
        super().__init__()

        self.manager = manager
        self.graph = PipeGraph(len(manager.rpc_workers), gm)
        self.sched = manager.scheduler(manager.batch_cnt, self.graph)
        self.workers = manager.rpc_workers

        manager.graph = self.graph

        mods = get_submodules(gm)
        mods = [mod.to("cpu") for mod in mods]
        self.sched.assign_train_steps_to_workers(manager.rpc_workers, mods)

        input_rank = self.graph.input_node.rank
        output_rank = self.graph.output_node.rank
        self.in_worker = self.workers[input_rank - 1]
        self.out_worker = self.workers[output_rank - 1]

        self.out_worker.rpc_sync().set_loss_fn(manager.loss_fn)

    def forward(self, *args, **kwargs):
        # TODO: fire workers from here if there is multiple runners

        # return Future[Token]
        batch_no = self.manager.get_batch_no()

        self.in_worker.rpc_sync().set_input(batch_no, args, kwargs)

        output_token = (
            self.out_worker.rpc_async().get_io_token(batch_no)
            if self.manager.is_train
            else None
        )
        backward_token = self.in_worker.rpc_async().get_io_token(batch_no)

        return [
            PipeTensor(batch_no, output_token, backward_token),
        ]


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
        self.batch_no = 0
        self.is_train = True
        self.graph = None

        self.set_option(option)
        self._compile()

    def set_option(self, option: PipeOption):
        self.stage_cnt = option.stage_cnt
        self.batch_cnt = option.batch_cnt
        self.device_cnt = option.device_cnt
        self.split_cnt = option.stage_cnt

        self.scheduler = get_scheduler(option.scheduler)

        # TODO: split by other strategies
        self.splitter = ParamSplit()

        # TODO: check this will remove (GC-out) worker objects
        self.rpc_workers: List[PyRRef] = []
        for rank in range(1, self.device_cnt + 1):
            device = f"cuda:{rank-1}"
            worker = rpc.remote(
                f"worker_{rank}",
                RPCWorker,
                args=(rank, device, option),
            )
            self.rpc_workers.append(worker)

        for worker in self.rpc_workers:
            worker.rpc_sync().set_workers(self.rpc_workers)

    def _compile(self):
        def compile_fn(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            # TODO: what if this function is called two times from a training loop?

            qualname_map = {}
            split_gm = self.splitter.split(gm, self.stage_cnt, qualname_map)

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
                        split_gm,
                        node,
                        user.target,
                        param_val,
                        qualname_map,
                        use_idx,
                        is_buffer,
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
        self.is_train = True
        self.opt_model.train()
        for worker in self.rpc_workers:
            worker.rpc_sync().set_train(True)

    def eval(self):
        self.is_train = False
        self.opt_model.eval()
        for worker in self.rpc_workers:
            worker.rpc_sync().set_train(False)

    def init_runner(self):
        for worker in self.rpc_workers:
            worker.rpc_sync().reset_cache()

        self.batch_no = 0

    def get_batch_no(self):
        batch_no = self.batch_no
        self.batch_no += 1
        return batch_no

    def run(self, target, *args, **kwargs):
        """
        1. optimizer.zero_grad()
        2. forward & backward & optimizer step
        3. return loss
        """

        # TODO: evaluation pass
        self.init_runner()

        args, kwargs = split_args_kwargs(args, kwargs, self.batch_cnt)
        targets, _ = split_args_kwargs([target], dict(), self.batch_cnt)

        # TODO: Custom Tensor model
        tokens: List[PipeTensor] = []
        for micro_args, micro_kwargs in zip(args, kwargs):
            tokens.append(self.opt_model(*micro_args, **micro_kwargs))

        out_rank = self.graph.output_node.rank
        out_worker = self.rpc_workers[out_rank - 1]
        for batch_id, target in enumerate(targets):
            out_worker.rpc_sync().set_target(batch_id, target)

        # ignite worker (add jobs to job queue)
        for worker in self.rpc_workers:
            worker.rpc_sync().fire()

        # run and wait tokens
        for pt in tokens:
            pt.output_token.wait()
            if pt.backward_token is not None:
                pt.backward_token.wait()

        loss = out_worker.rpc_sync().get_loss("sum")

        return loss
