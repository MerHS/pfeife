import logging
import time
from copy import deepcopy
from typing import List, Union

import torch
import torch._dynamo as dynamo
import torch.fx as fx
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from .worker import ThreadWorker
from ..batch import split_args_kwargs
from ..graph_split import ParamSplit
from ..loss import LossWrapper
from ..option import PipeOption
from ..pipe_graph import PipeGraph
from ..scheduler import get_scheduler
from ..utils import get_logger, get_submodules, move_param_to_callee


# TODO: make it a real tensor
class PipeTensor:
    def __init__(self, batch_no):
        self.batch_no = batch_no


class PipeGraphRunner(nn.Module):
    def __init__(
        self,
        manager: "PipeManager",
        gm: fx.GraphModule,
    ):
        super().__init__()

        self.manager = manager
        worker_cnt = len(manager.workers)
        self.graph = PipeGraph(worker_cnt, gm)
        self.sched = manager.scheduler(manager.batch_cnt, self.graph)
        self.workers = manager.workers

        manager.graph = self.graph

        mods = get_submodules(gm)
        mods = [deepcopy(mod) for mod in mods]

        # The graph can be created after the TorchDynamo compiler is activated
        # Thus, assigning a graph and a scheduler to the workers should be
        # deferred until this step.
        self.assign_train_steps_to_workers(manager.workers, mods)

        logger = get_logger()
        if logger.isEnabledFor(logging.DEBUG):
            graph_str = self.graph.to_str()
            logger.debug("=====pipeline graph======")
            logger.debug("\n" + graph_str)

        input_rank = self.graph.input_node.rank
        output_rank = self.graph.output_node.rank
        self.in_worker = self.workers[input_rank - 1]
        self.out_worker = self.workers[output_rank - 1]

        self.out_worker.set_loss_fn(manager.loss_fn)

    def assign_train_steps_to_workers(
        self, workers: List[ThreadWorker], modules: List[torch.nn.Module]
    ):
        sched = self.sched
        graph = self.graph

        for node in graph.internal_nodes:
            worker = workers[node.rank - 1]
            node.device = worker.device

            module = modules[node.idx - 1]
            worker.set_module(node.idx, module)

        input_node = graph.input_node
        input_node.device = workers[input_node.rank - 1].device
        output_node = graph.output_node
        output_node.device = workers[output_node.rank - 1].device

        for worker_id, worker in enumerate(workers):
            worker.set_graph(graph)
            worker.set_scheduler_steps(sched.get_train_steps(worker_id))

    def forward(self, *args, **kwargs):
        batch_no = self.manager.get_batch_no()
        self.in_worker.set_input(batch_no, args, kwargs)
        return [PipeTensor(batch_no)]


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
        self.base_time = time.time()
        self.logger = get_logger()

        self.set_option(option)
        self._compile()

    def __del__(self):
        for worker in self.workers:
            worker.clear_workers()

    def set_option(self, option: PipeOption):
        self.debug(f"got option: {option.__dict__}")

        self.stage_cnt = option.stage_cnt
        self.batch_cnt = option.batch_cnt
        self.device_cnt = option.device_cnt
        self.split_cnt = option.stage_cnt

        self.scheduler = get_scheduler(option.scheduler)

        # TODO: split by other strategies
        self.splitter = ParamSplit()
        self.workers: List[ThreadWorker] = []

        for rank in range(1, self.device_cnt + 1):
            # TODO: configurable device
            device = f"cuda:{rank-1}"
            worker = ThreadWorker(rank, device, option, self.base_time)
            self.workers.append(worker)

        for worker in self.workers:
            worker.set_workers(self.workers)

    def debug(self, msg):
        t = time.time()
        self.logger.debug(f"({t - self.base_time:8.5f})[Master] {msg}")

    def info(self, msg):
        t = time.time()
        self.logger.info(f"({t - self.base_time:8.5f})[Master] {msg}")

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

            self.debug(
                "\n---final graph---\n" + str(split_gm.graph) + "\n---------------\n"
            )

            runner = PipeGraphRunner(self, split_gm)

            return runner

        dynamo_ctx = dynamo.optimize(compile_fn)
        self.opt_model = dynamo_ctx(self.orig_model)

    def train(self):
        self.is_train = True
        self.opt_model.train()
        for worker in self.workers:
            worker.set_train(True)

    def eval(self):
        self.is_train = False
        self.opt_model.eval()
        for worker in self.workers:
            worker.set_train(False)

    def init_runner(self):
        for worker in self.workers:
            worker.reset_cache()

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
        self.debug("start run")

        # TODO: evaluation pass
        self.init_runner()

        args, kwargs = split_args_kwargs(args, kwargs, self.batch_cnt)
        targets, _ = split_args_kwargs([target], dict(), self.batch_cnt)

        # TODO: Custom Tensor model
        tokens: List[PipeTensor] = []
        for micro_args, micro_kwargs in zip(args, kwargs):
            tokens.append(self.opt_model(*micro_args, **micro_kwargs))

        out_rank = self.graph.output_node.rank
        out_worker = self.workers[out_rank - 1]
        for batch_id, target in enumerate(targets):
            out_worker.set_target(batch_id, target)

        # ignite worker
        threads = []
        self.debug("Run workers")
        for worker in self.workers:
            t = worker.run()
            threads.append(t)

        for t in threads:
            t.join()

        loss = out_worker.get_loss("sum")
        self.debug(f"got loss: {loss}")

        return loss
