import threading
from typing import TYPE_CHECKING, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch._dynamo.optimizations import BACKENDS
from torch._inductor.compile_fx import compile_fx
from torch.distributed.rpc import PyRRef
from torch.fx import Node
from torch.utils._pytree import tree_flatten, tree_map

from .option import PipeOption
from .scheduler import Step, StepWork
from .utils import to_device

if TYPE_CHECKING:
    from .pipe_graph import PipeGraph


class WrapperModule(torch.nn.Module):
    def __init__(self, compiled_submod, unwrap_singleton_tuple):
        super().__init__()
        self.compiled_submod = compiled_submod
        self.unwrap_singleton_tuple = unwrap_singleton_tuple

    def forward(self, *args):
        x = self.compiled_submod(*args)
        # TODO(whc)
        # for some reason the isinstance check is necessary if I split one node per submod
        # - even though I supposedly wrapped the output in a tuple in those cases, the real
        # compiled module was still returning a tensor
        if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
            return x[0]
        return x


class RPCWorker:
    def __init__(self, rank: int, device: str, option: PipeOption):
        self.option = option
        self.rank = rank
        self.device = device
        self.compiler = option.compiler
        self.mods: Dict[str, nn.Module] = dict()

        self.graph = None
        self.optimizer = None

        self.is_compiled = False
        self.is_train = True

        self.locks: Dict[str, threading.Condition] = dict()

        # handle job queue
        self.add_lock("job_queue")
        # handle io tokens for the master node
        self.add_lock("io_tokens")
        # handle io data for the master node
        self.add_lock("io_data")
        # handle communication. (fw act / bw grad)
        self.add_lock("fb_comm")

        if type(self.compiler) == str:
            if self.compiler == "inductor":
                torch._inductor.config.triton.cudagraphs = False
                self.compiler = compile_fx
            else:
                self.compiler = BACKENDS[self.compiler]

        self.optimizer_cls = (
            optim.Adam if option.optimizer_type == "adam" else optim.SGD
        )

        self.reset_cache()

        self.run_loop = threading.Thread(
            target=self.runner, name=f"worker_runner_{rank}", daemon=True
        )
        self.run_loop.start()

    def _init_optimizer(self):
        self.optimizer = self.optimizer_cls(
            self.stage_mod.parameters(), **self.option.optimizer_kwargs
        )

    def reset_cache(self):
        self.inputs = [None for _ in range(self.option.batch_cnt)]
        self.targets = [None for _ in range(self.option.batch_cnt)]
        self.losses = [None for _ in range(self.option.batch_cnt)]
        self.outputs = [None for _ in range(self.option.batch_cnt)]

        self.fw_results = dict()
        self.fw_inputs = dict()
        self.bw_grads = dict()

        self.io_tokens = [None for _ in range(self.option.batch_cnt)]
        self.job_queue = []

    def add_lock(self, name):
        lock = threading.Lock()
        cv = threading.Condition(lock)
        self.locks[name] = cv

    def get_lock(self, name):
        return self.locks[name]

    def set_workers(self, workers: List[PyRRef]):
        self.workers = workers

    def set_graph(self, pipe_graph: PipeGraph):
        self.graph = pipe_graph

    def set_scheduler_steps(self, steps: List[Step]):
        self.steps = steps

    def set_input(self, batch_id, args, kwargs):
        # TODO: check whether we should handle kwargs
        cv = self.get_lock("io_data")
        with cv:
            self.inputs[batch_id] = args
            cv.notify()

    def set_target(self, batch_id, target):
        cv = self.get_lock("io_data")
        with cv:
            self.targets[batch_id] = target
            cv.notify()

    def get_io_token(self, batch_id):
        cv = self.get_lock("io_tokens")
        with cv:
            cv.wait_for(lambda: self.io_tokens[batch_id] is not None)
            return self.io_tokens[batch_id]

    def get_loss(self, reduce="sum"):
        # reduce: 'sum', 'average', None (return list of losses)
        cv = self.get_lock("io_tokens")
        with cv:
            cv.wait_for(lambda: all([loss is not None for loss in self.losses]))
            if reduce == "sum":
                loss = torch.sum(self.losses)
            elif reduce == "average":
                loss = torch.mean(self.losses)
            else:
                loss = self.losses
            return loss

    def set_output(self, batch_id, output_value):
        self.outputs[batch_id] = output_value

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_module(self, mod_id, module):
        self.mods[mod_id] = module.to(self.device)

    def get_device(self):
        return self.device

    def set_device(self, device):
        self.device = device

    def set_train(self, train: bool):
        self.is_train = train
        for mod in self.mods.values():
            if self.is_train:
                mod.train()
            else:
                mod.eval()

    def fire(self):
        cv = self.get_lock("job_queue")
        with cv:
            # TODO: append job queue
            cv.notify()

    def runner(self):
        cv = self.get_lock("job_queue")
        while True:
            with cv:
                self.cv.wait_for(lambda: len(self.job_queue) > 0)
                job = self.job_queue.pop(0)

            # TODO: handle job
            self.handle_job(job)

    def handle_job(self, job):
        pass

    def compile_submod(self, submod, args):
        args = tree_map(to_device, args)

        unwrap_singleton_tuple = False
        for sn in submod.graph.nodes:
            if sn.op == "output":
                if not isinstance(sn.args[0], tuple):
                    unwrap_singleton_tuple = True
                    sn.args = (sn.args,)
        submod.recompile()

        wrapper = WrapperModule(
            self.compiler(submod, args),
            unwrap_singleton_tuple,
        )

        self.stage_mod = wrapper

    def forward(self, batch_no, *args):
        if self.rank == 0:
            args = [a.to(self.device) for a in args]
        else:
            fw_args = [a.detach().to(self.device).requires_grad_(True) for a in args]
            self.fw_inputs[batch_no] = fw_args
            args = [a.clone() for a in fw_args]

        with self.lock:
            if not self.is_compiled:
                self.compile_submod(args)
                self.is_compiled = True
            result = self.stage_mod(*args)
            self.fw_results[batch_no] = result

        result_cpu = [r.detach().cpu() for r in result]

        return result_cpu

    def forward_loss(self, batch_no, target, loss_fn):
        with self.lock:
            last_result = self.fw_results[batch_no]
            loss = loss_fn(last_result, target)
            self.fw_results[batch_no] = loss

        loss_cpu = loss.detach().cpu()

        return loss_cpu

    def backward(self, batch_no, grads):
        if grads is not None:
            grads = [a.to(self.device) for a in grads]

        with self.lock:
            torch.autograd.backward(self.fw_results[batch_no], grads)

        next_grads = []

        if self.rank != 0:
            back_input = self.fw_inputs[batch_no]
            next_grads = [g.grad.cpu() for g in back_input]

        return next_grads
