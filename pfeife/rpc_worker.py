import logging
import threading
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.rpc import PyRRef
from torch.utils._pytree import tree_map

from .compile import compile_module
from .option import PipeOption
from .pipe_graph import PipeGraph, PipeNode
from .scheduler import Step, StepWork
from .utils import get_logger, to_device


class RPCWorker:
    def __init__(self, rank: int, device: str, option: PipeOption, base_time=0):
        self.logger = get_logger()

        self.base_time = base_time
        self.option = option
        self.rank = rank
        self.device = device
        self.compiler = option.compiler

        self.mods: Dict[str, nn.Module] = dict()
        self.mod_compiled = set()

        self.graph = None
        self.optimizer = None
        self.params = []

        self.is_train = True

        self.locks: Dict[str, threading.Condition] = dict()

        # handle io tokens for the master node
        self.add_lock("io_tokens")
        # handle io data for the master node
        self.add_lock("io_data")
        # handle communication. (fw act / bw grad)
        self.add_lock("fb_comm")

        self.optimizer = None
        self.optimizer_cls = (
            optim.Adam if option.optimizer_type == "adam" else optim.SGD
        )

        self.reset_cache()

    def _init_optimizer(self):
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(
                self.params, **self.option.optimizer_kwargs
            )
        self.optimizer.zero_grad()

    def clear_workers(self):
        self.workers = []

    def parameters(self):
        return self.params

    def test_param_and_grad(self):
        return (self.params[0], self.params[0].grad)

    def debug(self, msg):
        t = time.time()
        self.logger.debug(f"({t - self.base_time:8.5f})[Worker {self.rank}] {msg}")

    def info(self, msg):
        t = time.time()
        self.logger.info(f"({t - self.base_time:8.5f})[Worker {self.rank}] {msg}")

    def reset_cache(self):
        self.inputs = [None for _ in range(self.option.batch_cnt)]
        self.targets = [None for _ in range(self.option.batch_cnt)]
        self.losses = [None for _ in range(self.option.batch_cnt)]
        self.outputs = [None for _ in range(self.option.batch_cnt)]

        self.fw_results = dict()
        self.fw_inputs = dict()
        self.bw_grads = dict()

        self.io_tokens = [None for _ in range(self.option.batch_cnt)]
        self.current_job: Step = None

    def add_lock(self, name):
        lock = threading.Lock()
        cv = threading.Condition(lock)
        self.locks[name] = cv

    def get_lock(self, name):
        return self.locks[name]

    def set_workers(self, workers: List[PyRRef]):
        self.workers = workers

    def ping(self, msg):
        self.debug(f"got {msg} from {self.rank}")

    def set_graph(self, pipe_graph: PipeGraph):
        self.graph = pipe_graph

    def set_scheduler_steps(self, steps: List[Step]):
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, step in enumerate(steps):
                self.debug(f"set step {i}: {step}")
        self.steps = steps

    def set_input(self, batch_id, args, kwargs):
        # TODO: check whether we should handle kwargs
        self.debug(f"set input for batch {batch_id}")

        args = to_device(args, self.device)

        io_cv = self.get_lock("io_data")
        with io_cv:
            self.inputs[batch_id] = args
            io_cv.notify_all()

        comm_cv = self.get_lock("fb_comm")
        input_node = self.graph.input_node
        with comm_cv:
            for edge in input_node.out_edges:
                value = args[edge.idx] if edge.idx is not None else args
                for end_node in edge.end_nodes:
                    comm_id = self.get_comm_id(0, end_node.idx, edge.idx, batch_id)
                    self.debug(f"input id for batch {batch_id}: {comm_id}")
                    self.fw_inputs[comm_id] = value
            comm_cv.notify_all()

    def set_target(self, batch_id, target):
        cv = self.get_lock("io_data")
        with cv:
            self.targets[batch_id] = target
            cv.notify_all()

    def get_io_token(self, batch_id):
        self.debug(f"create IO token for {batch_id}")
        cv = self.get_lock("io_tokens")
        with cv:
            cv.wait_for(lambda: self.io_tokens[batch_id] is not None)
            self.debug(f"resolve token for {batch_id}")
            return self.io_tokens[batch_id]

    def get_loss(self, reduce="sum"):
        # reduce: 'sum', 'average', None (return list of losses)
        self.debug("create loss token")
        cv = self.get_lock("io_tokens")
        with cv:
            cv.wait_for(lambda: all([loss is not None for loss in self.losses]))
            if reduce == "sum":
                loss = torch.sum(torch.stack(self.losses))
            elif reduce == "average":
                loss = torch.mean(torch.stack(self.losses))
            else:
                loss = self.losses

            self.debug("return loss")
            return loss

    def set_output(self, batch_id, output_value):
        self.outputs[batch_id] = output_value

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_module(self, mod_id, module):
        # self.info(f"got module of device {next(module.parameters()).device}. send to {self.device}")
        self.mods[mod_id + 1] = module.to(self.device)
        self.params.extend(list(module.parameters()))

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
        self.debug(f"activate worker {self.rank} from device {self.device}")
        self._init_optimizer()

        job_queue = [job for job in self.steps]

        while len(job_queue) > 0:
            job = job_queue.pop(0)

            # synchronous handle job
            self.handle_job(job)

    def handle_job(self, job: Step):
        self.debug(f"start {job}")
        node = self.graph.internal_nodes[job.node_id]

        if job.work == StepWork.SEND_ACT:
            self.send_act(node, job.batch_id)
        elif job.work == StepWork.RECV_ACT:
            self.recv_act(node, job.batch_id)
        elif job.work == StepWork.FORWARD:
            self.forward(node, job.batch_id)
        elif job.work == StepWork.SEND_GRAD:
            self.send_grad(node, job.batch_id)
        elif job.work == StepWork.RECV_GRAD:
            self.recv_grad(node, job.batch_id)
        elif job.work == StepWork.BACKWARD:
            self.backward(node, job.batch_id)
        elif job.work == StepWork.OPTIMIZER_STEP:
            self.optimizer_step(node, job.batch_id)

    def get_comm_id(self, start_id: int, end_id: int, edge_id: int, batch_id: int):
        return f"comm_{start_id}_{end_id}_{edge_id}_{batch_id}"

    def push_fw_act(self, key: str, value):
        self.debug(f"received forward activation: {key}")
        cv = self.get_lock("fb_comm")
        with cv:
            self.fw_inputs[key] = value
            cv.notify_all()

    def push_bw_grad(self, key: str, value):
        self.debug(f"received backward gradient: {key}")
        cv = self.get_lock("fb_comm")
        with cv:
            self.bw_grads[key] = value
            cv.notify_all()

    def send_act(self, node: PipeNode, batch_id: int):
        cv = self.get_lock("fb_comm")
        curr_rank = node.rank
        curr_id = node.idx

        # TODO: check this requires a lock
        value = self.fw_results[batch_id]

        # check multi rank output
        same_rank = False
        diff_rank = False
        for edge in node.out_edges:
            for end_node in edge.end_nodes:
                if end_node.rank != curr_rank:
                    same_rank = True
                else:
                    diff_rank = True

        if same_rank and diff_rank:

            def detach(v):
                if torch.is_tensor(v) and v.requires_grad:
                    return v.detach().requires_grad_(True)
                else:
                    return v

            value = tree_map(detach, value)

        with cv:
            for edge in node.out_edges:
                for end_node in edge.end_nodes:
                    comm_id = self.get_comm_id(
                        curr_id, end_node.idx, edge.idx, batch_id
                    )

                    self.debug(
                        f"<SEND_ACT> send to id {comm_id} (node: {end_node.idx} / rank: {end_node.rank})"
                    )

                    if edge.idx is not None:
                        value_send = value[edge.idx]
                    else:
                        value_send = value

                    if end_node.rank == curr_rank:
                        self.fw_inputs[comm_id] = value_send
                    else:
                        send_ref = self.workers[end_node.rank - 1]
                        send_ref.rpc_async().push_fw_act(comm_id, value_send)
            cv.notify_all()

    def recv_act(self, node: PipeNode, batch_id: int):
        cv = self.get_lock("fb_comm")
        curr_id = node.idx

        input_keys = []
        for edge in node.in_edges:
            comm_id = self.get_comm_id(edge.start_node.idx, curr_id, edge.idx, batch_id)
            input_keys.append(comm_id)

        if self.logger.isEnabledFor(logging.DEBUG):
            for comm_id in input_keys:
                self.debug(
                    f"<RECV_ACT> requires '{comm_id}' : {'found' if comm_id in self.fw_inputs else 'not found, wait for input'}"
                )

        def checker():
            for key in input_keys:
                if key not in self.fw_inputs:
                    return False
            return True

        with cv:
            cv.wait_for(checker)

    def send_grad(self, node: PipeNode, batch_id: int):
        cv = self.get_lock("fb_comm")
        curr_rank = node.rank
        curr_id = node.idx

        with cv:
            for edge in node.in_edges:
                start_node = edge.start_node
                comm_id = self.get_comm_id(start_node.idx, curr_id, edge.idx, batch_id)

                value = self.fw_inputs[comm_id]
                value = tree_map(lambda v: (v.grad if torch.is_tensor(v) else v), value)

                if start_node.rank == curr_rank:
                    self.bw_grads[comm_id] = value
                else:
                    send_ref = self.workers[start_node.rank - 1]
                    send_ref.rpc_async().push_bw_grad(comm_id, value)

            cv.notify_all()

    def recv_grad(self, node: PipeNode, batch_id: int):
        cv = self.get_lock("fb_comm")
        curr_id = node.idx

        grad_keys = []
        for edge in node.out_edges:
            for end_node in edge.end_nodes:
                if end_node == self.graph.output_node:
                    continue
                comm_id = self.get_comm_id(curr_id, end_node.idx, edge.idx, batch_id)
                grad_keys.append(comm_id)

        if self.logger.isEnabledFor(logging.DEBUG):
            for comm_id in grad_keys:
                self.debug(
                    f"<RECV_GRAD> requires '{comm_id}' : {'found' if comm_id in self.bw_grads else 'not found, wait for input'}"
                )

        def checker():
            for key in grad_keys:
                if key not in self.bw_grads:
                    return False
            return True

        with cv:
            cv.wait_for(checker)

    def optimizer_step(self, node: PipeNode, batch_id: int):
        self.optimizer.step()
        torch.cuda.current_stream(self.device).synchronize()
        self.debug("optimizer end")

    def forward(self, node: PipeNode, batch_id: int):
        curr_id = node.idx
        inputs = []
        for edge in node.in_edges:
            comm_id = self.get_comm_id(edge.start_node.idx, curr_id, edge.idx, batch_id)
            value = self.fw_inputs[comm_id]
            inputs.append(value)

        mod = self.mods[curr_id]

        if curr_id not in self.mod_compiled:
            self.debug(f"compile submodule {curr_id}, input device: {inputs[0].device}")
            mod = compile_module(self.compiler, mod, inputs)
            self.mods[curr_id] = mod
            self.mod_compiled.add(curr_id)

        result = mod(*inputs)

        # TODO: multi-node output node

        is_end_node = False
        for edge in node.out_edges:
            for out_node in edge.end_nodes:
                if out_node.is_io:
                    is_end_node = True
                    break
        is_end_node = is_end_node and self.loss_fn is not None

        if is_end_node:
            io_cv = self.get_lock("io_data")
            with io_cv:
                io_cv.wait_for(lambda: self.targets[batch_id] is not None)
                result = self.loss_fn(result, self.targets[batch_id])
                self.losses[batch_id] = result

        self.fw_results[batch_id] = result

        torch.cuda.current_stream(self.device).synchronize()

        if is_end_node:
            token_cv = self.get_lock("io_tokens")
            with token_cv:
                self.io_tokens[batch_id] = True
                token_cv.notify_all()

    def backward(self, node: PipeNode, batch_id: int):
        curr_id = node.idx

        grads = []
        for edge in node.out_edges:
            if edge.end_nodes[0] == self.graph.output_node:
                grads = None
                break

            grad = None
            # TODO: reduce properly (regarding edge idx)
            for end_node in edge.end_nodes:
                comm_id = self.get_comm_id(curr_id, end_node.idx, edge.idx, batch_id)
                comm_grad = self.bw_grads[comm_id]

                if grad is None:
                    grad = comm_grad
                else:
                    grad += comm_grad
            grads.append(grad)

        torch.autograd.backward(self.fw_results[batch_id], grads)
        torch.cuda.current_stream(self.device).synchronize()

        # is first node
        if node.idx == 1:
            token_cv = self.get_lock("io_tokens")
            with token_cv:
                self.io_tokens[batch_id] = True
                token_cv.notify_all()
