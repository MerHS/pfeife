import logging
import threading
import time
from typing import Dict, List
from queue import Queue

import torch
import torch.nn as nn
import torch.optim as optim
from torch.futures import Future
from torch.utils._pytree import tree_map

from ..compile import compile_module
from ..option import PipeOption
from ..pipe_graph import PipeGraph, PipeNode
from ..scheduler import Step, StepWork
from ..utils import get_logger, to_device


class ThreadWorker:
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
        self.buffers = dict()

        self.is_train = True

        self.optimizer = None
        self.optimizer_cls = (
            optim.Adam if option.optimizer_type == "adam" else optim.SGD
        )

        self.fw_queue = Queue()
        self.bw_queue = Queue()

        self.reset_cache()

    def reset_cache(self):
        self.targets = [None for _ in range(self.option.batch_cnt)]
        self.losses = [None for _ in range(self.option.batch_cnt)]

        self.fw_inputs = dict()
        self.fw_results = dict()
        self.bw_grads = dict()

    def _init_optimizer(self):
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(
                self.params, **self.option.optimizer_kwargs
            )
        self.optimizer.zero_grad()

    def debug(self, msg):
        t = time.time()
        self.logger.debug(f"({t - self.base_time:8.5f})[Worker {self.rank}] {msg}")

    def info(self, msg):
        t = time.time()
        self.logger.info(f"({t - self.base_time:8.5f})[Worker {self.rank}] {msg}")

    def parameters(self):
        return self.params

    def test_param_and_grad(self):
        return (self.params[0], self.params[0].grad)

    def clear_workers(self):
        self.workers = []

    def set_workers(self, workers: List["ThreadWorker"]):
        self.workers = workers

    def set_graph(self, pipe_graph: PipeGraph):
        self.graph = pipe_graph

    def set_scheduler_steps(self, steps: List[Step]):
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, step in enumerate(steps):
                self.debug(f"set step {i}: {step}")
        self.steps = steps

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

    def set_input(self, batch_id, args, kwargs):
        # TODO: check whether we should handle kwargs
        self.debug(f"set input for batch {batch_id}")

        args = to_device(args, self.device)

        # self.inputs[batch_id] = args

        input_node = self.graph.input_node
        for edge in input_node.out_edges:
            value = args[edge.idx] if edge.idx is not None else args
            for end_node in edge.end_nodes:
                comm_id = self.get_comm_id(0, end_node.idx, edge.idx, batch_id)
                self.fw_inputs[comm_id] = value

    def set_target(self, batch_id, target):
        self.targets[batch_id] = target

    def get_loss(self, reduce="sum"):
        # reduce: 'sum', 'average', None (return list of losses)

        if reduce == "sum":
            loss = torch.sum(torch.stack(self.losses))
        elif reduce == "average":
            loss = torch.mean(torch.stack(self.losses))
        else:
            loss = self.losses

        self.debug("return loss")
        return loss

    def run(self):
        self.debug(f"activate worker {self.rank} from device {self.device}")
        self._init_optimizer()

        self.thread = threading.Thread(target=self.run_thread, daemon=True)
        self.thread.start()
        return self.thread

    def run_thread(self):
        for job in self.steps:
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

    def send_fw(self, target_rank: int, key: str, value):
        target_worker = self.workers[target_rank - 1]
        target_worker.fw_queue.put((key, value))

    def send_bw(self, target_rank: int, key: str, value):
        target_worker = self.workers[target_rank - 1]
        target_worker.bw_queue.put((key, value))

    def wait_fw(self, key: str):
        def map_fw(v):
            if torch.is_tensor(v):
                return v.to(device=self.device).detach().requires_grad_(True)
            else:
                return v

        while key not in self.fw_inputs:
            (recv_key, recv_value) = self.fw_queue.get()
            recv_value = tree_map(map_fw, recv_value)
            self.fw_inputs[recv_key] = recv_value

    def wait_bw(self, key: str):
        while key not in self.bw_grads:
            (recv_key, recv_value) = self.bw_queue.get()
            recv_value = to_device(recv_value, self.device)
            self.bw_grads[recv_key] = recv_value

    def send_act(self, node: PipeNode, batch_id: int):
        curr_rank = node.rank
        curr_id = node.idx

        value = self.fw_results[batch_id]

        for edge in node.out_edges:
            for end_node in edge.end_nodes:
                comm_id = self.get_comm_id(curr_id, end_node.idx, edge.idx, batch_id)

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
                    self.send_fw(end_node.rank, comm_id, value_send)

    def recv_act(self, node: PipeNode, batch_id: int):
        curr_id = node.idx

        input_keys = []
        for edge in node.in_edges:
            comm_id = self.get_comm_id(edge.start_node.idx, curr_id, edge.idx, batch_id)
            input_keys.append(comm_id)

        for key in input_keys:
            self.wait_fw(key)

    def send_grad(self, node: PipeNode, batch_id: int):
        curr_rank = node.rank
        curr_id = node.idx

        for edge in node.in_edges:
            start_node = edge.start_node
            comm_id = self.get_comm_id(start_node.idx, curr_id, edge.idx, batch_id)

            value = self.fw_inputs[comm_id]
            value = tree_map(lambda v: (v.grad if torch.is_tensor(v) else v), value)

            if start_node.rank == curr_rank:
                self.bw_grads[comm_id] = value
            else:
                self.send_bw(start_node.rank, comm_id, value)

    def recv_grad(self, node: PipeNode, batch_id: int):
        curr_id = node.idx

        grad_keys = []
        for edge in node.out_edges:
            for end_node in edge.end_nodes:
                if end_node == self.graph.output_node:
                    continue
                comm_id = self.get_comm_id(curr_id, end_node.idx, edge.idx, batch_id)
                grad_keys.append(comm_id)

        for key in grad_keys:
            self.wait_bw(key)

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
            inputs.append(value.clone())

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
            result = self.loss_fn(result, self.targets[batch_id])
            self.losses[batch_id] = result

        self.fw_results[batch_id] = result

        # torch.cuda.current_stream(self.device).synchronize()

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
        # torch.cuda.current_stream(self.device).synchronize()
