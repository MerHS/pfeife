import logging
import threading
from enum import Enum
from typing import Dict, List
from queue import Queue
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.nvtx as nvtx
import torch.distributed as dist

from ..option import PipeOption, get_optimizer_cls
from ..schedule.scheduler import Step, StepWork
from ..graph.computation_graph import ParamSet
from ..graph.partition_graph import PartitionGraph, PartitionNode, module_fetch_attr
from ..utils import get_logger, to_device, fmem, list_params
from ..timing import CPUTimeRecorder, CUDATimeRecorder
from ..compile import compile_module


FILTER_GRAD_NONE = "<<FILTER_GRAD_NONE>>"


class SendType(Enum):
    FORWARD = 0
    GRAD = 1
    RESULT = 2


class SizeOnly:
    pass


def list_filter(f, t):
    l = []

    if type(t) in (list, tuple):
        for i in t:
            l.extend(list_filter(f, i))
    else:
        v = f(t)
        if v is not None:
            l.append(v)
    return l


def list_map(f, t):
    if type(t) is list:
        return [list_map(f, i) for i in t]
    elif type(t) is tuple:
        return tuple([list_map(f, i) for i in t])
    else:
        return f(t)


def filter_tensor(t):
    if torch.is_tensor(t) or t is FILTER_GRAD_NONE:
        return t


def filter_grad(t):
    if torch.is_tensor(t) and t.requires_grad:
        return t.grad


def filter_grad_buf(t):
    if torch.is_tensor(t) and t.requires_grad:
        if t.grad is None:
            return FILTER_GRAD_NONE
        return t.grad


def collect_req_grads(v):
    if torch.is_tensor(v) and v.requires_grad:
        return v
    return None


def to_size_only(t):
    return SizeOnly() if torch.is_tensor(t) else t


def run_reducer(rank, reducers, queues, world_size, backend="nccl"):
    reducer = reducers[rank]
    sq, rq = queues[rank]

    # os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    # os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    # print(f"trying to make a process group on {rank}/{world_size}", flush=True)

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # print(f"init reducer {rank}", flush=True)
    # sq.put("initialized")

    while True:
        (fn, load) = rq.get()

        if fn == "exit":
            # print(f"reducer exit {rank}")
            sq.put(None)
            break
        elif fn == "group":
            reducer.make_group(load)
        else:
            try:
                sq.put(reducer.all_reduce_tensor(*load))
            except Exception as e:
                err_msg = f"error in reducer {rank}: {e}"
                print(err_msg, flush=True)
                sq.put(err_msg)

        del load


class GradReduceProcess:
    def __init__(self, rank: int, option: PipeOption):
        self.option = option
        self.rank = rank
        self.world_size = option.device_cnt

        self.tensors = dict()
        self.groups = dict()

    def set_queues(self, send_queue, recv_queue):
        self.send_queue = send_queue
        self.recv_queue = recv_queue

    def send_all_reduce(self, *load):
        self.recv_queue.put(("send", load))
        ret_val = self.send_queue.get()
        return ret_val

    def send_make_group(self, group):
        self.recv_queue.put(("group", group))  # non-blocking

    def exit(self):
        self.recv_queue.put(("exit", None))
        self.send_queue.get()

    def make_group(self, group):
        group = tuple(group)
        if group not in self.groups:
            self.groups[group] = dist.new_group(group)

        # print("made group", self.rank, group, flush=True)

    def all_reduce_tensor(self, key, group, tensor):
        group = tuple(group)

        # print("try to reduce", self.rank, key, group, tensor.shape, flush=True)

        proc_group = self.groups[group]

        dist.all_reduce(tensor, group=proc_group)
        dist.barrier(group=proc_group)

        # print("finish reduce", self.rank, key, group, tensor.shape, flush=True)


class ThreadWorker:
    def __init__(self, rank: int, loss_fn, option: PipeOption, base_time=0):
        self.logger = get_logger()

        self.rank = rank
        self.loss_fn = loss_fn

        self.option = option
        self.base_time = base_time

        self.device = option.device_map[rank] if not option.cpu_test else "cpu"
        self.compiler = option.compiler

        self.part_graph: PartitionGraph = None
        self.mods: Dict[int, nn.Module] = dict()
        self.mod_compiled = set()
        self.params = []
        self.optimizer = None

        self.is_train = True
        self.steps = None
        self.runner = None

        self.reducer = GradReduceProcess(self.rank, option)
        self.reducer_queues = None
        self.optimizer_cls = get_optimizer_cls(option.optimizer_type)

        self.steps = []

        self.fw_queue = Queue()
        self.bw_queue = Queue()
        self.send_queues = [Queue() for _ in range(option.device_cnt + 1)]
        self.job_queue = Queue()

        # Real inputs and outputs
        self.fw_inputs = dict()
        self.fw_results = dict()
        self.bw_grads = dict()
        self.fw_events = dict()
        self.bw_events = dict()

        # IO Tensor buffers
        self.input_buffers = dict()
        self.grad_buffers = dict()

        # Received Buffers
        self.copy_act_buffers = dict()
        self.copy_grad_buffers = dict()
        self.loss_grads = dict()

        if option.cpu_test:
            self.send_streams = [None]
            self.send_ctxs = [nullcontext()]
        elif option.single_comm_thread:
            self.send_streams = [torch.cuda.Stream(self.device)]
            self.send_ctxs = [torch.cuda.stream(self.send_streams[0])]
        else:
            self.send_streams = [
                torch.cuda.Stream(self.device) for _ in range(option.device_cnt + 1)
            ]
            self.send_ctxs = [
                torch.cuda.stream(self.send_streams[i])
                for i in range(option.device_cnt + 1)
            ]

        self.recv_flags = [dict() for _ in range(option.device_cnt + 1)]

        if not option.cpu_test:
            self.fw_stream = torch.cuda.Stream(self.device)
            self.fw_ctx = torch.cuda.stream(self.fw_stream)

            self.recorder = CUDATimeRecorder()

            thread_fn = (
                self._send_thread_as_is
                if option.no_io_cache
                else self._send_thread_copy
            )

            if option.single_comm_thread:
                send_ranks = [0]
            else:
                send_ranks = range(-1, self.option.device_cnt)

            self.send_threads = []
            for rank in send_ranks:
                t = threading.Thread(
                    target=thread_fn,
                    daemon=True,
                    args=(rank,),
                    name=f"send_from{self.rank}_to{rank}",
                )
                t.start()
                self.send_threads.append(t)
        else:
            self.fw_stream = None
            self.fw_ctx = nullcontext()

            self.recorder = CPUTimeRecorder()

        self.reset_cache()

        if self.option.cpu_test:
            target = self._handle_thread_cpu
        else:
            target = self._handle_thread

        self.job_thread = threading.Thread(
            target=target, daemon=True, name=f"job_thread_{self.rank}"
        )
        self.job_thread.start()

    def reset_cache(self):
        self.targets = [None for _ in range(self.option.batch_cnt)]
        self.losses = [None for _ in range(self.option.batch_cnt)]

        for recv_flag in self.recv_flags:
            recv_flag.clear()

        self.bw_grad_tokens = dict()

        self.fw_events.clear()
        self.bw_events.clear()

        self.fw_inputs.clear()
        self.fw_results.clear()
        self.bw_grads.clear()
        self.loss_grads.clear()

        self.recorder.reset()

    def _init_optimizer(self, set_to_none=False):
        if self.optimizer is None:
            if len(self.params) > 0:
                self.optimizer = self.optimizer_cls(
                    self.params, **self.option.optimizer_kwargs
                )
                # TODO: should we change it to none?
                self.optimizer.zero_grad(set_to_none=set_to_none)
        else:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def debug(self, msg):
        self.logger.debug("[Worker %s] %s", self.rank, msg)

    def info(self, msg):
        self.logger.info("[Worker %s] %s", self.rank, msg)

    def warn(self, msg):
        self.logger.warn("[Worker %s] %s", self.rank, msg)

    def error(self, msg):
        self.logger.error("[Worker %s] %s", self.rank, msg)

    def get_event_logs(self):
        return self.recorder.get_event_logs()

    def parameters(self):
        return self.params

    def test_param_and_grad(self):
        return (self.params[0], self.params[0].grad)

    def clear_workers(self):
        self.workers = []

    def set_workers(self, workers: List["ThreadWorker"]):
        self.workers = workers

    def set_runner(self, runner):
        self.runner = runner

    def collect_reduce_grads(self, param_set: ParamSet, grad_ranks_all):
        attrs: Dict[str, str] = dict()
        submods: Dict[str, str] = dict()

        target_mods = dict()

        # param_idx ->  gradient + shared ranks
        grad_ranks: Dict[int, List[int]] = defaultdict(list)
        grad_tensors: Dict[int, torch.Tensor] = dict()

        for (target, rank), (
            moved_target,
            is_attr,
        ) in param_set.part_graph_target.items():
            if rank != self.rank:
                continue

            if is_attr:
                attrs[moved_target] = target
            else:
                submods[moved_target] = target

        for gm in self.mods.values():
            for node in gm.graph.nodes:
                if node.target in attrs:
                    target_mods[node.target] = gm
                if node.target in submods:
                    target_mods[node.target] = gm

        items = [*attrs.items(), *submods.items()]
        for moved_target, target in items:
            param_idxs = param_set.param_set[target]
            shared_idxs = param_set.shared_idx[moved_target]
            mod = target_mods[moved_target]

            attr = module_fetch_attr(mod, moved_target)
            params = list_params(attr)

            for shared_id in shared_idxs:
                pos = param_idxs.index(shared_id)
                tensor = params[pos]
                grad_tensors[shared_id] = tensor

        for idx in grad_tensors.keys():
            grad_ranks[idx] = grad_ranks_all[idx]

        self.grad_ranks = sorted(list(grad_ranks.items()))

        # print(f"found shared params {self.rank}: {self.grad_ranks}")
        self.grad_tensors = grad_tensors

        return grad_tensors

    def set_partition_graph(self, part_graph: PartitionGraph):
        self.part_graph = part_graph

        batch_cnt = self.option.batch_cnt

        for node in part_graph.nodes.values():
            if node.rank != self.rank:
                continue

            module = node.graph_module.to(self.device)
            self.mods[node.idx] = module
            self.params.extend(list(module.parameters()))

        # skip if io opt is disabled
        if self.option.no_io_cache:
            return

        for node in part_graph.nodes.values():
            if node.rank != self.rank:
                continue

            # fill in input activation buffers and send it to input nodes
            for input_idx, in_str in enumerate(node.inputs):
                in_node_idx = node.input_map[in_str]
                if in_node_idx < 0:
                    continue

                in_rank = part_graph.nodes[in_node_idx].rank
                in_worker = self.workers[in_rank]

                for batch_id in range(batch_cnt):
                    comm_id = self.get_comm_id(
                        in_node_idx, node.idx, input_idx, batch_id
                    )

                    tensors = []
                    types = []
                    for size, dtype, _, req_grad in node.input_shapes[input_idx]:
                        t = torch.zeros(
                            size,
                            dtype=dtype,
                            device=self.device,
                            requires_grad=req_grad,
                        )
                        tensors.append(t)
                        types.append((dtype, size, req_grad))

                    self.input_buffers[comm_id] = tensors
                    in_worker.set_act_buffer(comm_id, tensors)

            # fill in output grad buffers and send it to input nodes
            for out_idx, out_str in enumerate(node.outputs):
                for out_node_idx in node.output_map[out_str]:
                    if out_node_idx < 0:
                        continue

                    next_node = self.part_graph.nodes[out_node_idx]
                    out_worker = self.workers[next_node.rank]
                    input_idx = next_node.inputs.index(out_str)

                    for batch_id in range(batch_cnt):
                        comm_id = self.get_comm_id(
                            node.idx, out_node_idx, input_idx, batch_id
                        )

                        tensors = []
                        types = []
                        for size, dtype, _, req_grad in node.output_shapes[out_idx]:
                            if req_grad:
                                t = torch.zeros(size, dtype=dtype, device=self.device)
                                tensors.append(t)
                                types.append((dtype, size))

                        self.grad_buffers[comm_id] = tensors
                        out_worker.set_grad_buffer(comm_id, tensors)

    def set_act_buffer(self, comm_id, tensors):
        self.copy_act_buffers[comm_id] = tensors

    def set_grad_buffer(self, comm_id, tensors):
        self.copy_grad_buffers[comm_id] = tensors

    def set_scheduler_steps(self, steps: List[Step]):
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, step in enumerate(steps):
                self.debug(f"set step {i}: {step}")
        self.steps = steps

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

    def load_input(self, batch_id, node_id, input_id, value):
        # TODO: check whether we should handle kwargs
        self.debug(f"set input for batch {batch_id} / input_id: {input_id}")

        value = to_device(value, self.device)
        comm_id = self.get_comm_id(-1, node_id, input_id, batch_id)
        self.fw_inputs[comm_id] = value

    def set_target(self, batch_id, target):
        self.targets[batch_id] = target

    def get_loss(self, reduce="sum"):
        # reduce: 'sum', 'average', None (return list of losses)

        if reduce == "sum":
            loss = sum(self.losses)
        elif reduce == "average":
            loss = sum(self.losses) / len(self.losses)
        else:
            loss = self.losses

        self.debug("return loss")
        return loss

    def run(self):
        self.debug(f"activate worker {self.rank} from device {self.device}")

        if self.compiler is not None:
            torch.compiler.cudagraph_mark_step_begin()

        self._init_optimizer(set_to_none=True)

        if not self.option.cpu_test:
            self.fw_stream.synchronize()
            for t in self.send_streams:
                t.synchronize()

        self.job_queue.put(self.steps)

        return self.job_thread

    def _handle_thread_cpu(self):
        while True:
            steps = self.job_queue.get()

            for job in steps:
                self._handle_job_cpu(job)

            if self.runner is not None:
                self.runner.pipe_done()

    def init_reducer(self, queue, param_set, grad_ranks_all):
        self.reducer.set_queues(*queue)
        for group in grad_ranks_all.values():
            self.reducer.send_make_group(group)

        self.collect_reduce_grads(param_set, grad_ranks_all)

    def _handle_thread(self):
        # TODO: clean all and remove thread
        while True:
            steps = self.job_queue.get()

            for job in steps:
                self._handle_job(job)

            if self.runner is not None:
                self.runner.pipe_done()

    def _get_send_queue(self, rank):
        if self.option.single_comm_thread:
            return self.send_queues[0]
        else:
            return self.send_queues[rank]

    def _send_thread_copy(self, send_rank):
        send_queue = self.send_queues[send_rank]
        recv_flag = self.recv_flags[send_rank]

        if send_rank < 0:  # send to the main thread
            while True:
                event, batch_idx, value, out_rank, send_type = send_queue.get()

                out_idx, req_grad = out_rank
                self.runner.return_output(batch_idx, out_idx, value, req_grad, event)
        else:
            send_stream = self.send_streams[send_rank]
            send_ctx = self.send_ctxs[send_rank]

            def send_tensor(event, out_rank, comm_id, value, send_type):
                event.wait(send_stream)
                if send_type is SendType.FORWARD:
                    buffer = self.copy_act_buffers[comm_id]
                elif send_type is SendType.GRAD:
                    buffer = self.copy_grad_buffers[comm_id]
                else:
                    buffer = []

                if send_type is not SendType.RESULT:
                    tensors = list_filter(filter_tensor, value)

                    if len(buffer) != len(tensors):
                        self.error(
                            f"send buffer size mismatch {len(buffer)} != {len(tensors)} / {comm_id} / {send_type}"
                        )

                    with send_ctx:
                        with torch.no_grad():
                            for recv_t, send_t in zip(buffer, tensors):
                                if send_t is FILTER_GRAD_NONE:
                                    continue
                                recv_t.copy_(send_t, non_blocking=True)

                send_event = torch.cuda.Event()
                send_event.record(send_stream)

                # for out_rank, comm_id, value in payloads:
                if send_type is SendType.FORWARD:
                    send_values = list_map(to_size_only, value)
                    self.workers[out_rank].recv_act_channel(
                        comm_id, send_values, send_event
                    )
                elif send_type is SendType.GRAD:
                    self.workers[out_rank].recv_grad_channel(comm_id, send_event)
                else:
                    # comm_id = batch_id
                    out_idx, req_grad = out_rank
                    self.runner.return_output(comm_id, out_idx, value, req_grad, event)

            pending_load = dict()

            while True:
                send_load = send_queue.get()
                to_send = []

                if len(send_load) == 2:  # request
                    comm_id, is_fwd = send_load
                    recv_flag[send_load] = True
                    if send_load in pending_load:
                        to_send.append(pending_load.pop(send_load))

                elif len(send_load) == 3:  # payload
                    event, payloads, send_type = send_load
                    for out_rank, comm_id, value in payloads:
                        is_fwd = send_type is SendType.FORWARD
                        load = (event, out_rank, comm_id, value, send_type)
                        if out_rank >= 0:
                            if (comm_id, is_fwd) not in recv_flag:
                                pending_load[(comm_id, is_fwd)] = load
                            else:
                                to_send.append(load)
                        else:
                            send_tensor(*load)

                for load in to_send:
                    send_tensor(*load)

    def _send_thread_as_is(self, send_rank):
        send_queue = self.send_queues[send_rank]

        if send_rank < 0:  # send to the main thread
            while True:
                event, batch_idx, value, out_rank, send_type = send_queue.get()

                out_idx, req_grad = out_rank
                self.runner.return_output(batch_idx, out_idx, value, req_grad, event)
        else:
            send_stream = self.send_streams[send_rank]
            send_ctx = self.send_ctxs[send_rank]

            while True:
                event, payloads, send_type = send_queue.get()

                curr_rank = self.rank
                event.wait(send_stream)

                with send_ctx:
                    for end_rank, comm_id, value_send in payloads:

                        def detach(v):
                            if torch.is_tensor(v):
                                req_grad = v.requires_grad
                                v = v.to(device=self.workers[end_rank].device).detach()
                                if req_grad:
                                    v = v.requires_grad_(True)
                            return v

                        if send_type is SendType.FORWARD:
                            value_send = list_map(detach, value_send)
                        elif send_type is SendType.GRAD:
                            value_send = to_device(
                                value_send, self.workers[end_rank].device
                            )

                        send_stream.synchronize()

                        # self.debug(f"send {comm_id} to {end_rank}")
                        if send_type is SendType.FORWARD:
                            if end_rank == curr_rank:
                                self.fw_inputs[comm_id] = value_send
                            else:
                                self._send_fw(end_rank, comm_id, value_send)
                        elif send_type is SendType.GRAD:
                            if end_rank == curr_rank:
                                self.bw_grads[comm_id] = value_send
                            else:
                                self._send_bw(end_rank, comm_id, value_send)

    def recv_act_channel(self, comm_id, size_only_value, send_event):
        # self.debug(f"recv_act_channel {comm_id}")

        send_buffers = self.input_buffers[comm_id]
        idx = 0

        def assign_buffer(t):
            nonlocal idx
            if isinstance(t, SizeOnly):
                idx += 1
                # to enable in-place operation
                tx = send_buffers[idx - 1]
                tx.grad = None
                return tx
            else:
                return t

        value = list_map(assign_buffer, size_only_value)

        self.fw_queue.put((comm_id, value, send_event))

    def recv_grad_channel(self, comm_id, send_event):
        # self.debug(f"recv_grad_channel {comm_id}")
        self.bw_queue.put((comm_id, send_event))

    def recv_loss_grad(self, batch_idx: int, node_id: int, output_idx: int, grad):
        # self.debug(
        #     f"got loss_grad of {batch_idx} / {node_id} / {output_idx} / isNone {grad is None}"
        # )

        comm_id = self.get_comm_id(node_id, -1, output_idx, batch_idx)

        if not self.option.no_side_effect:
            self.loss_grads[comm_id] = [grad]

        if self.option.no_io_cache:
            self.bw_queue.put((comm_id, grad))
        else:
            self.bw_queue.put((comm_id, None))

    def _handle_job(self, job: Step):
        self.debug(job)

        nvtx.range_push(f"DEV{self.rank}_{job}")

        node = None
        if job.node_id >= 0:
            node = self.part_graph.nodes[job.node_id]

        if job.work == StepWork.SEND_ACT:
            self.send_act_async(node, job.batch_id)
        elif job.work == StepWork.RECV_ACT:
            self.recv_act_async(node, job.batch_id)
        elif job.work == StepWork.FORWARD:
            self.forward(node, job.batch_id)
        elif job.work == StepWork.SEND_GRAD:
            self.send_grad_async(node, job.batch_id)
        elif job.work == StepWork.RECV_GRAD:
            self.recv_grad_async(node, job.batch_id)
        elif job.work == StepWork.BACKWARD:
            self.backward(node, job.batch_id)
        elif job.work == StepWork.OPTIMIZER_STEP:
            self.optimizer_step(node, job.batch_id)

        # torch.cuda.synchronize(self.device)
        # print(job, self.device, fmem(torch.cuda.memory_allocated(self.device)))
        # print(job, self.device, fmem(torch.cuda.max_memory_allocated(self.device)))
        # print(job, self.device, fmem(torch.cuda.memory_reserved(self.device)))

        nvtx.range_pop()

    def _handle_job_cpu(self, job: Step):
        self.debug(job)
        node = None
        if job.node_id >= 0:
            node = self.part_graph.nodes[job.node_id]

        if job.work == StepWork.SEND_ACT:
            self.send_act_sync(node, job.batch_id)
        elif job.work == StepWork.RECV_ACT:
            self.recv_act_async(node, job.batch_id)
        elif job.work == StepWork.FORWARD:
            self.forward(node, job.batch_id)
        elif job.work == StepWork.SEND_GRAD:
            self.send_grad_sync(node, job.batch_id)
        elif job.work == StepWork.RECV_GRAD:
            self.recv_grad_async(node, job.batch_id)
        elif job.work == StepWork.BACKWARD:
            self.backward(node, job.batch_id)
        elif job.work == StepWork.OPTIMIZER_STEP:
            self.optimizer_step(node, job.batch_id)

    def get_comm_id(self, start_id: int, end_id: int, input_id: int, batch_id: int):
        """
        start_id: id of the starting node (-1: comes from external runner)
        end_id: id of the ending node
        input_id: place order of the node.inputs
        batch_id: batch index
        """
        return f"comm_{start_id}_{end_id}_{input_id}_{batch_id}"

    def get_result_id(self, node_id: int, batch_id: int):
        return f"result_{node_id}_{batch_id}"

    def _send_fw(self, target_rank: int, key: str, value):
        target_worker = self.workers[target_rank]
        target_worker.fw_queue.put((key, value, None))

    def send_act_sync(self, node: PartitionNode, batch_id: int):
        curr_rank = node.rank
        curr_id = node.idx

        result_id = self.get_result_id(node.idx, batch_id)
        results = self.fw_results[result_id]
        if type(results) is not tuple:
            results = (results,)

        payloads = []
        for out_idx, out_str in enumerate(node.outputs):
            for out_node_idx in node.output_map[out_str]:
                value = results[out_idx]
                if out_node_idx < 0:
                    if self.option.no_side_effect:
                        continue
                    out_type = (node.idx, out_idx)
                    out_idx = self.part_graph.outputs.index(out_type)

                    req_grad = value.requires_grad
                    self.runner.return_output(batch_id, out_idx, value, req_grad, None)
                    continue

                out_node = self.part_graph.nodes[out_node_idx]
                input_idx = out_node.inputs.index(out_str)
                out_rank = out_node.rank

                comm_id = self.get_comm_id(curr_id, out_node_idx, input_idx, batch_id)

                payloads.append((out_rank, comm_id, value))

        for end_rank, comm_id, value_send in payloads:

            def detach(v):
                if torch.is_tensor(v):
                    req_grad = v.requires_grad
                    v = v.to(device=self.workers[end_rank].device).detach()
                    if req_grad:
                        v = v.requires_grad_(True)
                return v

            if self.option.no_io_cache:
                value_send = list_map(detach, value_send)

                # self.debug(f"send {comm_id} to {end_rank}")
                if end_rank == curr_rank:
                    self.fw_inputs[comm_id] = value_send
                else:
                    self._send_fw(end_rank, comm_id, value_send)
            else:
                buffer = self.copy_act_buffers[comm_id]
                tensors = list_filter(filter_tensor, value_send)

                with torch.no_grad():
                    for recv_t, send_t in zip(buffer, tensors):
                        recv_t.copy_(send_t)

                send_values = list_map(to_size_only, value_send)
                self.workers[end_rank].recv_act_channel(comm_id, send_values)

    def send_act_async(self, node: PartitionNode, batch_id: int):
        event = torch.cuda.Event()
        event.record(self.fw_stream)
        curr_id = node.idx
        payloads = defaultdict(list)

        result_id = self.get_result_id(node.idx, batch_id)
        results = self.fw_results[result_id]

        if type(results) not in (tuple, list):
            results = (results,)

        if self.option.no_io_cache:
            for out_idx, out_str in enumerate(node.outputs):
                for out_node_idx in node.output_map[out_str]:
                    if out_node_idx < 0:
                        if self.option.no_side_effect:
                            continue

                        value = results[out_idx]
                        out_type = (node.idx, out_idx)
                        output_idx = self.part_graph.outputs.index(out_type)

                        req_grad = value.requires_grad
                        self._get_send_queue(-1).put(
                            (
                                event,
                                batch_id,
                                value,
                                (output_idx, req_grad),
                                SendType.RESULT,
                            )
                        )
                        continue

                    value = results[out_idx]
                    out_node = self.part_graph.nodes[out_node_idx]
                    input_idx = out_node.inputs.index(out_str)
                    out_rank = out_node.rank

                    comm_id = self.get_comm_id(
                        curr_id, out_node_idx, input_idx, batch_id
                    )

                    payloads[out_rank].append((out_rank, comm_id, value))

            for out_rank, payload in payloads.items():
                self._get_send_queue(out_rank).put((event, payload, SendType.FORWARD))
        else:
            for out_idx, out_str in enumerate(node.outputs):
                for out_node_idx in node.output_map[out_str]:
                    value = results[out_idx]

                    if out_node_idx < 0:
                        if self.option.no_side_effect:
                            continue
                        out_type = (node.idx, out_idx)
                        output_idx = self.part_graph.outputs.index(out_type)

                        req_grad = value.requires_grad
                        self._get_send_queue(-1).put(
                            (
                                event,
                                batch_id,
                                value,
                                (output_idx, req_grad),
                                SendType.RESULT,
                            )
                        )

                        continue

                    out_node = self.part_graph.nodes[out_node_idx]
                    input_idx = out_node.inputs.index(out_str)
                    out_rank = out_node.rank

                    comm_id = self.get_comm_id(
                        curr_id, out_node_idx, input_idx, batch_id
                    )
                    payloads[out_rank].append((out_rank, comm_id, value))

            for out_rank, payload in payloads.items():
                self._get_send_queue(out_rank).put((event, payload, SendType.FORWARD))

    def recv_act_async(self, node: PartitionNode, batch_id: int):
        curr_id = node.idx
        curr_rank = node.rank

        input_keys = []
        for edge_idx, in_str in enumerate(node.inputs):
            in_idx = node.input_map[in_str]
            comm_id = self.get_comm_id(in_idx, curr_id, edge_idx, batch_id)
            input_keys.append((in_idx, comm_id))

        for in_idx, key in input_keys:
            # send recv request
            if in_idx < 0:
                continue
            in_rank = self.part_graph.nodes[in_idx].rank
            if in_rank < 0:
                continue
            send_queue = self.workers[in_rank].send_queues[curr_rank]
            send_queue.put((key, True))

        for _, key in input_keys:
            while key not in self.fw_inputs:
                (recv_key, recv_value, send_event) = self.fw_queue.get()
                self.fw_events[recv_key] = send_event
                self.fw_inputs[recv_key] = recv_value

    def _send_bw(self, target_rank: int, key: str, value):
        target_worker = self.workers[target_rank]
        target_worker.bw_queue.put((key, value))

    def send_grad_sync(self, node: PartitionNode, batch_id: int):
        curr_rank = node.rank
        curr_id = node.idx

        if self.option.no_io_cache:
            payloads = []
            for input_idx, in_str in enumerate(node.inputs):
                in_node_idx = node.input_map[in_str]
                if in_node_idx < 0:
                    continue

                in_rank = self.part_graph.nodes[in_node_idx].rank
                comm_id = self.get_comm_id(in_node_idx, curr_id, input_idx, batch_id)

                value = self.fw_inputs[comm_id]
                grads = list_map(filter_grad, value)
                payloads.append((in_rank, comm_id, grads))

                # GC-out fw_inputs
                del self.fw_inputs[comm_id]

            for end_rank, comm_id, value_send in payloads:
                value_send = to_device(value_send, self.workers[end_rank].device)

                # self.debug(f"send {comm_id} to {end_rank}")
                if end_rank == curr_rank:
                    self.bw_grads[comm_id] = value_send
                else:
                    self._send_bw(end_rank, comm_id, value_send)
        else:
            for input_idx, in_str in enumerate(node.inputs):
                in_node_idx = node.input_map[in_str]
                if in_node_idx < 0:
                    continue

                in_rank = self.part_graph.nodes[in_node_idx].rank
                comm_id = self.get_comm_id(in_node_idx, curr_id, input_idx, batch_id)

                value = self.input_buffers[comm_id]
                grads = list_map(filter_grad, value)

                # GC-out fw_inputs
                del self.fw_inputs[comm_id]

                buffer = self.copy_grad_buffers[comm_id]
                tensors = list_map(filter_grad_buf, grads)

                with torch.no_grad():
                    for recv_t, send_t in zip(buffer, tensors):
                        recv_t.copy_(send_t)

                self.workers[in_rank].recv_grad_channel(comm_id)

    def send_grad_async(self, node: PartitionNode, batch_id: int):
        event = torch.cuda.Event()
        event.record(self.fw_stream)
        payloads = defaultdict(list)
        curr_id = node.idx

        if self.option.no_io_cache:
            for input_idx, in_str in enumerate(node.inputs):
                in_node_idx = node.input_map[in_str]
                if in_node_idx < 0:
                    continue

                in_rank = self.part_graph.nodes[in_node_idx].rank
                comm_id = self.get_comm_id(in_node_idx, curr_id, input_idx, batch_id)

                value = self.fw_inputs[comm_id]
                grads = list_map(filter_grad, value)
                payloads[in_rank].append((in_rank, comm_id, grads))

                # GC-out fw_inputs
                del self.fw_inputs[comm_id]

            for in_rank, payload in payloads.items():
                self._get_send_queue(in_rank).put((event, payload, SendType.GRAD))
        else:
            for input_idx, in_str in enumerate(node.inputs):
                in_node_idx = node.input_map[in_str]
                if in_node_idx < 0:
                    continue

                in_rank = self.part_graph.nodes[in_node_idx].rank
                comm_id = self.get_comm_id(in_node_idx, curr_id, input_idx, batch_id)

                value = self.input_buffers[comm_id]
                grads = list_map(filter_grad_buf, value)
                payloads[in_rank].append((in_rank, comm_id, grads))

                # GC-out fw_inputs
                del self.fw_inputs[comm_id]

            for in_rank, payload in payloads.items():
                self._get_send_queue(in_rank).put((event, payload, SendType.GRAD))

    def recv_grad_async(self, node: PartitionNode, batch_id: int):
        curr_id = node.idx
        curr_rank = node.rank

        grad_keys = []
        recv_ids = []
        for out_idx, out_str in enumerate(node.outputs):
            for out_node_idx in node.output_map[out_str]:
                if out_node_idx < 0:
                    if not self.option.no_side_effect:
                        grad_keys.append(
                            self.get_comm_id(curr_id, -1, out_idx, batch_id)
                        )
                    continue
                next_node = self.part_graph.nodes[out_node_idx]
                input_idx = next_node.inputs.index(out_str)

                comm_id = self.get_comm_id(curr_id, out_node_idx, input_idx, batch_id)
                grad_keys.append(comm_id)
                recv_ids.append((comm_id, out_node_idx))

        for key, out_idx in recv_ids:
            if out_idx < 0:
                continue
            out_rank = self.part_graph.nodes[out_idx].rank
            if out_rank < 0:
                continue
            out_queue = self.workers[out_rank].send_queues[curr_rank]
            out_queue.put((key, False))

        if self.option.no_io_cache:
            for key in grad_keys:
                while key not in self.bw_grads:
                    (recv_key, recv_value) = self.bw_queue.get()
                    self.bw_grads[recv_key] = recv_value
        else:
            for key in grad_keys:
                while key not in self.bw_grad_tokens:
                    recv_key, send_event = self.bw_queue.get()
                    self.bw_events[recv_key] = send_event
                    self.bw_grad_tokens[recv_key] = True

    def all_reduce_grad(self):
        if self.reducer_queues is None:
            return

        # grad_tensor = list(self.grad_tensors.values())[0]
        # print(self.rank, grad_tensor.grad[:5])

        for grad_id, grad_ranks in self.grad_ranks:
            grad_tensor = self.grad_tensors[grad_id]
            self.debug(
                f"trying all_reduce_grad {grad_id} {grad_ranks} {grad_tensor.grad.shape}"
            )
            err_msg = self.reducer.send_all_reduce(
                grad_id, grad_ranks, grad_tensor.grad
            )
            if err_msg is not None:
                self.error(err_msg)
                raise Exception(err_msg)

    def optimizer_step(self, node: PartitionNode, batch_id: int):
        torch.cuda.current_stream(self.device).synchronize()
        self.fw_stream.synchronize()
        self.all_reduce_grad()

        start_event = self.recorder.record_event(self.device)
        if self.optimizer:
            self.optimizer.step()
        end_event = self.recorder.record_event(self.device)

        if not self.option.cpu_test:
            torch.cuda.current_stream(self.device).synchronize()
            self.fw_stream.synchronize()

        self.recorder.append_fw(-(self.rank + 1), start_event, end_event)
        self.debug("optimizer end")

    def forward(self, node: PartitionNode, batch_id: int):
        curr_id = node.idx
        inputs = []

        def clone_value(t):
            if torch.is_tensor(t):
                return t.clone()
            else:
                return t

        for input_idx, in_str in enumerate(node.inputs):
            in_idx = node.input_map[in_str]
            comm_id = self.get_comm_id(in_idx, curr_id, input_idx, batch_id)

            value = self.fw_inputs[comm_id]
            if comm_id in self.fw_events:
                self.fw_events[comm_id].wait(self.fw_stream)

            with self.fw_ctx:
                inputs.append(list_map(clone_value, value))

        mod = self.mods[curr_id]

        if self.compiler is not None and curr_id not in self.mod_compiled:
            self.debug(f"compile submodule {curr_id} with {self.compiler}")
            mod = compile_module(self.compiler, mod, inputs)
            self.mods[curr_id] = mod
            self.mod_compiled.add(curr_id)

        start_event = self.recorder.record_event(self.device, self.fw_stream)

        with self.fw_ctx:
            result = mod(*inputs)
        del inputs

        is_end_node = node.idx == len(self.part_graph.nodes) - 1
        if self.option.no_side_effect and is_end_node:
            with self.fw_ctx:
                # result = result.sum()
                result = self.loss_fn(result, self.targets[batch_id])
                self.losses[batch_id] = result.item()

        end_event = self.recorder.record_event(self.device, self.fw_stream)
        self.recorder.append_fw(curr_id, start_event, end_event)

        result_id = self.get_result_id(node.idx, batch_id)

        self.fw_results[result_id] = result

    def backward(self, node: PartitionNode, batch_id: int):
        if self.option.no_io_cache:
            self.backward_no_buffer(node, batch_id)
        else:
            self.backward_buffer(node, batch_id)

    def backward_buffer(self, node: PartitionNode, batch_id: int):
        curr_id = node.idx

        grads_list = []

        for out_idx, out_str in enumerate(node.outputs):
            grads = None
            for out_node_idx in node.output_map[out_str]:
                if out_node_idx < 0:
                    if self.option.no_side_effect:
                        local_grads = [None]
                    else:
                        comm_id = self.get_comm_id(curr_id, -1, out_idx, batch_id)
                        local_grads = self.loss_grads[comm_id]

                        # gc-out
                        # TODO: check this really gc out
                        self.loss_grads[comm_id] = [None]
                else:
                    next_node = self.part_graph.nodes[out_node_idx]
                    input_idx = next_node.inputs.index(out_str)

                    comm_id = self.get_comm_id(
                        curr_id, out_node_idx, input_idx, batch_id
                    )

                    if comm_id in self.bw_events:
                        self.bw_events[comm_id].wait(self.fw_stream)
                    local_grads = self.grad_buffers[comm_id]

                    if len(local_grads) == 0:
                        local_grads = [None]

                if grads is None:
                    grads = local_grads
                elif local_grads is not None:
                    new_grads = []
                    for g, lg in zip(grads, local_grads):
                        if g is None:
                            new_grads.append(lg)
                        elif lg is None:
                            new_grads.append(g)
                        else:
                            g.add_(lg)
                            new_grads.append(g)
                    grads = new_grads

            if grads is None:
                grads = [None]

            grads_list.extend(grads)

        result_id = self.get_result_id(node.idx, batch_id)
        results = self.fw_results[result_id]

        # TODO: maybe not necessary if we can guarantee that the output is
        # always a tuple of primitive values or tensors
        if len(node.outputs) == 1:
            results = [results]

        # tensors = []
        # for r in results:
        #     tensors.extend(list_filter(collect_req_grads, r))

        # print(len(results), len(tensors), len(grads_list), curr_id, batch_id)

        if len(results) != len(grads_list):
            self.error(
                f"grad count mismatch {len(results)} != {len(grads_list)} / {curr_id} - {batch_id}"
            )

        fin_tensor = []
        fin_grads = []

        # TODO: check that this is true
        for t, g in zip(results, grads_list):
            if not (torch.is_tensor(t) and t.requires_grad):
                continue
            # filter non-scalar without gradient
            if g is None and t.ndim > 0:
                continue
            fin_tensor.append(t)
            fin_grads.append(g)

        if len(fin_tensor) != len(fin_grads):
            self.error(
                f"require_grads grad count mismatch {len(fin_tensor)} != {len(fin_grads)} / {curr_id} - {batch_id}"
            )

        start_event = self.recorder.record_event(self.device, self.fw_stream)
        with self.fw_ctx:
            torch.autograd.backward(fin_tensor, fin_grads, retain_graph=False)
        end_event = self.recorder.record_event(self.device, self.fw_stream)

        self.recorder.append_bw(curr_id, start_event, end_event)

        # GC-out forward / backward results
        del self.fw_results[result_id]

    def backward_no_buffer(self, node: PartitionNode, batch_id: int):
        curr_id = node.idx

        grads = []
        grad_ids = []

        for out_idx, out_str in enumerate(node.outputs):
            grad = None
            for out_node_idx in node.output_map[out_str]:
                if out_node_idx < 0:
                    if self.option.no_side_effect:
                        local_grad = None
                    else:
                        comm_id = self.get_comm_id(curr_id, -1, out_idx, batch_id)
                        local_grad = self.loss_grads[comm_id][0]

                        # gc-out
                        self.loss_grads[comm_id] = None
                else:
                    next_node = self.part_graph.nodes[out_node_idx]
                    input_idx = next_node.inputs.index(out_str)

                    comm_id = self.get_comm_id(
                        curr_id, out_node_idx, input_idx, batch_id
                    )
                    local_grad = self.bw_grads[comm_id]
                    grad_ids.append(comm_id)

                if grad is None:
                    grad = local_grad
                elif local_grad is not None:
                    grad += local_grad

            grads.append(grad)

        result_id = self.get_result_id(node.idx, batch_id)
        results = self.fw_results[result_id]

        # TODO: maybe not necessary if we can guarantee that the output is
        # always a tuple of primitive values or tensors
        if len(node.outputs) == 1:
            results = [results]

        results = list_map(collect_req_grads, results)

        r_filt = []
        g_filt = []
        for r, g in zip(results, grads):
            if r is not None and (g is not None or r.ndim == 0):
                r_filt.append(r)
                g_filt.append(g)

        start_event = self.recorder.record_event(self.device, self.fw_stream)
        with self.fw_ctx:
            torch.autograd.backward(r_filt, g_filt, retain_graph=False)
        end_event = self.recorder.record_event(self.device, self.fw_stream)

        self.recorder.append_bw(curr_id, start_event, end_event)

        # GC-out forward / backward results
        del self.fw_results[result_id]
        for comm_id in grad_ids:
            del self.bw_grads[comm_id]
