import pickle
import logging
from copy import deepcopy
from enum import Enum
from typing import Dict, List
from queue import Queue
from collections import defaultdict, deque
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ..utils import fmem, list_params
from ..option import PipeOption, get_optimizer_cls
from ..schedule.scheduler import Step, StepWork, Scheduler, SingleCommScheduler
from ..schedule.planner import SchedCommNode, DependencyGraph
from ..graph.computation_graph import ParamSet
from ..graph.partition_graph import PartitionGraph, PartitionNode, module_fetch_attr
from ..device_bench import NCCLDeviceBench
from ..utils import get_logger, to_device, to_device_graphmodule
from ..timing import CPUTimeRecorder, CUDATimeRecorder
from ..compile import compile_module

from .net_states import get_state


FILTER_GRAD_NONE = "<<FILTER_GRAD_NONE>>"


class LoopFlag(Enum):
    STOP_LOOP = 0
    CONTINUE_LOOP = 1
    GATHER_PARAMS = 2
    LOG_MEMORY_USAGE = 3


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
            return torch.zeros_like(t)
        return t.grad


def collect_req_grads(v):
    if torch.is_tensor(v) and v.requires_grad:
        return v
    return None


def to_size_only(t):
    return SizeOnly() if torch.is_tensor(t) else t


class ProcessPipe:
    # Pipe connection between pp ranks
    def __init__(self, device, pp_group_ranks, group):
        self.device = device
        self.group = group
        self.pp_group_ranks = pp_group_ranks
        self.load_tensor = torch.zeros(1, device=device, dtype=torch.long)

    def send(self, pp_rank, data):
        pkl_bytes = torch.tensor(
            np.frombuffer(pickle.dumps(data), dtype=np.uint8),
            device=self.device,
            dtype=torch.uint8,
        )
        tensor_size = pkl_bytes.numel()

        self.load_tensor[0] = tensor_size
        target_rank = self.pp_group_ranks[pp_rank]

        dist.send(self.load_tensor, target_rank, self.group)
        dist.send(pkl_bytes, target_rank, self.group)

    def recv(self, pp_rank):
        recv_rank = self.pp_group_ranks[pp_rank]
        self.load_tensor.zero_()
        dist.recv(self.load_tensor, recv_rank, self.group)
        tensor_size = self.load_tensor[0]
        pkl_bytes = torch.zeros(tensor_size, dtype=torch.uint8, device=self.device)
        dist.recv(pkl_bytes, recv_rank, self.group)
        return pickle.loads(pkl_bytes.cpu().numpy().tobytes())


class WorkerScaler:
    def __init__(self, option):
        self.scaler = torch.GradScaler() if option.mixed_precision else None

    def step(self, optimizer):
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def scale(self, inputs):
        return self.scaler.scale(inputs) if self.scaler is not None else inputs


class ProcessWorker:
    def __init__(self, runner=None):
        self.net_state = get_state()

        self.pp_rank = self.net_state.pp_rank
        self.pp_size = self.net_state.pp_size
        self.global_rank = self.net_state.rank
        self.world_size = self.net_state.world_size

        self.device = self.net_state.device
        self.pp_group = self.net_state.pp_group
        self.pipe = ProcessPipe(
            self.device, self.net_state.pp_group_ranks, self.pp_group
        )

        self.runner = runner
        self.is_main_proc = self.runner is not None

        self.logger = get_logger()

    def wait_for_init(self):
        self.run_device_bench()  # set self.option

        option = self.option

        # sent from distribute_graph
        pg, sched, optimizer, optim_dict, param_set = self.pipe.recv(self.pp_size - 1)

        if isinstance(sched, SingleCommScheduler):
            sched.dep_graph = DependencyGraph.deserialize(*sched.dep_graph)

        self.debug("worker process wait for init")

        self.set_option(option)
        self.set_partition_graph(pg, optimizer, optim_dict)
        self.set_scheduler_steps(sched)

        self.init_reducer(param_set)

        # acknowledgement to the main process
        self.pipe.send(self.pp_size - 1, "ack")

        self.debug("worker process initialized")

    def mem_usage_str(self):
        idle_mem = self.mem_usage_info.get("idle", -1)
        max_mem = self.mem_usage_info.get("max", -1)
        reserved_mem = self.mem_usage_info.get("reserved", -1)
        return f"idle: {fmem(idle_mem)}, max: {fmem(max_mem)}, alloc: {fmem(max_mem - idle_mem)}, reserved: {fmem(reserved_mem)}"

    def log_memory_usage(self):
        self.info(f"memory usage: {self.mem_usage_str()}")

    def run_loop(self):
        while True:
            torch.cuda.reset_peak_memory_stats(self.device)
            prev_max = torch.cuda.max_memory_allocated(self.device)

            do_loop = self.prepare_workers()
            if not do_loop:
                self.log_memory_usage()
                break

            self.reset_cache()
            self.load_input()
            self.run()

            dev_max = torch.cuda.max_memory_allocated(self.device)
            dev_resv = torch.cuda.max_memory_reserved(self.device)

            self.mem_usage_info["idle"] = prev_max
            self.mem_usage_info["max"] = dev_max
            self.mem_usage_info["reserved"] = dev_resv

    def run_warmup(self):
        for pp_rank in range(self.pp_size):
            if pp_rank < self.pp_rank:
                self.pipe.send(pp_rank, "ping")
            elif self.pp_rank < pp_rank:
                pong = self.pipe.recv(pp_rank)
                that_rank = self.net_state.pp_group_ranks[pp_rank]
                self.debug(f"recv {pong} {self.global_rank} <- {that_rank}")

        for pp_rank in range(self.pp_size):
            if pp_rank < self.pp_rank:
                pong = self.pipe.recv(pp_rank)
                that_rank = self.net_state.pp_group_ranks[pp_rank]
                self.debug(f"recv {pong} {that_rank} -> {self.global_rank}")
            elif self.pp_rank < pp_rank:
                self.pipe.send(pp_rank, "ping-back")

        dist.barrier()
        self.debug("barrier end")

    def run_device_bench(self, option=None):
        if option is None:
            option = self.pipe.recv(self.pp_size - 1)
            option.set_logger()
            self.option = option
        else:
            for pp_rank in range(self.pp_size):
                if pp_rank != self.pp_rank:
                    self.pipe.send(pp_rank, option)

        self.run_warmup()

        if self.option.device_bench is not None:
            # load device bench from file
            device_bench = NCCLDeviceBench(
                [i for i in range(self.pp_size)],
                rank=self.pp_rank,
                curr_device=self.device,
            )

            device_bench.run_bench(file_path=self.option.device_bench)
            return device_bench

    def _remove_param_from_optim(self, optimizer):
        param_list = [pg["params"] for pg in optimizer.param_groups]
        for pg in optimizer.param_groups:
            pg["params"] = []

        copy_optim = deepcopy(optimizer)

        for pg, pl in zip(optimizer.param_groups, param_list):
            pg["params"] = pl

        return copy_optim

    def distribute_graph(
        self,
        rank_pgs: Dict[int, PartitionGraph],
        sched: Scheduler,
        optimizer,
        optim_dict,
        param_set,
    ):
        """
        The main process is on `world_size - 1` rank.
        Inputs are first moved to the rank 0 process and than sequentially
        executes the forward pass.
        """
        this_pg = None

        copy_optim = self._remove_param_from_optim(optimizer)

        if isinstance(sched, SingleCommScheduler):
            dep_graph = sched.dep_graph
            sched.dep_graph = sched.dep_graph.serialize()

        # Q: this assumes that each pp_rank is unique. is it true?
        # A: yes, from split_rank_gm
        for pp_rank, pg in rank_pgs.items():
            if pp_rank != self.pp_rank:
                self.pipe.send(pp_rank, (pg, sched, copy_optim, optim_dict, param_set))
            else:
                this_pg = pg

        if isinstance(sched, SingleCommScheduler):
            sched.dep_graph = dep_graph

        self.set_partition_graph(this_pg, optimizer, optim_dict)
        self.init_reducer(param_set)

        # acknowledgement from the worker processes
        for pp_rank in range(self.pp_size - 1):
            ack = self.pipe.recv(pp_rank)

    def init_reducer(self, param_set):
        self.grad_ranks = None
        self.grad_tensors = None

        if len(param_set.shared_idx) == 0:
            return

        net_state = self.net_state
        pp_main_ranks = net_state.get_dp_group_ranks(net_state.pp_main_rank)

        groups_all = dict()
        grad_ranks_all = dict()
        for idx, ranks in param_set.shared_rank.items():
            for pp_main_rank in pp_main_ranks:
                ranks = tuple(
                    sorted([net_state.get_global_rank(r, pp_main_rank) for r in ranks])
                )
                groups_all[ranks] = dist.new_group(ranks)
                if pp_main_rank == net_state.pp_main_rank:
                    grad_ranks_all[idx] = ranks

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
            if rank != self.pp_rank:
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
        self.grad_tensors = grad_tensors
        self.groups_all = groups_all

        self.debug(
            f"found shared params {self.grad_ranks} / {len(self.grad_tensors)} tensors"
        )

        return grad_tensors

    def all_reduce_grad(self):
        if self.grad_tensors is None:
            return

        global_rank = self.net_state.rank

        for grad_id, grad_ranks in self.grad_ranks:
            if global_rank not in grad_ranks:
                continue

            grad_tensor = self.grad_tensors[grad_id]
            self.debug(
                f"trying all_reduce_grad {grad_id} {grad_ranks} {grad_tensor.grad.shape}"
            )

            proc_group = self.groups_all[grad_ranks]
            dist.all_reduce(grad_tensor.grad, group=proc_group)

    def destroy_workers(self):
        # is main process
        if self.runner is not None:
            self.close_workers()
        self.net_state.destroy()
        self.runner = None

    def close_workers(self):
        self.log_memory_usage()
        self.input_check_tensor.fill_(LoopFlag.STOP_LOOP.value)
        dist.broadcast(
            self.input_check_tensor, self.net_state.pp_main_rank, group=self.pp_group
        )

    def prepare_workers(self, set_flag=None):
        while True:
            self._broadcast_loop_flag(set_flag)

            loop_flag = self.input_check_tensor.item()
            if loop_flag == LoopFlag.STOP_LOOP.value:  # stop
                return False
            elif loop_flag == LoopFlag.GATHER_PARAMS.value:  # continue
                self.gather_params()
            elif loop_flag == LoopFlag.LOG_MEMORY_USAGE.value:
                self.all_log_memory_usage()
            else:
                return True

    def _broadcast_loop_flag(self, flag):
        if flag is not None:
            self.input_check_tensor.fill_(flag.value)
        dist.broadcast(
            self.input_check_tensor,
            self.net_state.pp_main_rank,
            group=self.pp_group,
        )

    def gather_params(self):
        master_rank = self.pp_size - 1
        param_dict = dict()

        if self.pp_rank == master_rank:
            self._broadcast_loop_flag(LoopFlag.GATHER_PARAMS)
            for rank in self.net_state.pp_group_ranks:
                if rank == master_rank:
                    continue
                params = self.pipe.recv(rank)
                param_dict.update(params)

            return param_dict

        for idx, part_node in self.part_graph.nodes.items():
            if part_node.rank != self.pp_rank:
                continue
            params = [p.data for p in part_node.graph_module.parameters()]
            param_dict[idx] = params

        self.pipe.send(master_rank, param_dict)

    def all_log_memory_usage(self):
        master_rank = self.pp_size - 1
        if self.pp_rank == master_rank:
            self._broadcast_loop_flag(LoopFlag.LOG_MEMORY_USAGE)

        self.log_memory_usage()

    def broadcast_comp_graph(self, option: PipeOption, is_master: bool):
        filesize_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
        if is_master:
            pkl_bytes = torch.tensor(
                np.frombuffer(pickle.dumps(option.graph_bench), dtype=np.uint8),
                device=self.device,
            )
            tensor_size = pkl_bytes.numel()
            filesize_tensor[0] = tensor_size

        dist.broadcast(
            filesize_tensor,
            self.net_state.world_size - 1,
            group=self.net_state.dp_group,
        )

        if is_master:
            dist.broadcast(
                pkl_bytes, self.net_state.world_size - 1, group=self.net_state.dp_group
            )
        else:
            filename_tensor = torch.zeros(
                filesize_tensor.item(), dtype=torch.uint8, device=self.device
            )
            dist.broadcast(
                filename_tensor,
                self.net_state.world_size - 1,
                group=self.net_state.dp_group,
            )
            option.graph_bench = pickle.loads(filename_tensor.cpu().numpy().tobytes())

    def set_option(self, option: PipeOption):
        self.option = option
        self.compiler = option.compiler

        self.part_graph: PartitionGraph = None
        self.mods: Dict[int, nn.Module] = dict()
        self.mod_compiled = set()
        self.params = []
        self.optimizer = None
        self.scaler = WorkerScaler(option)

        self.is_train = True
        self.steps = None

        self.optimizer_cls = get_optimizer_cls(option.optimizer_type)

        self.steps = []
        self.comm_steps: List[List[SchedCommNode]] = [[] for _ in range(self.pp_size)]
        self.comm_steps_pending = []

        self.fw_queue = Queue()
        self.bw_queue = Queue()

        # self.main_stream = torch.cuda.Stream(device=self.device)
        self.main_stream = torch.cuda.default_stream(self.device)
        self.input_check_tensor = torch.zeros(
            LoopFlag.CONTINUE_LOOP.value, device=self.device, dtype=torch.uint8
        )

        # Real inputs and outputs
        self.fw_inputs = dict()
        self.fw_results = dict()
        self.bw_grads = dict()
        self.bw_tokens = set()
        self.loss_grads = dict()

        self.completed_jobs = set()
        self.act_irecv = defaultdict(list)
        self.act_isend = defaultdict(list)
        self.grad_isend = defaultdict(list)
        self.grad_irecv = defaultdict(list)
        self.recv_step_last = dict()

        # IO Tensor buffers
        self.input_buffers = dict()
        self.grad_buffers = dict()

        self.mem_usage_info = dict()
        self.recorder = CUDATimeRecorder()

        self.reset_cache()

    def reset_cache(self):
        self.fw_inputs.clear()
        self.fw_results.clear()
        self.bw_grads.clear()
        self.bw_tokens.clear()
        self.loss_grads.clear()
        self.completed_jobs.clear()

        self.recv_thres = 0

        for send in self.act_isend.values():
            for ev in send:
                ev.wait()
        for send in self.grad_isend.values():
            for ev in send:
                ev.wait()

        self.act_isend.clear()
        self.act_irecv.clear()
        self.grad_isend.clear()
        self.grad_irecv.clear()

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

        for buffers in self.input_buffers.values():
            for tensor in buffers:
                if tensor.requires_grad:
                    tensor.grad = None

    def _init_scheduler(self):
        if self.dep_graph is None:
            return

        self.comm_steps_pending = [deque(p) for p in self.comm_steps]

    def debug(self, msg):
        self.logger.debug("[Worker %s] %s", self.global_rank, msg)

    def info(self, msg):
        self.logger.info("[Worker %s] %s", self.global_rank, msg)

    def error(self, msg):
        self.logger.error("[Worker %s] %s", self.global_rank, msg)

    def get_event_logs(self):
        return self.recorder.get_event_logs()

    def parameters(self):
        return self.params

    def test_param_and_grad(self):
        return (self.params[0], self.params[0].grad)

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

        self.grad_tensors = grad_tensors

        return grad_tensors

    def get_mp_ctx(self):
        return (
            torch.autocast(dtype=self.option.mixed_precision, device_type="cuda")
            if self.option.mixed_precision
            else nullcontext()
        )

    def set_partition_graph(
        self, part_graph: PartitionGraph, optimizer=None, optim_dict=None
    ):
        self.part_graph = part_graph

        batch_cnt = self.option.batch_cnt

        for idx, node in part_graph.nodes.items():
            if node.rank != self.pp_rank:
                continue

            module = to_device_graphmodule(node.graph_module, self.device)
            module.train()
            self.mods[node.idx] = module

            if optimizer is None:
                self.params.extend(list(module.parameters()))
                continue

            for param_id, param in enumerate(module.parameters()):
                if (idx, param_id) in optim_dict:
                    opt_id = optim_dict[(idx, param_id)]
                    param.grad = None
                    optimizer.param_groups[opt_id]["params"].append(param)

        self.optimizer = optimizer

        for node in part_graph.nodes.values():
            if node.rank != self.pp_rank:
                continue

            # fill in input activation buffers and send it to input nodes
            for input_idx, in_str in enumerate(node.inputs):
                in_node_idx = node.input_map[in_str]

                # skip main proc
                if in_node_idx < 0 and self.is_main_proc:
                    continue

                for batch_id in range(batch_cnt):
                    comm_id = self.get_comm_id(
                        in_node_idx, node.idx, input_idx, batch_id
                    )

                    tensors = []
                    for size, dtype, _, req_grad in node.input_shapes[input_idx]:
                        t = torch.zeros(
                            size,
                            dtype=dtype,
                            device=self.device,
                            requires_grad=req_grad,
                        )
                        tensors.append(t)

                    self.input_buffers[comm_id] = tensors

            # fill in output grad buffers and send it to input nodes
            for out_idx, out_str in enumerate(node.outputs):
                for out_node_idx in node.output_map[out_str]:
                    if out_node_idx < 0:
                        continue

                    next_node = part_graph.nodes[out_node_idx]
                    input_idx = next_node.inputs.index(out_str)

                    for batch_id in range(batch_cnt):
                        comm_id = self.get_comm_id(
                            node.idx, out_node_idx, input_idx, batch_id
                        )

                        tensors = []
                        for size, dtype, _, req_grad in node.output_shapes[out_idx]:
                            if req_grad:
                                t = torch.zeros(size, dtype=dtype, device=self.device)
                                tensors.append(t)

                        self.grad_buffers[comm_id] = tensors

    def set_scheduler_steps(self, sched: Scheduler):
        steps = sched.steps[self.pp_rank]

        self.sched = sched
        self.steps = steps
        self.recv_step_last = dict()

        for i, step in enumerate(steps):
            self.debug(f"set step {i}: {step}")

        if isinstance(sched, SingleCommScheduler):
            self.dep_graph: DependencyGraph = sched.dep_graph
            for rank in range(sched.device_cnt):
                if rank == self.pp_rank:
                    continue
                rank_pair = (
                    (rank, self.pp_rank)
                    if rank <= self.pp_rank
                    else (self.pp_rank, rank)
                )
                comm_steps = self.dep_graph.channel_comms[rank_pair]
                self.comm_steps[rank] = comm_steps

                step_last = defaultdict(int)
                self.recv_step_last[rank] = step_last

                len_comm = len(comm_steps)
                for comm_idx, comm_node in enumerate(comm_steps):
                    dst_comp = comm_node.dst_comp
                    if dst_comp.rank != self.pp_rank:
                        continue
                    step_last[
                        (dst_comp.node_id, dst_comp.batch_id, dst_comp.is_forward)
                    ] = (len_comm - comm_idx)

                if rank > self.pp_rank:
                    for i, step in enumerate(comm_steps):
                        self.debug(
                            f"set comm step ({self.pp_rank} {rank}) {i}: {step} => {step.text_next()}"
                        )

        else:
            self.dep_graph = None

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

    def distribute_input(self, node: PartitionNode, batch_id, input_id, value):
        self.debug(f"set input for batch {batch_id} / input_id: {input_id}")

        value = to_device(value, self.device)
        comm_id = self.get_comm_id(-1, node.idx, input_id, batch_id)
        rank = node.rank

        target_rank = (
            self.net_state.pp_group_ranks[rank]
            if rank != self.pp_rank
            else self.pp_rank
        )
        self.debug(
            f"distribute input: ({self.pp_rank} -> {target_rank}) {comm_id} {type(value)}"
        )

        if rank != self.pp_rank:
            ev = dist.isend(
                value.contiguous(), target_rank, group=self.net_state.pp_group
            )
            self.act_isend[comm_id].append(ev)
        else:
            self.fw_inputs[comm_id] = value

    def load_input(self):
        ev_list = []
        for batch_id in range(self.option.batch_cnt):
            for usage in self.part_graph.inputs:
                for part_id, input_idx in usage:
                    node = self.part_graph.nodes[part_id]
                    if node.rank != self.pp_rank:
                        continue

                    comm_id = self.get_comm_id(-1, node.idx, input_idx, batch_id)
                    input_buffer = self.input_buffers[comm_id]

                    for tensor in input_buffer:
                        ev = dist.irecv(
                            tensor,
                            self.net_state.pp_main_rank,
                            group=self.net_state.pp_group,
                        )
                        self.act_irecv[comm_id].append((ev, comm_id))
                        ev_list.append(ev)

                    self.debug(f"load_input {comm_id} {len(input_buffer)}")

                    if len(input_buffer) == 0:
                        self.fw_inputs[comm_id] = None
                    elif len(input_buffer) == 1:
                        self.fw_inputs[comm_id] = input_buffer[0]
                    else:
                        self.fw_inputs[comm_id] = input_buffer

        for ev in ev_list:
            ev.wait()

    def _preconfig_recv(self):
        for job in self.steps:
            node = self.part_graph.nodes[job.node_id] if job.node_id >= 0 else None
            if job.work == StepWork.RECV_ACT:
                self.recv_act_async(node, job.batch_id)
            elif job.work == StepWork.RECV_GRAD:
                self.recv_grad_async(node, job.batch_id)

    def set_target(self, batch_id, target):
        self.targets[batch_id] = target

    def run(self):
        self._init_optimizer(set_to_none=False)
        self._init_scheduler()

        # self.main_stream.synchronize()

        torch.cuda.reset_peak_memory_stats(self.device)
        prev_max = torch.cuda.max_memory_allocated(self.device)

        if self.dep_graph is None:
            for job in self.steps:
                self._handle_job(job)
        else:
            for job in self.steps:
                self._handle_job_with_comm(job)

        if self.runner is not None:
            self.runner.pipe_done()

        dev_max = torch.cuda.max_memory_allocated(self.device)
        dev_resv = torch.cuda.max_memory_reserved(self.device)

        self.mem_usage_info["idle"] = prev_max
        self.mem_usage_info["max"] = dev_max
        self.mem_usage_info["reserved"] = dev_resv

    def run_with_module(self, pipe_idx):
        self._init_optimizer(set_to_none=False)
        self._init_scheduler()

        # self.main_stream.synchronize()

        torch.cuda.reset_peak_memory_stats(self.device)
        prev_max = torch.cuda.max_memory_allocated(self.device)

        job_fn = (
            self._handle_job if self.dep_graph is None else self._handle_job_with_comm
        )
        last_send = [0 for _ in range(self.option.batch_cnt)]

        for job_id, job in enumerate(self.steps):
            if job.work == StepWork.SEND_ACT:
                last_send[job.batch_id] = job_id  # mark last send_act

        for job_id, job in enumerate(self.steps):
            job_fn(job)
            if job.work == StepWork.SEND_ACT and job_id == last_send[job.batch_id]:
                self.runner.finalize_output(pipe_idx, job.batch_id)

        dev_max = torch.cuda.max_memory_allocated(self.device)
        dev_resv = torch.cuda.max_memory_reserved(self.device)

        self.mem_usage_info["idle"] = prev_max
        self.mem_usage_info["max"] = dev_max
        self.mem_usage_info["reserved"] = dev_resv

    def recv_loss_grad(
        self, rank: int, batch_idx: int, node_id: int, output_idx: int, grad
    ):
        comm_id = self.get_comm_id(node_id, -1, output_idx, batch_idx)
        # print(
        #     f"recv loss grad {comm_id} {rank} {batch_idx} {node_id} {output_idx} {grad.shape if grad is not None else None}"
        # )
        self.loss_grads[comm_id] = grad
        self.bw_queue.put(comm_id)

    def _handle_job(self, job: Step):
        self.debug(job)

        nvtx.range_push(f"DEV{self.pp_rank}_{job}")

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

        nvtx.range_pop()

    def _handle_job_with_comm(self, job: Step):
        self.debug(job)

        nvtx.range_push(f"DEV{self.global_rank}_{job}")

        node = None
        if job.node_id >= 0:
            node = self.part_graph.nodes[job.node_id]

        if job.work == StepWork.SEND_ACT or job.work == StepWork.SEND_GRAD:
            self.send_comm()
            self.send_same_node(node, job.batch_id, job.work == StepWork.SEND_ACT)
            if job.work == StepWork.SEND_ACT:
                self.send_result_comm(node, job.batch_id)
        elif job.work == StepWork.RECV_ACT:
            self.recv_comm(node, job.batch_id, True)
        elif job.work == StepWork.RECV_GRAD:
            self.recv_comm(node, job.batch_id, False)
            self.recv_grad_comm(node, job.batch_id)
        elif job.work == StepWork.FORWARD:
            self.forward(node, job.batch_id)
            self.completed_jobs.add((node.idx, job.batch_id, True))
        elif job.work == StepWork.BACKWARD:
            self.backward(node, job.batch_id)
            self.completed_jobs.add((node.idx, job.batch_id, False))
        elif job.work == StepWork.OPTIMIZER_STEP:
            self.optimizer_step(node, job.batch_id)

        nvtx.range_pop()

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

    def send_act_async(self, node: PartitionNode, batch_id: int, dst_comp=None):
        curr_id = node.idx
        result_id = self.get_result_id(node.idx, batch_id)
        results = self.fw_results[result_id]

        if type(results) not in (tuple, list):
            results = (results,)

        payloads = []
        for out_idx, out_str in enumerate(node.outputs):
            for out_node_idx in node.output_map[out_str]:
                value = results[out_idx]

                if out_node_idx < 0:
                    out_type = (node.idx, out_idx)
                    output_idx = self.part_graph.outputs.index(out_type)
                    req_grad = value.requires_grad

                    if dst_comp is not None and self.runner is not None:
                        self.runner.return_output(
                            batch_id, output_idx, value, req_grad, node.idx
                        )

                    continue

                out_node = self.part_graph.nodes[out_node_idx]
                input_idx = out_node.inputs.index(out_str)
                out_rank = out_node.rank

                if dst_comp is not None and out_node.idx != dst_comp.node_id:
                    continue

                comm_id = self.get_comm_id(curr_id, out_node_idx, input_idx, batch_id)
                tensors = list_filter(filter_tensor, value)

                for send_t in tensors:
                    # add sort info for alignment
                    payloads.append(
                        (out_rank, out_node_idx, input_idx, curr_id, comm_id, send_t)
                    )

        payloads.sort()
        for out_rank, _, input_idx, curr_id, comm_id, send_t in payloads:
            global_out_rank = self.net_state.pp_group_ranks[out_rank]

            # self.info(
            #     f"send act {node.idx} {batch_id} - {out_rank}/{global_out_rank}: {comm_id} {send_t.shape} {send_t.dtype} {send_t.sum().item()}"
            # )

            ev = dist.isend(
                send_t.contiguous(), global_out_rank, group=self.net_state.pp_group
            )
            self.act_isend[comm_id].append(ev)

    def send_result_comm(self, node: PartitionNode, batch_id: int):
        result_id = self.get_result_id(node.idx, batch_id)
        results = self.fw_results[result_id]

        if type(results) not in (tuple, list):
            results = (results,)

        for out_idx, out_str in enumerate(node.outputs):
            for out_node_idx in node.output_map[out_str]:
                value = results[out_idx]

                # send result
                if out_node_idx < 0:
                    out_type = (node.idx, out_idx)
                    output_idx = self.part_graph.outputs.index(out_type)
                    req_grad = value.requires_grad

                    if self.runner is not None:
                        self.runner.return_output(
                            batch_id, output_idx, value, req_grad, node.idx
                        )

    def send_same_node(self, node: PartitionNode, batch_id: int, is_fwd: bool):
        curr_id = node.idx
        if is_fwd:
            result_id = self.get_result_id(node.idx, batch_id)
            results = self.fw_results[result_id]
            for out_idx, out_str in enumerate(node.outputs):
                for out_node_idx in node.output_map[out_str]:
                    if out_node_idx < 0:
                        continue

                    # send same rank result
                    out_node = self.part_graph.nodes[out_node_idx]
                    input_idx = out_node.inputs.index(out_str)
                    out_rank = out_node.rank
                    if out_rank != self.pp_rank:
                        continue

                    value = results[out_idx]
                    comm_id = self.get_comm_id(
                        node.idx, out_node_idx, input_idx, batch_id
                    )
                    tensors = list_filter(filter_tensor, value)
                    input_buffers = self.input_buffers[comm_id]
                    with torch.no_grad():
                        for recv_t, send_t in zip(input_buffers, tensors):
                            recv_t.copy_(send_t)

                    if len(input_buffers) == 0:
                        self.fw_inputs[comm_id] = None
                    elif len(input_buffers) == 1:
                        self.fw_inputs[comm_id] = input_buffers[0]
                    else:
                        self.fw_inputs[comm_id] = tuple(input_buffers)
        else:
            for input_idx, in_str in enumerate(node.inputs):
                in_node_idx = node.input_map[in_str]
                if in_node_idx < 0:
                    continue

                in_node = self.part_graph.nodes[in_node_idx]
                in_rank = in_node.rank
                if in_rank != self.pp_rank:
                    continue

                comm_id = self.get_comm_id(in_node_idx, curr_id, input_idx, batch_id)

                value = self.input_buffers[comm_id]
                grads = list_map(filter_grad_buf, value)

                # GC-out fw_inputs
                del self.fw_inputs[comm_id]

                grad_buf = self.grad_buffers[comm_id]
                with torch.no_grad():
                    for recv_t, send_t in zip(grad_buf, grads):
                        recv_t.copy_(send_t)

    def recv_act_async(self, node: PartitionNode, batch_id: int, src_comp=None):
        curr_id = node.idx
        ev_list = []
        buffer_list = []

        input_keys = []
        for edge_idx, in_str in enumerate(node.inputs):
            in_idx = node.input_map[in_str]
            comm_id = self.get_comm_id(in_idx, curr_id, edge_idx, batch_id)
            input_keys.append((edge_idx, in_idx, comm_id))

        input_keys.sort()

        for edge_idx, in_idx, comm_id in input_keys:
            # send recv request
            if in_idx < 0:
                continue
            else:
                in_node = self.part_graph.nodes[in_idx]
                in_rank = in_node.rank

            if in_rank == self.pp_rank:
                continue

            if src_comp is not None and in_node.idx != src_comp.node_id:
                continue

            input_buffers = self.input_buffers[comm_id]

            for tensor in input_buffers:
                global_in_rank = self.net_state.pp_group_ranks[in_rank]
                ev = dist.irecv(tensor, global_in_rank, group=self.net_state.pp_group)
                self.act_irecv[(node.idx, batch_id)].append((ev, comm_id))
                ev_list.append(ev)

            buffer_list.append((comm_id, input_buffers))

        if src_comp is None:
            for ev in ev_list:
                ev.wait()

        for comm_id, input_buffers in buffer_list:

            if len(input_buffers) == 0:
                self.fw_inputs[comm_id] = None
            elif len(input_buffers) == 1:
                self.fw_inputs[comm_id] = input_buffers[0]
            else:
                self.fw_inputs[comm_id] = tuple(input_buffers)

        return ev_list

    def send_grad_async(self, node: PartitionNode, batch_id: int, dst_comp=None):
        event = torch.cuda.Event()
        event.record(self.main_stream)

        payloads = []
        curr_id = node.idx

        for input_idx, in_str in enumerate(node.inputs):
            in_node_idx = node.input_map[in_str]
            if in_node_idx < 0:
                continue

            in_node = self.part_graph.nodes[in_node_idx]
            in_rank = in_node.rank
            comm_id = self.get_comm_id(in_node_idx, curr_id, input_idx, batch_id)

            if dst_comp is not None:
                if in_node.idx != dst_comp.node_id:
                    continue

            value = self.input_buffers[comm_id]
            grads = list_map(filter_grad_buf, value)

            payloads.append((in_rank, in_node_idx, input_idx, comm_id, grads))

            # GC-out fw_inputs
            del self.fw_inputs[comm_id]

        payloads.sort()

        for in_rank, _, _, comm_id, grads in payloads:
            for tensor in grads:
                if tensor is None:
                    # skip non req_grad / if it requires, its grad will be a zero tensor from filter_grad_buf
                    continue

                global_in_rank = self.net_state.pp_group_ranks[in_rank]
                ev = dist.isend(tensor, global_in_rank, group=self.net_state.pp_group)
                self.grad_isend[comm_id].append(ev)

    def recv_grad_async(self, node: PartitionNode, batch_id: int, src_comp=None):
        curr_id = node.idx

        grad_keys = []
        recv_ids = []
        for out_idx, out_str in enumerate(node.outputs):
            for out_node_idx in node.output_map[out_str]:
                if out_node_idx < 0:
                    out_node_idx = -1
                    input_idx = out_idx

                    # TODO: check skip other than last node
                    if src_comp is not None and self.runner is None:
                        continue
                else:
                    next_node = self.part_graph.nodes[out_node_idx]
                    input_idx = next_node.inputs.index(out_str)

                comm_id = self.get_comm_id(curr_id, out_node_idx, input_idx, batch_id)

                if src_comp is not None and out_node_idx < 0:
                    grad_keys.append(comm_id)
                recv_ids.append((input_idx, comm_id, out_node_idx))

        recv_ids.sort()

        ev_list = []
        for _, comm_id, out_idx in recv_ids:
            if out_idx < 0:
                continue

            out_node = self.part_graph.nodes[out_idx]
            out_rank = out_node.rank
            if out_rank < 0:
                continue

            if src_comp is not None and out_node.idx != src_comp.node_id:
                continue

            grad_buf = self.grad_buffers[comm_id]
            if grad_buf is None:
                continue

            for tensor in grad_buf:
                if tensor is None:
                    continue

                global_out_rank = self.net_state.pp_group_ranks[out_rank]
                ev = dist.irecv(tensor, global_out_rank, group=self.net_state.pp_group)
                self.grad_irecv[(node.idx, batch_id)].append((ev, comm_id))
                ev_list.append(ev)

        if src_comp is None:
            for ev in ev_list:
                ev.wait()

        for comm_id in grad_keys:
            while comm_id not in self.bw_tokens:
                # print(f"recv_grad_async wait for {comm_id}")
                recv_key = self.bw_queue.get()
                self.bw_tokens.add(recv_key)

        return ev_list

    def exec_comm(self, comm):
        self.debug(f"exec comm {comm}")
        src_comp = comm.src_comp
        dst_comp = comm.dst_comp
        is_fwd = src_comp.rank == self.pp_rank

        node_id = src_comp.node_id
        batch_id = src_comp.batch_id

        if is_fwd:
            # self.debug(f"enqueue send {comm}")
            part_node = self.part_graph.nodes[node_id]
            if src_comp.is_forward:
                self.send_act_async(part_node, batch_id, dst_comp)
            else:
                self.send_grad_async(part_node, batch_id, dst_comp)
        else:
            # self.debug(f"enqueue recv {comm}")
            dst_node_id = dst_comp.node_id
            dst_batch_id = dst_comp.batch_id
            part_node = self.part_graph.nodes[dst_node_id]

            if dst_comp.is_forward:
                self.recv_act_async(part_node, dst_batch_id, src_comp)
            else:
                self.recv_grad_async(part_node, dst_batch_id, src_comp)

    def send_comm(self):
        # collect sendable ranks and nodes
        send_comms = []

        # filter only comms until sendable comm is met
        for rank, comm_q in enumerate(self.comm_steps_pending):
            if rank == self.pp_rank:
                continue

            enq_pos = -1
            for idx, comm in enumerate(comm_q):
                comm = comm_q[idx]
                src_comp = comm.src_comp

                # enqueue send/recv until the last completed send is met
                if src_comp.rank == self.pp_rank:
                    if (
                        src_comp.node_id,
                        src_comp.batch_id,
                        src_comp.is_forward,
                    ) in self.completed_jobs:
                        enq_pos = idx
                    else:
                        break

            for _ in range(enq_pos + 1):
                send_comms.append(comm_q.popleft())

        for comm in send_comms:
            self.exec_comm(comm)

    def recv_comm(self, node: PartitionNode, batch_id: int, is_fwd: bool):
        recv_comms = []

        comm_q = self.comm_steps_pending[self.pp_rank]

        for rank, comm_q in enumerate(self.comm_steps_pending):
            if rank == self.pp_rank:
                continue

            # pop up until required recv_comm is met
            last_recv_step = self.recv_step_last[rank][(node.idx, batch_id, is_fwd)]
            while len(comm_q) >= last_recv_step > 0:
                recv_comms.append(comm_q.popleft())

            while comm_q:
                comm = comm_q[0]
                dst_comp = comm.dst_comp
                if dst_comp.rank != self.pp_rank:
                    break

                if dst_comp.node_id != node.idx:
                    break

                if comm.src_comp.is_forward != is_fwd:
                    break

                recv_comms.append(comm_q.popleft())

        for comm in recv_comms:
            self.exec_comm(comm)

        if is_fwd:
            ev_list = self.act_irecv[(node.idx, batch_id)]
        else:
            ev_list = self.grad_irecv[(node.idx, batch_id)]

        for i, (ev, comm_id) in enumerate(ev_list):
            ev.wait()

    def recv_grad_comm(self, node: PartitionNode, batch_id: int):
        # receive from the main proc / skip other than last node
        if self.runner is None:
            return

        curr_id = node.idx

        grad_keys = []
        recv_ids = []
        for out_idx, out_str in enumerate(node.outputs):
            for out_node_idx in node.output_map[out_str]:
                if not (out_node_idx < 0):
                    continue

                out_node_idx = -1
                input_idx = out_idx
                comm_id = self.get_comm_id(curr_id, out_node_idx, input_idx, batch_id)
                grad_keys.append(comm_id)
                recv_ids.append((comm_id, out_node_idx))

        for comm_id in grad_keys:
            while comm_id not in self.bw_tokens:
                # print(f"wait for {comm_id}")
                recv_key = self.bw_queue.get()
                self.bw_tokens.add(recv_key)

    def optimizer_step(self, node: PartitionNode = None, batch_id: int = None):
        self.all_reduce_grad()

        if self.optimizer:
            self.scaler.step(self.optimizer)

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

            with self.get_mp_ctx():
                inputs.append(list_map(clone_value, value))

        mod = self.mods[curr_id]

        if self.compiler is not None and curr_id not in self.mod_compiled:
            mod = compile_module(self.compiler, mod, inputs)
            self.mods[curr_id] = mod
            self.mod_compiled.add(curr_id)

        # TODO: compile module
        if len(self.net_state.dp_mesh) > 1 and curr_id not in self.mod_compiled:
            if self.option.use_fsdp:
                mod = FSDP(
                    mod,
                    process_group=self.net_state.dp_group,
                    # use_orig_params=True,
                    device_id=self.device,
                )
            else:
                mod = DDP(
                    mod,
                    # device_ids=list(self.net_state.dp_group_ranks),
                    process_group=self.net_state.dp_group,
                    gradient_as_bucket_view=True,
                    # static_graph=True,
                )
            self.mods[curr_id] = mod
            self.mod_compiled.add(curr_id)

        start_event = self.recorder.record_event(self.device, self.main_stream)

        dp_ctx = (
            mod.no_sync()
            if len(self.net_state.dp_mesh) > 1 and batch_id != self.option.batch_cnt - 1
            else nullcontext()
        )

        with self.get_mp_ctx():
            with dp_ctx:
                result = mod(*inputs)

        del inputs

        end_event = self.recorder.record_event(self.device, self.main_stream)
        self.recorder.append_fw(curr_id, start_event, end_event)

        result_id = self.get_result_id(node.idx, batch_id)
        self.fw_results[result_id] = result

    def backward(self, node: PartitionNode, batch_id: int):
        curr_id = node.idx
        grads_list = []

        for out_idx, out_str in enumerate(node.outputs):
            grads = None
            for out_node_idx in node.output_map[out_str]:
                if out_node_idx < 0:
                    if self.runner is None:
                        continue

                    comm_id = self.get_comm_id(curr_id, -1, out_idx, batch_id)
                    local_grads = self.loss_grads[comm_id]

                    # gc-out
                    self.loss_grads[comm_id] = [None]
                else:
                    next_node = self.part_graph.nodes[out_node_idx]
                    input_idx = next_node.inputs.index(out_str)

                    comm_id = self.get_comm_id(
                        curr_id, out_node_idx, input_idx, batch_id
                    )

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
            elif torch.is_tensor(grads):
                grads = [grads]

            grads_list.extend(grads)

        result_id = self.get_result_id(node.idx, batch_id)
        results = self.fw_results[result_id]

        if len(node.outputs) == 1:
            results = [results]

        if len(results) != len(grads_list):
            self.error(
                f"grad count mismatch {len(results)} != {len(grads_list)} / {curr_id} - {batch_id}"
            )

        fin_tensor = []
        fin_grads = []

        for t, g in zip(results, grads_list):
            # filter non-scalar without gradient
            if g is None and t.ndim > 0:
                continue
            fin_tensor.append(t)
            fin_grads.append(g)

        if len(fin_tensor) != len(fin_grads):
            self.error(
                f"require_grads grad count mismatch {len(fin_tensor)} != {len(fin_grads)} / {curr_id} - {batch_id}"
            )

        fin_tensor = self.scaler.scale(fin_tensor)

        if len(self.net_state.dp_mesh) > 1 and batch_id != self.option.batch_cnt - 1:
            mod = self.mods[curr_id]
            with mod.no_sync():
                torch.autograd.backward(fin_tensor, fin_grads, retain_graph=False)
        else:
            torch.autograd.backward(fin_tensor, fin_grads, retain_graph=False)

        # GC-out forward / backward results
        del self.fw_results[result_id]
