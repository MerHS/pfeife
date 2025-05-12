from typing import List
from queue import Queue
from functools import partial
from copy import deepcopy
from collections import defaultdict
import atexit
import tempfile
import threading
import uuid
import os
import gc

import torch
import torch.nn as nn
import torch.fx as fx

from ..net_states import get_state
from ...option import PipeOption
from ...batch import split_args_kwargs
from ..worker import ProcessWorker, LoopFlag
from ...utils import to_device, fmem

from ...local.graph_bench import gen_comp_graph, gen_local_comp_graph
from ...schedule.scheduler import (
    BFSScheduler,
    DFSScheduler,
    PipelessScheduler,
    BeamScheduler,
    SingleCommScheduler,
)
from ...schedule.planner import get_pipe_class
from ...graph.partition_graph import partition_module_by_pgraph

STEP_TOKEN = "###<<STEP_TOKEN>>###"


class PartialRunner:
    def __init__(self, option: PipeOption = PipeOption(), optimizer=None):
        self.exec_fn = None
        self.option = option
        self.optimizer = optimizer

        self.net_state = get_state()
        self._compiled = False
        self._batch_threads = []
        self._train_queues = [Queue() for _ in range(option.batch_cnt)]
        self._return_queues = [Queue() for _ in range(option.batch_cnt)]
        self._get_grad = dict()

        self.worker = ProcessWorker(self)
        self.worker.set_option(option)
        PartialModule.set_runner(self)
        atexit.register(self.worker.destroy_workers)

        for i in range(option.batch_cnt):
            t = threading.Thread(target=self._exec_loop, args=(i,), daemon=True)
            self._batch_threads.append(t)
            t.start()

    def set_exec_fn(self, exec_fn):
        self.exec_fn = exec_fn

    def step(self, *args, **kwargs):
        # args, kwargs: input values for exec_fn
        # must be able to split into chunks
        main_device = self.net_state.device

        args, kwargs = split_args_kwargs(args, kwargs, self.option.batch_cnt)
        # args, kwargs = to_device((args, kwargs), main_device)
        self._output_list = [[] for _ in range(self.option.batch_cnt)]

        if not self._compiled:
            self._probe_forward(args[0], kwargs[0])
            if self.option.print_graph:
                print(
                    f"Found {len(PartialModule._pp_modules)} graphs in the main module"
                )
            for mod in PartialModule._pp_modules:
                if mod.should_pp:
                    mod._compile(self.optimizer)
            self._compiled = True

        self.reset()

        # TODO: more than one pipe module
        self.results = []
        for bid, (arg, kwarg) in enumerate(zip(args, kwargs)):
            self._train_queues[bid].put((arg, kwarg))
            self._return_queues[bid].get()

        for bid in range(self.option.batch_cnt):
            PartialModule._module_queues[bid][0].put(
                STEP_TOKEN
            )  # notify to _wait_for_execution

        # run main job
        self.worker.run_with_module(0)

        results = self.results
        self.results = []

        return results

    def reset(self):
        for pm in PartialModule._all_modules:
            pm.reset()

        for mod in PartialModule._pp_modules:
            mod.outputs = [None for _ in range(self.option.batch_cnt)]

        self.worker.reset_cache()
        self.worker.prepare_workers(LoopFlag.CONTINUE_LOOP)
        self._get_grad.clear()

    def gather_params(self):
        for pm in PartialModule._all_modules:
            pm.gather_params()

    def return_output(self, batch_idx, output_idx, output_val, req_grad, node_idx):
        outputs = PartialModule._pp_modules[0].outputs[batch_idx]

        grad_dict = self._get_grad
        if batch_idx not in grad_dict:
            grad_dict[batch_idx] = [node_idx] * len(outputs)

        if torch.is_tensor(output_val) and req_grad:
            new_output = output_val.detach()
            new_output.requires_grad_()
            new_output.register_hook(
                partial(self.backward_hook, batch_idx, node_idx, output_idx)
            )

            # grad_dict[batch_idx][output_idx] = node_idx

            output_val = new_output

        # TODO: not just a first module
        outputs[output_idx] = output_val

    def backward_hook(self, batch_idx, node_idx, output_idx, grad):
        # print(f"Backward hook: {batch_idx} {node_idx} {output_idx}")
        self._get_grad[batch_idx][output_idx] = True
        self.worker.recv_loss_grad(0, batch_idx, node_idx, output_idx, grad)

    def finalize_output(self, pipe_idx, batch_idx):
        # outputs = PartialModule._pp_modules[pipe_idx].outputs[batch_idx]
        queue = PartialModule._module_queues[batch_idx][pipe_idx]
        queue.put(None)

        ret = self._return_queues[batch_idx].get()

        for output_idx, maybe_node_idx in enumerate(self._get_grad[batch_idx]):
            if maybe_node_idx is not True:
                # print(f"finalize output {batch_idx} {maybe_node_idx} {output_idx}")
                self.worker.recv_loss_grad(
                    0, batch_idx, maybe_node_idx, output_idx, None
                )

        self.results.append(ret)

    def _exec_loop(self, batch_idx):
        main_queue = self._train_queues[batch_idx]
        while True:
            args, kwargs = main_queue.get()
            ret_val = self.exec_fn(*args, **kwargs)
            self._return_queues[batch_idx].put(ret_val)

    def _probe_forward(self, args, kwargs):
        self.exec_fn(*args, **kwargs)
        self._compiled = True

    def log_memory_usage(self):
        self.worker.all_log_memory_usage()

    def train(self):
        # TODO
        pass

    def eval(self):
        # TODO
        pass


class PartialModule(nn.Module):
    _train_ready = False
    _module_queues = []  # [[Queue for pipe_nodes] for batch_cnt]
    _pipe_node_cnt = 0
    _runner: PartialRunner = None
    _all_modules = []
    _pp_modules = []
    _option: PipeOption = None
    _device_bench = None

    def __init__(self, gm: fx.GraphModule):
        super().__init__()
        self.gm = gm
        self.curr_batch = 0
        self.net_state = get_state()
        self.device = self.net_state.device

        self.compiled = False
        self.probed = False
        self.pipe_idx = 0

        # TODO: get max should_pp
        can_pp = all([not mod.should_pp for mod in self._all_modules])
        # has enough pp nodes
        should_pp = len(gm.graph.nodes) >= 2 * self.net_state.pp_size
        should_pp = (
            should_pp
            and len([n for n in gm.graph.nodes if n.op == "call_module"])
            >= self.net_state.pp_size
        )

        # pipeline only if the number of fx nodes >= the number of devices * 2
        self.should_pp = can_pp and should_pp

        if self._option.print_graph:
            idx = len(self._all_modules) + 1
            print(
                f"================ GRAPH {idx}: {len(gm.graph.nodes)} nodes / PP enbaled: {self.should_pp} ================"
            )
            self.gm.graph.print_tabular()

        self.outputs = []
        self.rank_pgs = None
        self._all_modules.append(self)

    @staticmethod
    def set_runner(runner: PartialRunner):
        option = runner.option
        PartialModule._runner = runner
        PartialModule._module_queues = [[] for _ in range(option.batch_cnt)]
        PartialModule._option = option
        PartialModule._device_bench = runner.worker.run_device_bench(option)

    def gather_params(self):
        if self.rank_pgs is None:
            print(
                f"Model is not yet compiled as a pipelined model. Ignore gather_params."
            )
            return

        worker = self._runner.worker
        param_dict = worker.gather_params()
        for idx, params in param_dict.items():
            part_node = self.partition_graph.nodes[idx]
            part_graph = self.rank_pgs[part_node.rank].nodes[idx].graph_module
            for param, param_data in zip(part_graph.parameters(), params):
                param.data.copy_(param_data)

    def _probe_forward(self, args):
        # must be called from _probe_forward
        self._pp_modules.append(self)
        self._pipe_node_cnt += 1
        self.pipe_idx = self._pipe_node_cnt - 1

        for qs in self._module_queues:
            qs.append(Queue())

        option = self._option
        worker = self._runner.worker
        is_master = self.net_state.rank == self.net_state.world_size - 1
        has_dp = len(self.net_state.dp_group_ranks) > 1

        # master of the main
        if has_dp and is_master and option.graph_bench is None:
            file_name = f"pfeife_{uuid.uuid4()}.graph"
            option.graph_bench = os.path.join(tempfile.gettempdir(), file_name)

        if has_dp and not is_master:
            worker.broadcast_comp_graph(option, is_master)

        # TODO: temp impl.
        # pp_group_ranks = self.net_state.pp_group_ranks
        # local_devices = [torch.device(f"cuda:{i}") for i in pp_group_ranks]

        # comp_graph, output = gen_comp_graph(
        #     self.gm, args, local_devices, option
        # )

        comp_graph, output = gen_local_comp_graph(self.gm, args, self.device, option)

        output = to_device(output, self.device)

        self.comp_graph = comp_graph

        return output

    def _compile(self, optimizer):
        option = self._option
        worker = self._runner.worker
        is_master = self.net_state.rank == self.net_state.world_size - 1
        has_dp = len(self.net_state.dp_group_ranks) > 1
        device_bench = self._device_bench
        comp_graph = self.comp_graph
        self.comp_graph = None

        if has_dp and is_master:
            worker.broadcast_comp_graph(option, is_master)

        pipe_gen_class = get_pipe_class(option.partitioner)
        pipe_gen = pipe_gen_class(comp_graph, device_bench, option)

        if self.device == torch.device("cpu"):
            max_mem = 2**62
        else:
            max_mem = torch.cuda.mem_get_info(self.device)[1]

        max_mem = [max_mem] * option.device_cnt

        pipe_graph, sched_params, dep_graph = pipe_gen.gen_graph(
            option.batch_cnt, max_mem, option.split_points
        )

        if is_master:
            pipe_graph.print_graph()

        world_size = self.net_state.world_size
        world_nnodes = self.net_state.nnodes
        pp_group_ranks = self.net_state.pp_group_ranks
        ndev_per_node = world_size // world_nnodes

        # TODO: set world device
        world_devices = [
            torch.device(f"cuda:{rank % ndev_per_node}") for rank in pp_group_ranks
        ]

        self.partition_graph = partition_module_by_pgraph(
            self.gm, pipe_graph, world_devices
        )

        if option.print_graph:
            for idx, part_graph in self.partition_graph.nodes.items():
                gm = part_graph.graph_module
                print(
                    f"================ PARTITION GRAPH {idx}: {len(gm.graph.nodes)} nodes ================"
                )
                gm.graph.print_tabular()

        # clear memory
        param_set = comp_graph.param_set
        del comp_graph, pipe_graph, pipe_gen_class, pipe_gen
        self.comp_graph = None

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        sched = self._get_scheduler(option.scheduler, sched_params, dep_graph)
        rank_pgs, optim_dict = self.partition_graph.split_rank_gm(optimizer)
        self.rank_pgs = rank_pgs

        worker.distribute_graph(rank_pgs, sched, optimizer, optim_dict, param_set)
        worker.set_scheduler_steps(sched)

        self.gm = None

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        # # print each device's memory usage
        # local_dev_cnt = torch.cuda.device_count()
        # local_devices = [torch.device(f"cuda:{i}") for i in range(local_dev_cnt)]
        # for device in local_devices:
        #     torch.cuda.reset_max_memory_allocated(device)
        #     torch.cuda.reset_max_memory_cached(device)
        #     print(
        #         f"device {device} max mem: {fmem(torch.cuda.max_memory_allocated(device))}"
        #     )

    def _get_scheduler(self, sched_type, sched_params, dep_graph=None):
        if sched_type == "beam":
            sched = BeamScheduler(sched_params)
        elif sched_type == "single_comm":
            sched = SingleCommScheduler(sched_params, dep_graph)
        elif sched_type == "bfs":
            sched = BFSScheduler(sched_params)
        elif sched_type == "dfs":
            sched = DFSScheduler(sched_params)
        elif sched_type == "pipeless":
            sched = PipelessScheduler(sched_params)
        else:
            raise RuntimeError(f"Invalid scheduler type: {sched_type}")

        return sched

    def reset(self):
        self.curr_batch = 0

    def forward(self, *args):
        if not self.should_pp:
            # pass through
            args = to_device(args, self.device)
            return self.gm(*args)

        if not self.probed:
            self.probed = True
            return self._probe_forward(args)

        curr_batch = self.curr_batch
        self.curr_batch += 1

        self._wait_for_execution(curr_batch, args)  # wait until its turn to run
        return self._step(curr_batch)  # transparently run the original forward

    def _step(self, curr_batch):
        # TODO: add forward/backward hooks
        # TODO: add batch scheduler

        curr_queue = self._module_queues[curr_batch][self.pipe_idx]
        curr_queue.get()  # wait from worker.run_with_module

        return self.outputs[curr_batch]

    def _wait_for_execution(self, curr_batch, args):
        # TODO: temp impl. send args to the first worker and wait
        self.outputs[curr_batch] = [
            None for _ in range(len(self.partition_graph.outputs))
        ]

        for input_val, usage in zip(args, self.partition_graph.inputs):
            for part_id, input_idx in usage:
                node = self.partition_graph.nodes[part_id]
                self._runner.worker.distribute_input(
                    node, curr_batch, input_idx, input_val
                )

        self._runner._return_queues[curr_batch].put(STEP_TOKEN)

        curr_queue = self._module_queues[curr_batch][self.pipe_idx]
        curr_queue.get()


def partial_compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    module = PartialModule(gm)
    return module
