import os
from typing import List, Optional, Dict
from threading import Thread
from queue import Queue
import uuid
import tempfile
import atexit

import torch
import torch.fx as fx
import torch.nn as nn

# TODO: bench by multiprocess
from ..local.graph_bench import gen_comp_graph
from ..schedule.scheduler import (
    BFSScheduler,
    DFSScheduler,
    PipelessScheduler,
    BeamScheduler,
    SingleCommScheduler,
)
from ..schedule.planner import get_pipe_class
from .net_states import get_state
from ..graph.partition_graph import partition_module_by_pgraph
from ..utils import get_logger
from ..option import PipeOption
from .worker import ProcessWorker


class RunnerThreadPool:
    def __init__(self, model: nn.Module, runner: "PipeGraphRunner", loss_fn):
        self.model = model
        self.runner = runner
        self.loss_fn = loss_fn

        self.option = runner.option

        self.batch_idx = 0
        self.batch_cnt = runner.batch_cnt
        self.device = self.runner.device

        self.loss_stream = dict()
        self.loss_ctx = dict()

        self.main_queue = Queue()
        self.runner_queue = Queue()

        self.worker_queues = [Queue() for _ in range(self.batch_cnt)]
        self.per_batch_threads = [
            Thread(target=self._per_batch_thread, daemon=True, args=(rank,))
            for rank in range(self.batch_cnt)
        ]
        self.losses = []

        for t in self.per_batch_threads:
            t.start()

        # join -> _forward: disabled
        runner.main_queue = self.runner_queue

        # forward -> add_input
        # _per_batch_thread -> join
        runner.thread_pool_queue = self.main_queue

    def add_input(self, args, kwargs, target):
        self.worker_queues[self.batch_idx].put((args, kwargs, target))

        # recevie ACK from pipe graph runner ("fwd_load_input": batch_id)
        _ = self.main_queue.get()

        self.batch_idx += 1

    def _per_batch_thread(self, batch_idx: int):
        this_queue = self.worker_queues[batch_idx]

        while True:
            args, kwargs, target = this_queue.get()

            ret_val = self.model(*args, **kwargs)

            loss = self.loss_fn(ret_val, target)
            loss.backward()

            loss_item = loss.item()
            # loss_item = 0

            # TODO: need it?
            # torch.cuda.synchronize(self.device)

            self.main_queue.put(("bwd_done", batch_idx))
            self.losses.append(loss_item)

            del loss
            del ret_val

    def join(self):
        """
        All ACKs are received from the main channel before calling it.

        "fwd_outputs": batch_id, outputs
        "bwd_done": batch_id
        """
        outputs_list = [None for _ in range(self.batch_cnt)]
        done_list = [False for _ in range(self.batch_cnt)]

        pgraph = self.runner.partition_graph
        pgraph_nodes = pgraph.nodes
        pgraph_output = pgraph.outputs

        done = False
        while done is False:
            msg = self.main_queue.get()
            batch_idx = -1

            if msg[0] == "fwd_outputs":
                batch_idx = msg[1]
                outputs = msg[2]
                outputs_list[batch_idx] = outputs
            elif msg[0] == "bwd_done":
                batch_idx = msg[1]
                done_list[batch_idx] = True
            elif msg[0] == "pipe_done":
                done = True
            else:
                raise RuntimeError("Invalid message type: {}".format(msg[0]))

            if (
                batch_idx >= 0
                and outputs_list[batch_idx] is not None
                and done_list[batch_idx]
            ):
                outputs = outputs_list[batch_idx]
                outputs_list[batch_idx] = None  # GC out

                # TODO: assuming final outputs are all primitive values or tensors
                for output_val, (node_id, output_idx) in zip(outputs, pgraph_output):
                    node = pgraph_nodes[node_id]
                    rank = node.rank

                    grad = None
                    if torch.is_tensor(output_val) and output_val.requires_grad:
                        grad = output_val.grad

                    # TODO: receive loss grad
                    self.runner.worker.recv_loss_grad(
                        rank, batch_idx, node_id, output_idx, grad
                    )
                    del output_val, grad

                del outputs  # GC out

    def reset(self):
        self.batch_idx = 0
        self.losses.clear()


class PipeGraphRunner(nn.Module):
    def __init__(
        self,
        option: PipeOption,
    ):
        super().__init__()
        self.logger = get_logger()

        self.gm: Optional[fx.GraphModule] = None
        self.worker = ProcessWorker(self)
        self.worker.set_option(option)

        self.net_state = get_state()
        self.device = self.worker.device
        self.device_cnt = option.device_cnt

        self.pp_rank = self.net_state.pp_rank
        self.is_main = self.net_state.is_main_rank
        self.rank = self.net_state.rank
        self.world_size = self.net_state.world_size

        self.option = option

        self.batch_no = 0
        self.batch_cnt = option.batch_cnt
        self.partition_graph = None

        self.main_queue: Queue = None
        self.thread_pool_queue: Queue = None

        self.forward_queues = [Queue() for _ in range(self.batch_cnt)]
        self.worker_queue = Queue()

        self.initialized = False
        atexit.register(self.close)

    def set_graph_module(self, gm: fx.GraphModule):
        self.gm = gm

    def gen_partition_graph(self, input_tensors):
        print_graph = self.option.print_graph
        if print_graph:
            print("==================== Original Graph ====================")
            self.gm.graph.print_tabular()

        device_bench = self.worker.run_device_bench()

        is_master = self.net_state.rank == self.net_state.world_size - 1
        has_dp = len(self.net_state.dp_group_ranks) > 1

        # master of the main
        if has_dp and is_master and self.option.graph_bench is None:
            file_name = f"pfeife_{uuid.uuid4()}.graph"
            self.option.graph_bench = os.path.join(tempfile.gettempdir(), file_name)

        if has_dp and not is_master:
            self.worker.broadcast_comp_graph(self.option, is_master)

        local_dev_cnt = torch.cuda.device_count()
        local_devices = [torch.device(f"cuda:{i}") for i in range(local_dev_cnt)]
        comp_graph, output = gen_comp_graph(
            self.gm, input_tensors, local_devices, self.option
        )

        if has_dp and is_master:
            self.worker.broadcast_comp_graph(self.option, is_master)

        pipe_gen_class = get_pipe_class(self.option.partitioner)
        pipe_gen = pipe_gen_class(comp_graph, device_bench, self.option)

        if self.device == torch.device("cpu"):
            max_mem = 2**62
        else:
            max_mem = torch.cuda.mem_get_info(self.device)[1]

        max_mem = [max_mem] * self.device_cnt

        pipe_graph, sched_params, dep_graph = pipe_gen.gen_graph(
            self.batch_cnt, max_mem, self.option.split_points
        )

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

        pipe_graph.print_graph()
        if print_graph:
            for node_idx, part_node in self.partition_graph.nodes.items():
                print(
                    f"==================== Partition Graph: {node_idx} ===================="
                )
                part_node.graph_module.graph.print_tabular()

        # clear memory

        del comp_graph, pipe_graph, pipe_gen_class, pipe_gen

        if type(output) == list or type(output) == tuple:
            new_output = []
            for out in output:
                if torch.is_tensor(out):
                    new_output.append(out.detach())
                else:
                    new_output.append(out)
            if type(output) == tuple:
                new_output = tuple(new_output)
        elif torch.is_tensor(output):
            new_output = output.detach()
        else:
            new_output = output
        output = new_output

        torch.cuda.empty_cache()

        sched = self._get_scheduler(self.option.scheduler, sched_params, dep_graph)
        rank_pgs = self.partition_graph.split_rank_gm()

        self.worker.distribute_graph(rank_pgs, sched)
        self.worker.set_scheduler_steps(sched)

        torch.cuda.empty_cache()

        self.worker_thread = Thread(target=self._wait_for_worker_run, daemon=True)
        self.worker_thread.start()

        self.temp_output = output

        return output

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

    def prepare_workers(self):
        self.batch_no = 0
        self.worker.reset_cache()
        self.worker.prepare_workers()

    def close(self):
        self.worker.destroy_workers()

    def return_output(self, batch_idx, output_idx, output_val, req_grad, node_idx=None):
        # print(f"return output: {batch_idx}, {output_idx}, {req_grad}")
        self.forward_queues[batch_idx].put((output_idx, output_val, req_grad))

    def pipe_done(self):
        self.thread_pool_queue.put(("pipe_done",))

    def _wait_for_worker_run(self):
        while True:
            _ = self.worker_queue.get()
            self.worker.run()

    def forward(self, *args, **kwargs):
        if self.initialized is False:
            self.initialized = True
            # return the result of first test batch
            return self.gen_partition_graph(args)

        # print(f"forward {self.batch_no}, {self.batch_cnt}")

        batch_no = self.batch_no
        fw_queue = self.forward_queues[batch_no]
        self.batch_no += 1

        for input_val, usage in zip(args, self.partition_graph.inputs):
            for part_id, input_idx in usage:
                node = self.partition_graph.nodes[part_id]
                self.worker.distribute_input(node, batch_no, input_idx, input_val)

        # ACK-load_input to the thread pool / matching batch_no
        self.thread_pool_queue.put(("fwd_load_input", batch_no))

        if self.batch_no == self.batch_cnt:
            self.worker_queue.put(None)

        # wait for outputs
        outputs = [None for _ in range(len(self.partition_graph.outputs))]

        # TODO: temporarily skip outputs of internal nodes
        last_out_cnt = 0
        for part_id, _ in self.partition_graph.outputs:
            node = self.partition_graph.nodes[part_id]
            if node.rank == self.net_state.pp_rank:
                last_out_cnt += 1

        # for _ in range(len(self.partition_graph.outputs)):
        for _ in range(last_out_cnt):
            output_idx, output_val, req_grad = fw_queue.get()

            if torch.is_tensor(output_val):
                output_val = output_val.detach()
                if req_grad:
                    output_val.requires_grad_()
            outputs[output_idx] = output_val

        self.thread_pool_queue.put(("fwd_outputs", batch_no, outputs))

        return outputs

    def train(self):
        self.gm.train()

    def eval(self):
        self.gm.eval()
