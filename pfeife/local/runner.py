import logging
import time
from typing import List, Optional, Dict
from collections import defaultdict
from threading import Thread
from queue import Queue

import torch
import torch.fx as fx
import torch.nn as nn
import torch.multiprocessing as mp

from .graph_bench import gen_comp_graph, gen_comp_graph_cpu
from ..schedule.planner import get_pipe_class
from ..schedule.scheduler import (
    BFSScheduler,
    DFSScheduler,
    PipelessScheduler,
    NoSkewScheduler,
    SingleCommScheduler,
    BeamScheduler,
)
from ..graph.partition_graph import partition_module_by_pgraph
from ..device_bench import DeviceBench
from ..utils import get_logger, fmem
from ..option import PipeOption
from .worker import ThreadWorker, run_reducer


class PipeTensor(torch.Tensor):
    def __init__(self, batch_no: int):
        super().__init__()
        self.batch_no = batch_no


class RunnerThreadPool:
    def __init__(self, model: nn.Module, runner: "PipeGraphRunner", loss_fn):
        self.model = model
        self.runner = runner
        self.workers = runner.workers
        self.loss_fn = loss_fn

        self.option = runner.option
        self.batch_idx = 0
        self.batch_cnt = runner.batch_cnt
        self.device_cnt = self.option.device_cnt

        self.loss_stream = dict()
        self.loss_ctx = dict()
        self.loss_device = None

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

        self.initialized = False

    def add_input(self, args, kwargs, target):
        if self.initialized is False:
            # run dynamo once and genereate graphs
            self.model(*args, **kwargs)
            self.initialized = True

        self.worker_queues[self.batch_idx].put((args, kwargs, target))

        # recevie ACK from pipe graph runner ("fwd_load_input": batch_id)
        _ = self.main_queue.get()

        self.batch_idx += 1

    def _per_batch_thread(self, batch_idx: int):
        this_queue = self.worker_queues[batch_idx]

        while True:
            args, kwargs, target = this_queue.get()

            ret_val = self.model(*args, **kwargs)

            # populate loss_device
            if self.loss_device is None:
                loss = self.loss_fn(ret_val, target)
                loss.backward()
                loss_item = loss.item()
                self.loss_device = loss.device
            else:
                if self.loss_device not in self.loss_stream:
                    stream = torch.cuda.Stream(device=self.loss_device)
                    self.loss_stream[self.loss_device] = stream
                    self.loss_ctx[self.loss_device] = torch.cuda.stream(stream)
                stream = self.loss_stream[self.loss_device]
                ctx = self.loss_ctx[self.loss_device]

                with ctx:
                    loss = self.loss_fn(ret_val, target)
                    loss.backward()
                    loss_item = loss.item()
                stream.synchronize()

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

        done_cnt = 0
        while done_cnt < self.device_cnt:
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
                done_cnt += 1
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
                    self.workers[rank].recv_loss_grad(
                        batch_idx, node_id, output_idx, grad
                    )
                    del output_val, grad

                del outputs  # GC out

        # runner.reducer_proc.join()

        # for worker in self.workers:
        # worker.job_thread.join()

    def reset(self):
        self.batch_idx = 0
        self.losses.clear()


class PipeGraphRunner(nn.Module):
    def __init__(
        self,
        devices: List[str],
        workers: List[ThreadWorker],
        device_bench: DeviceBench,
        option: PipeOption,
    ):
        super().__init__()
        self.logger = get_logger()

        self.gm: Optional[fx.GraphModule] = None

        self.devices = devices
        self.device_bench = device_bench
        self.workers = workers
        self.option = option
        self.batch_cnt = option.batch_cnt

        self.batch_no = 0
        self.partition_graph = None
        self.reducer_proc = None

    def set_graph_module(self, gm: fx.GraphModule):
        self.gm = gm

    def init_reducer(self, param_set):
        smp = mp.get_context("spawn")
        self.ctx = smp

        queues = [[smp.Queue(), smp.Queue()] for _ in self.workers]
        reducers = [worker.reducer for worker in self.workers]

        for queue, worker in zip(queues, self.workers):
            worker.reducer_queues = queue

        world_size = len(self.workers)
        self.reducer_proc = mp.spawn(
            # target=self.reducer.run,
            run_reducer,
            args=(reducers, queues, world_size, self.option.reducer_backend),
            nprocs=world_size,
            daemon=True,
            join=False,
        )

        grad_ranks_all = dict()
        for idx, ranks in param_set.shared_rank.items():
            grad_ranks_all[idx] = tuple(sorted(ranks))

        for queue, worker in zip(queues, self.workers):
            worker.init_reducer(queue, param_set, grad_ranks_all)

    def gen_partition_graph(self, example_inputs):
        print_graph = self.option.print_graph
        if print_graph:
            print("==================== Original Graph ====================")
            self.gm.graph.print_tabular()

        if self.option.cpu_test:
            comp_graph, output = gen_comp_graph_cpu(self.gm, example_inputs)
        else:
            comp_graph, output = gen_comp_graph(
                self.gm, example_inputs, self.devices, self.option
            )

        pipe_gen_class = get_pipe_class(self.option.partitioner)
        self.pipe_gen = pipe_gen_class(comp_graph, self.device_bench, self.option)

        max_mem = []
        for dev in self.devices:
            if dev == "cpu":
                max_mem.append(2**62)
            else:
                max_mem.append(torch.cuda.mem_get_info(dev)[1])

        self.pipe_graph, sched_params, _ = self.pipe_gen.gen_graph(
            self.batch_cnt, max_mem, self.option.split_points
        )

        # TODO: debug
        # if self.logger.level == logging.DEBUG:
        # self.pipe_gen.save_graph_png(self.pipe_graph, self.batch_cnt, len(self.devices))

        self.partition_graph = partition_module_by_pgraph(
            self.gm, self.pipe_graph, self.devices
        )

        if print_graph:
            for node_idx, part_node in self.partition_graph.nodes.items():
                print(
                    f"==================== Partition Graph: {node_idx} ===================="
                )
                part_node.graph_module.graph.print_tabular()

        # TODO: reassign device
        for pnode in self.pipe_graph.nodes:
            pnode.device = self.workers[pnode.rank].device

        for worker in self.workers:
            worker.set_partition_graph(self.partition_graph)

        self.set_scheduler(self.option.scheduler, sched_params)

        self.pipe_graph.print_graph()

        torch.cuda.empty_cache()

        param_set = comp_graph.param_set
        if len(param_set.shared_idx) > 0:
            self.init_reducer(param_set)

        return output

    def set_scheduler(self, sched_type, sched_params, dep_graph=None):
        # TODO: select scheduler
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

        for worker in self.workers:
            worker.set_scheduler_steps(sched.steps[worker.rank])

    def reset(self):
        self.batch_no = 0

    def forward(self, *args, **kwargs):
        if self.partition_graph is None:
            self.gen_partition_graph(args)

        batch_no = self.batch_no
        self.batch_no += 1

        for place_id, input_usage in enumerate(self.partition_graph.inputs):
            for part_id, input_idx in input_usage:
                self.workers[part_id].load_input(
                    batch_no, part_id, input_idx, args[place_id]
                )

        # TODO: find what partitions is an output partition
        # TODO: should we return fake tensor or some random (primitive) value?
        # max_node = max(self.partition_graph.nodes.keys())
        # output_cnt = len(self.partition_graph.nodes[max_node].outputs)
        output_cnt = len(self.partition_graph.outputs)

        return [PipeTensor(batch_no) for _ in range(output_cnt)]

    def train(self):
        self.gm.train()

    def eval(self):
        self.gm.eval()


class PipeGraphRunnerWithSE(PipeGraphRunner):
    def __init__(
        self,
        devices: List[str],
        workers: List[ThreadWorker],
        device_bench: DeviceBench,
        option: PipeOption,
    ):
        super().__init__(devices, workers, device_bench, option)

        self.main_queue: Queue = None
        self.thread_pool_queue: Queue = None

        self.forward_queues = [Queue() for _ in range(self.batch_cnt)]

        self.initialized = False

        for worker in self.workers:
            worker.set_runner(self)

    def return_output(self, batch_idx, output_idx, output_val, req_grad, event=None):
        self.forward_queues[batch_idx].put((output_idx, output_val, req_grad, event))

    def pipe_done(self):
        self.thread_pool_queue.put(("pipe_done",))

    def forward(self, *args, **kwargs):
        if self.initialized is False:
            self.initialized = True
            # return the result of first test batch
            return self.gen_partition_graph(args)

        batch_no = self.batch_no
        fw_queue = self.forward_queues[batch_no]
        self.batch_no += 1

        for input_val, usage in zip(args, self.partition_graph.inputs):
            for part_id, input_idx in usage:
                node = self.partition_graph.nodes[part_id]
                self.workers[node.rank].load_input(
                    batch_no, node.idx, input_idx, input_val
                )

        # ACK-load_input to the thread pool / matching batch_no
        self.thread_pool_queue.put(("fwd_load_input", batch_no))

        self.logger.debug(f"batch {batch_no} input loaded")

        # wait for outputs
        outputs = [None for _ in range(len(self.partition_graph.outputs))]

        for _ in range(len(self.partition_graph.outputs)):
            output_idx, output_val, req_grad, event = fw_queue.get()
            if event is not None:
                event.synchronize()

            if torch.is_tensor(output_val):
                output_val = output_val.detach()
                if req_grad:
                    output_val.requires_grad_()
            outputs[output_idx] = output_val

        # print(batch_no, args, outputs[0].sum())

        self.thread_pool_queue.put(("fwd_outputs", batch_no, outputs))

        return outputs
