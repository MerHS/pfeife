import os
import time
from typing import List, Union
from collections import defaultdict

import torch
import torch._dynamo as dynamo
import torch.fx as fx
from torch.nn.modules.loss import _Loss
import torch.multiprocessing as mp
import torch.distributed as dist

from .worker import ThreadWorker
from ..batch import split_args_kwargs
from ..loss import LossWrapper
from ..option import PipeOption
from .runner import PipeGraphRunner, PipeTensor, RunnerThreadPool, PipeGraphRunnerWithSE
from ..utils import get_logger, Recorder, to_device, fmem
from ..device_bench import DeviceBench


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
        self.base_time = time.time()
        self.logger = get_logger()
        self.runner_thread_pool = None

        self.set_option(option)

    def __del__(self):
        for worker in self.workers:
            worker.clear_workers()

    def set_option(self, option: PipeOption):
        self.debug(f"got option: {option.__dict__}")

        # TODO: set this
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")

        self.option = option
        self.batch_cnt = option.batch_cnt
        self.device_cnt = option.device_cnt

        self.clear_record()

        # TODO: split by other strategies
        self.workers: List[ThreadWorker] = []
        self.devices = []

        for rank in range(self.device_cnt):
            if option.cpu_test:
                device = "cpu"
            elif option.device_map is None:
                device = f"cuda:{rank}"
            else:
                device = option.device_map[rank]

            worker = ThreadWorker(rank, self.loss_fn, option, self.base_time)

            self.workers.append(worker)
            self.devices.append(device)

        for worker in self.workers:
            worker.set_workers(self.workers)

        if option.cpu_test:
            device_bench = None
        else:
            device_bench = DeviceBench(self.devices)
            device_bench.run_bench(file_path=option.device_bench)

        runner_cls = PipeGraphRunner if option.no_side_effect else PipeGraphRunnerWithSE
        self.runner = runner_cls(self.devices, self.workers, device_bench, self.option)

        self.compiled = False

    def debug(self, msg):
        self.logger.debug("[Master] %s", msg)

    def info(self, msg):
        self.logger.info("[Master] %s", msg)

    def _compile(self):
        # Assign the model to the first device so that the usage of the device is not changed
        # TODO: what if the middle of the model uses also inlined device? (tensor.to(some_params.device))
        # some tensor-generating functions like torch.ones(..., device=...) will bake parameters
        dynamo.reset()
        # self.orig_model = self.orig_model.to(self.devices[0])

        graph_len = []

        def compile_fn(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            # TODO: what if this function is called two times from a training time?
            # ANSWER: return new handler for each graph.

            self.runner.set_graph_module(gm)

            graph_len.append(len(gm.graph.nodes))
            # gm.graph.print_tabular()

            if len(graph_len) > 1:
                import sys

                print(
                    f"multiple graph detected ({graph_len}): cannot be run on pfeife."
                )
                sys.exit(0)

            return self.runner

        dynamo_ctx = dynamo.optimize(compile_fn)
        self.opt_model = dynamo_ctx(self.orig_model)

        if not self.option.no_side_effect:
            self.runner_thread_pool = RunnerThreadPool(
                self.opt_model, self.runner, self.loss_fn
            )

    def train(self):
        self.is_train = True
        # self.opt_model.train()
        # for worker in self.workers:
        #     worker.set_train(True)

    def eval(self):
        self.is_train = False
        # self.opt_model.eval()
        # for worker in self.workers:
        #     worker.set_train(False)

    def _set_train(self):
        if self.is_train:
            self.opt_model.train()
            for worker in self.workers:
                worker.set_train(True)
        else:
            self.opt_model.eval()
            for worker in self.workers:
                worker.set_train(False)

    def init_runner(self):
        if self.runner_thread_pool is not None:
            self.runner_thread_pool.reset()

        for worker in self.workers:
            worker.reset_cache()

        if self.runner is not None:
            self.runner.reset()

    def clear_record(self):
        self.skip_test = 2
        self.fw_times = defaultdict(Recorder)
        self.bw_times = defaultdict(Recorder)

    def record_times(self):
        # skip n times
        if self.skip_test > 0:
            self.skip_test -= 1
            return

        fw_times = dict()
        bw_times = dict()

        for worker in self.workers:
            fw, bw = worker.get_event_logs()

            fw_times.update(fw)
            bw_times.update(bw)

        for k, t in fw_times.items():
            self.fw_times[k].update(t)

        for k, t in bw_times.items():
            self.bw_times[k].update(t)

    def print_record(self):
        fw_times = list(sorted(self.fw_times.items()))
        bw_times = list(sorted(self.bw_times.items()))

        for node_id, record in fw_times:
            if node_id < 0:
                header = f"Optimizer #{-node_id - 1}"
            else:
                header = f"Forward Node {node_id}"
            print(
                f"{header} - mean: {record.mean:5.3f}ms, stddev: {record.stddev:5.3f}ms"
            )
        for node_id, record in bw_times:
            print(
                f"Backward Node {node_id} - mean: {record.mean:5.3f}ms, stddev: {record.stddev:5.3f}ms"
            )

    def run(self, target, *args, **kwargs):
        """
        1. optimizer.zero_grad()
        2. forward & backward & optimizer step
        3. return loss
        """
        self.debug("start run")

        args, kwargs = split_args_kwargs(args, kwargs, self.batch_cnt)
        targets, _ = split_args_kwargs([target], dict(), self.batch_cnt)
        targets = [t[0] for t in targets]  # untuple

        # args, kwargs = to_device((args, kwargs), self.devices[0])

        if not self.compiled:
            # populate huggingface model's buffer, and then compile it
            # self.orig_model(*args[0], **kwargs[0])

            self._compile()
            self._set_train()
            self.compiled = True

        # TODO: evaluation pass
        self.init_runner()

        if self.option.no_side_effect:
            tokens: List[PipeTensor] = []
            for micro_args, micro_kwargs in zip(args, kwargs):
                # token = self.opt_model(micro_args[0])
                token = self.opt_model(*micro_args, **micro_kwargs)
                tokens.append(token)

            pipe_graph = self.runner.pipe_graph
            out_node = pipe_graph.nodes[-1]
            out_worker = self.workers[out_node.rank]

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
        else:
            targets = to_device(targets, self.devices[-1])

            for m_args, m_kwargs, m_targets in zip(args, kwargs, targets):
                self.runner_thread_pool.add_input(m_args, m_kwargs, m_targets)

            for worker in self.workers:
                worker.run()

            # wait until
            self.runner_thread_pool.join()

            loss = sum(self.runner_thread_pool.losses)

        self.record_times()

        return loss
