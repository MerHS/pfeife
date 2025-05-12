import os
import time
from typing import List, Union
from collections import defaultdict

import torch
import torch._dynamo as dynamo
import torch.fx as fx
from torch.nn.modules.loss import _Loss

from .net_states import get_state
from ..batch import split_args_kwargs
from ..loss import LossWrapper
from ..option import PipeOption
from .runner import RunnerThreadPool, PipeGraphRunner
from .worker import LoopFlag
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

        self.net_state = get_state()
        self.main_device = self.net_state.device
        self.is_graph_fetched = False

        self.set_option(option)

    def set_option(self, option: PipeOption):
        self.debug(f"got option: {option.__dict__}")

        self.option = option
        self.batch_cnt = option.batch_cnt
        self.device_cnt = option.device_cnt
        self.rank = self.net_state.rank

        # reset option based on net_state
        # option.device_cnt = self.device_cnt
        option.device_map = None  # TODO: set device_map

        self.clear_record()

        self.runner = PipeGraphRunner(option)
        self.compiled = False

    def debug(self, msg):
        self.logger.debug("[Master] %s", msg)

    def info(self, msg):
        self.logger.info("[Master] %s", msg)

    def _compile(self, test_args, test_kwargs):
        # Assign the model to the first device so that the usage of the device is not changed
        # TODO: what if the middle of the model uses also inlined device? (tensor.to(some_params.device))
        # some tensor-generating functions like torch.ones(..., device=...) will bake parameters
        dynamo.reset()
        # self.orig_model = self.orig_model.to(self.devices[0])

        graph_lens = []

        def compile_fn(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            # TODO: what if this function is called two times from a training time?
            # ANSWER: return new handler for each graph.

            self.runner.set_graph_module(gm)
            graph_lens.append(len(gm.graph.nodes))

            if len(graph_lens) > 1:
                raise RuntimeError(
                    f"multiple graph detected ({graph_lens}): cannot be run on pfeife."
                )

            return self.runner

        dynamo_ctx = dynamo.optimize(compile_fn)
        self.opt_model = dynamo_ctx(self.orig_model)
        # self.opt_model = torch.compile(self.orig_model, backend=compile_fn)
        self.opt_model(*test_args, **test_kwargs)

        self.runner_thread_pool = RunnerThreadPool(
            self.opt_model, self.runner, self.loss_fn
        )

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def _set_train(self):
        # TODO: set train
        if self.is_train:
            self.opt_model.train()
            # for worker in self.workers:
            #     worker.set_train(True)
        else:
            self.opt_model.eval()
            # for worker in self.workers:
            #     worker.set_train(False)

    def _init_runner(self):
        self.runner_thread_pool.reset()
        self.runner.prepare_workers(LoopFlag.CONTINUE_LOOP)

    def close(self):
        self.runner.close()

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

        # TODO: do not split args at the worker processes
        args, kwargs = split_args_kwargs(args, kwargs, self.batch_cnt)
        targets, _ = split_args_kwargs([target], dict(), self.batch_cnt)
        targets = [t[0] for t in targets]  # untuple

        # TODO: sample inputs

        if not self.compiled:
            self._compile(args[0], kwargs[0])
            # self._set_train() # TODO
            self.compiled = True

        # TODO: evaluation pass
        self._init_runner()

        targets = to_device(targets, self.main_device)

        for m_args, m_kwargs, m_targets in zip(args, kwargs, targets):
            self.runner_thread_pool.add_input(m_args, m_kwargs, m_targets)

        # wait until
        self.runner_thread_pool.join()
        loss = sum(self.runner_thread_pool.losses)

        self.debug("manager done")

        # TODO: record times
        # self.record_times()

        return loss
