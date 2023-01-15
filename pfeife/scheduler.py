from typing import List, Any, Tuple
from enum import Enum

import torch

from .trace import Trace


class Step(Enum):
    IDLE = 1
    FORWARD = 2
    BACKWARD = 3
    OPTIMIZER_STEP = 4
    # TODO: do we need send/recv/ddp_sync?


def SchedGPipe():
    """
    Default GPipe scheduler
    """

    def __init__(self, traces: List[Trace], device_cnt: int, batch_cnt: int):
        self.traces = traces
        self.device_cnt = device_cnt
        self.batch_cnt = batch_cnt
        self.sched = self.get_sched()

    def _sync_gpus(self):
        for device_id in range(self.device_cnt):
            torch.cuda.synchronize(f"cuda:{device_id}")

    def get_sched(self):
        device_cnt = (
            self.device_cnt if self.device_cnt <= self.batch_cnt else self.batch_cnt
        )
        batch_cnt = self.batch_cnt

        # (Step, device_no, microbatch_no)
        sched: List[Tuple[Step, int, int]] = []

        return sched

    def exec(self, targets: List[Any], loss_fn):
        raise NotImplementedError()


def Sched1F1B(SchedGPipe):
    def __init__(self, traces: List[Trace], device_cnt: int, batch_cnt: int):
        super(Sched1F1B, self).__init__(traces, device_cnt, batch_cnt)

    def get_sched(self):
        pass
