from typing import List, Any, Tuple
from enum import Enum

import torch

from .trace import StepTrace


class Step(Enum):
    IDLE = 1
    FORWARD = 2
    BACKWARD = 3
    OPTIMIZER_STEP = 4
    # TODO: do we need send/recv/ddp_sync?


class SyncScheduler:
    """
    Default Synchronous scheduler
    """

    def __init__(self, device_cnt: int, batch_cnt: int):
        self.device_cnt = device_cnt
        self.batch_cnt = batch_cnt
        self.sched = self.get_sched()

    def _sync_gpus(self):
        for device_id in range(self.device_cnt):
            torch.cuda.synchronize(f"cuda:{device_id}")

    def get_sched(self):
        raise NotImplementedError()

    def exec(self, traces: List[StepTrace], targets: List[Any], loss_fn):
        losses = [None for _ in traces]
        futures = [None for _ in traces]

        for step_sched in self.sched:
            # TODO: async run by RPC
            self._sync_gpus()

            for step, dev_no, batch_no in step_sched:
                print(step, dev_no, batch_no)
                if step == Step.IDLE:
                    continue

                curr_trace = traces[batch_no]

                if step == Step.FORWARD:
                    curr_trace.forward_step()
                    if dev_no == self.device_cnt - 1:
                        losses[batch_no] = curr_trace.forward_loss(
                            targets[batch_no], loss_fn
                        )

                elif step == Step.BACKWARD:
                    if dev_no == self.device_cnt - 1:
                        loss = losses[batch_no].wait()
                        losses[batch_no] = loss
                    futures[batch_no] = curr_trace.backward_step()

        for f in futures:
            f.wait()

        return losses


class SchedGPipe(SyncScheduler):
    def __init__(self, device_cnt: int, batch_cnt: int):
        super().__init__(device_cnt, batch_cnt)

    def get_sched(self):
        device_cnt = (
            self.device_cnt if self.device_cnt <= self.batch_cnt else self.batch_cnt
        )
        batch_cnt = self.batch_cnt
        self.device_cnt = device_cnt

        # (Step, device_no, microbatch_no)
        sched: List[List[Tuple[Step, int, int]]] = []

        # calculate the width of trapozide
        step_width = batch_cnt + device_cnt - 1

        # forward
        forward_sched = []
        for step_id in range(step_width):
            step_sched = []

            rng = (max(0, step_id - batch_cnt + 1), min(device_cnt, step_id + 1))
            max_batch = min(step_id, batch_cnt - 1)

            for pre_idle in range(0, rng[0]):
                step_sched.append((Step.IDLE, pre_idle, -1))

            for forward in range(rng[0], rng[1]):
                step_sched.append((Step.FORWARD, forward, max_batch))
                max_batch -= 1

            for post_idle in range(rng[1], device_cnt):
                step_sched.append((Step.IDLE, post_idle, -1))

            forward_sched.append(list(reversed(step_sched)))

        # backward
        sched.extend(forward_sched)
        for forward_step in reversed(forward_sched):
            step_sched = []
            for (step, dev_no, micro_no) in reversed(forward_step):
                if step == Step.FORWARD:
                    step = Step.BACKWARD
                step_sched.append((step, dev_no, micro_no))
            sched.append(step_sched)

        # TODO: add optimizer step

        return sched


class Sched1F1B(SyncScheduler):
    def __init__(self, device_cnt: int, batch_cnt: int):
        super().__init__(device_cnt, batch_cnt)

    def get_sched(self):
        device_cnt = (
            self.device_cnt if self.device_cnt <= self.batch_cnt else self.batch_cnt
        )
        batch_cnt = self.batch_cnt
        self.device_cnt = device_cnt

        # (Step, device_no, microbatch_no)
        sched: List[List[Tuple[Step, int, int]]] = []

        # calculate the width of trapozide
        step_width = batch_cnt + device_cnt - 1

        # forward
        forward_sched = []
        for step_id in range(step_width):
            step_sched = []

            rng = (min(0, step_id - batch_cnt + 1), max(device_cnt, step_id))
            max_batch = min(step_id, batch_cnt - 1)

            for pre_idle in range(0, rng[0]):
                step_sched.append((Step.IDLE, pre_idle, -1))

            for forward in range(rng[0], rng[1]):
                step_sched.append((Step.FORWARD, forward, max_batch))
                max_batch -= 1

            for post_idle in range(range[1], device_cnt):
                step_sched.append((Step.IDLE), post_idle, -1)

            forward_sched.append(step_sched)

        # backward
        sched.extend(forward_sched)
        for forward_step in reversed(forward_sched):
            step_sched = []
            for (step, dev_no, micro_no) in reversed(forward_step):
                if step == Step.FORWARD:
                    step = Step.BACKWARD
                step_sched.append((step, dev_no, micro_no))
            sched.append(step_sched)

        # TODO: add optimizer step

        return sched
