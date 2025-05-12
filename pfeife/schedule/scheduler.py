import functools
from typing import List, Tuple, Dict
from enum import Enum
from dataclasses import dataclass

# from .solver import GraphSolver


class StepWork(Enum):
    # if the sendee is an output node,
    # 1) send the result to output_node.rank if the current rank differs to the output node or,
    # 2) save the result to the output buffer
    SEND_ACT = 1
    RECV_ACT = 2
    FORWARD = 3
    SEND_GRAD = 4

    # receive output from another rank (if exists) and calculate loss if the source of grad is the output node
    RECV_GRAD = 5
    BACKWARD = 6
    OPTIMIZER_STEP = 7


@dataclass
class Step:
    work: StepWork
    node_id: int
    batch_id: int

    def __str__(self):
        return f"{self.work} / (node {self.node_id} of batch {self.batch_id})"


class PipeSchedParams:
    def __init__(
        self,
        device_cnt,
        batch_cnt,
        node_cnt,
        loop_batch,
        prefetch_fwd,
    ):
        self.device_cnt: int = device_cnt
        self.batch_cnt: int = batch_cnt
        self.node_cnt: int = node_cnt
        self.loop_batch: int = loop_batch
        self.prefetch_fwd: int = prefetch_fwd

        assert batch_cnt >= loop_batch, "batch_cnt should be greater than loop_batch"

        # for each device, list of (node_idx, batch_id, is_fwd)
        self.rank_works: List[List[Tuple[int, int, bool]]]

        self._build_sched()

    def _build_sched(self):
        batch_cnt = self.batch_cnt
        device_cnt = self.device_cnt
        node_cnt = self.node_cnt
        loop_batch = self.loop_batch
        prefetch_fwd = self.prefetch_fwd

        rank_works = [[] for _ in range(device_cnt)]

        rank_fwd = [[] for _ in range(device_cnt)]
        rank_bwd = [[] for _ in range(device_cnt)]

        for loop_start in range(0, batch_cnt, loop_batch):
            loop_cnt = (loop_start - 1) % loop_batch + 1

            for node_idx in range(node_cnt):
                rank = node_idx % device_cnt
                fwd = rank_fwd[rank]
                for batch_id in range(
                    loop_start, min(loop_start + loop_cnt, batch_cnt)
                ):
                    fwd.append((node_idx, batch_id))

            for node_idx in reversed(range(node_cnt)):
                rank = node_idx % device_cnt
                bwd = rank_bwd[rank]
                for batch_id in range(
                    loop_start, min(loop_start + loop_cnt, batch_cnt)
                ):
                    bwd.append((node_idx, batch_id))

        for rank in range(device_cnt):
            rank_sched = rank_works[rank]
            fwd = rank_fwd[rank]
            bwd = rank_bwd[rank]

            curr_bwd = 0
            max_fwd_idx = max(fwd, key=lambda x: x[0])[0]
            last_dist = node_cnt - max_fwd_idx
            alter_idx = (
                (node_cnt - last_dist) // device_cnt * loop_batch
                + last_dist
                + min(last_dist - 1, prefetch_fwd)
                - 1
            )
            for iter_idx, (fwd_idx, fwd_batch) in enumerate(fwd):
                rank_sched.append((fwd_idx, fwd_batch, True))
                if iter_idx >= alter_idx:
                    bwd_idx, bwd_batch = bwd[curr_bwd]
                    rank_sched.append((bwd_idx, bwd_batch, False))
                    curr_bwd += 1

            for bwd_idx, bwd_batch in bwd[curr_bwd:]:
                rank_sched.append((bwd_idx, bwd_batch, False))

        self.rank_works = rank_works

    def __str__(self):
        return f"PipeSched(node_cnt={self.node_cnt}, loop_batch={self.loop_batch}, prefetch_fwd={self.prefetch_fwd})"

    def __repr__(self):
        return self.__str__()


class PipeSchedSet:
    def __init__(self, device_cnt, batch_cnt, node_cnt):
        self.device_cnt: int = device_cnt
        self.batch_cnt: int = batch_cnt
        self.node_cnt: int = node_cnt

        self.scheds: List[PipeSchedParams] = []
        self.builder = functools.partial(
            PipeSchedParams, device_cnt, batch_cnt, node_cnt
        )
        self._build_set()

    def _build_set(self):
        node_cnt = self.node_cnt
        batch_cnt = self.batch_cnt
        device_cnt = self.device_cnt
        builder = self.builder

        if node_cnt > device_cnt:  # looped
            if batch_cnt % device_cnt != 0:
                print(
                    "WARNING: batch_cnt is not divisible by device_cnt for a looped schedule.\nThis may increase internal pipeline bubble."
                )

            for loop_batch in range(device_cnt, batch_cnt + 1):
                if loop_batch > device_cnt and batch_cnt % loop_batch != 0:
                    continue
                for prefetch_fwd in range(0, device_cnt):
                    self.scheds.append(builder(loop_batch, prefetch_fwd))
        else:
            for prefetch_fwd in range(0, device_cnt):
                self.scheds.append(builder(batch_cnt, prefetch_fwd))


class Scheduler:
    """
    Default Synchronous scheduler
    """

    def __init__(self, sched_params: PipeSchedParams):
        self.sched_params = sched_params
        self.node_cnt = sched_params.node_cnt
        self.device_cnt = sched_params.device_cnt
        self.batch_cnt = sched_params.batch_cnt

        # self.cluster: list of (node indices for each rank)
        # self.sched: list of List[Step]
        self.steps = self._build_steps()

    def _build_steps(self):
        # returns node clusters and steps
        raise NotImplementedError()

    def get_train_steps(self, worker_id: int) -> List[Step]:
        if worker_id >= len(self.steps):
            return []
        return self.steps[worker_id]

    def get_eval_steps(self, worker_id: int) -> List[Step]:
        # TODO: implement it
        pass

    def _add_forward(self, rank_sched: List[Step], node_ids, batch_id):
        if isinstance(node_ids, int):
            node_ids = [node_ids]

        for node_id in node_ids:
            rank_sched.append(Step(StepWork.RECV_ACT, node_id, batch_id))
            rank_sched.append(Step(StepWork.FORWARD, node_id, batch_id))
            rank_sched.append(Step(StepWork.SEND_ACT, node_id, batch_id))

        return rank_sched

    def _add_backward(self, rank_sched: List[Step], node_ids, batch_id):
        if isinstance(node_ids, int):
            node_ids = [node_ids]

        for node_id in node_ids:
            rank_sched.append(Step(StepWork.RECV_GRAD, node_id, batch_id))
            rank_sched.append(Step(StepWork.BACKWARD, node_id, batch_id))
            rank_sched.append(Step(StepWork.SEND_GRAD, node_id, batch_id))

        return rank_sched


class BeamScheduler(Scheduler):
    def _build_steps(self):
        # rank: (Step, node_no, microbatch_no)
        steps: List[List[Step]] = []

        for rank_job in self.sched_params.rank_works:
            rank_steps = []

            for node_idx, batch_id, is_fwd in rank_job:
                if is_fwd:
                    self._add_forward(rank_steps, node_idx, batch_id)
                else:
                    self._add_backward(rank_steps, node_idx, batch_id)

            steps.append(rank_steps)

        for rank_steps in steps:
            if len(rank_steps) > 0:
                rank_steps.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

        return steps


class ForwardOnlyScheduler(Scheduler):
    def _build_steps(self):
        # rank: (Step, node_no, microbatch_no)
        steps: List[List[Step]] = []

        for rank_job in self.sched_params.rank_works:
            rank_steps = []

            for node_idx, batch_id, is_fwd in rank_job:
                if is_fwd:
                    self._add_forward(rank_steps, node_idx, batch_id)

            steps.append(rank_steps)

        for rank_steps in steps:
            if len(rank_steps) > 0:
                rank_steps.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

        return steps


class SingleCommScheduler(Scheduler):
    def __init__(self, sched_params: PipeSchedParams, dep_graph):
        self.dep_graph = dep_graph

        super().__init__(sched_params)

    def _build_steps(self):
        # rank: (Step, node_no, microbatch_no)
        steps: List[List[Step]] = []

        for rank_job in self.sched_params.rank_works:
            rank_steps = []

            for node_idx, batch_id, is_fwd in rank_job:
                if is_fwd:
                    self._add_forward(rank_steps, node_idx, batch_id)
                else:
                    self._add_backward(rank_steps, node_idx, batch_id)

            steps.append(rank_steps)

        for rank_steps in steps:
            if len(rank_steps) > 0:
                rank_steps.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

        return steps


class NoSkewScheduler(Scheduler):
    def _build_steps(self):
        # rank: (Step, node_no, microbatch_no)
        sched: List[List[Step]] = []

        for rank, rank_pipe_sched in enumerate(self.sched_params.rank_works):
            rank_sched = []

            for node_idx, batch_id, is_fwd in rank_pipe_sched:
                if is_fwd:
                    self._add_forward(rank_sched, node_idx, batch_id)
                else:
                    self._add_backward(rank_sched, node_idx, batch_id)

            sched.append(rank_sched)

        for rank_sched in sched:
            if len(rank_sched) > 0:
                rank_sched.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

        return sched


class BFSScheduler(Scheduler):
    """
    Assume that devices and edges are properly set
    so that the simple linear BFS scheduling can be executed
    """

    def _build_steps(self):
        batch_cnt = self.batch_cnt
        device_cnt = self.device_cnt

        # rank: (Step, node_no, microbatch_no)
        sched: List[List[Step]] = [[] for _ in range(device_cnt)]

        for node_id in range(self.node_cnt):
            rank = node_id % device_cnt
            rank_sched = sched[rank]
            for batch_id in range(batch_cnt):
                self._add_forward(rank_sched, node_id, batch_id)

        for node_id in reversed(range(self.node_cnt)):
            rank = node_id % device_cnt
            rank_sched = sched[rank]
            for batch_id in range(batch_cnt):
                self._add_backward(rank_sched, node_id, batch_id)

        for rank_sched in sched:
            if len(rank_sched) > 0:
                rank_sched.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

        return sched


class DFSScheduler(Scheduler):
    """
    Similar to Synchronous 1F1B scheduler
    """

    def _build_steps(self):
        node_cnt = self.node_cnt
        batch_cnt = self.batch_cnt
        device_cnt = self.device_cnt

        # rank: (Step, node_no, microbatch_no)
        sched: List[List[Step]] = [[] for _ in range(device_cnt)]

        rank_fwd = [[] for _ in range(device_cnt)]
        rank_bwd = [[] for _ in range(device_cnt)]

        for batch_start in range(0, batch_cnt, device_cnt):
            batch_end = min(batch_start + device_cnt, batch_cnt)
            for node_id in range(node_cnt):
                rank = node_id % device_cnt
                fwd = rank_fwd[node_id]
                for batch_id in range(batch_start, batch_end):
                    fwd.append((node_id, batch_id))

            for node in reversed(range(node_cnt)):
                rank = node_id % device_cnt
                bwd = rank_bwd[node_id]
                for batch_id in range(batch_start, batch_end):
                    bwd.append((node_id, batch_id))

        for rank in range(device_cnt):
            rank_sched = sched[rank]
            fwd = rank_fwd[rank]
            bwd = rank_bwd[rank]

            curr_bwd = 0
            for fwd_idx, fwd_batch in fwd:
                self._add_forward(rank_sched, fwd_idx, fwd_batch)
                if fwd_batch >= device_cnt or (fwd_batch + 1 >= node_cnt - fwd_idx):
                    bwd_idx, bwd_batch = bwd[curr_bwd]
                    self._add_backward(rank_sched, bwd_idx, bwd_batch)
                    curr_bwd += 1

            for bwd_idx, bwd_batch in bwd[curr_bwd:]:
                self._add_backward(rank_sched, bwd_idx, bwd_batch)

        for rank_sched in sched:
            if len(rank_sched) > 0:
                rank_sched.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

        return sched


class PipelessScheduler(Scheduler):
    """
    There is no pipeline. Run all the batch at once from a single device
    and then send the results
    """

    def _build_steps(self):
        node_cnt = self.node_cnt
        batch_cnt = self.batch_cnt
        device_cnt = self.device_cnt

        # rank: (Step, node_no, microbatch_no)
        sched: List[List[Step]] = [[] for _ in range(device_cnt)]

        for node_id in range(node_cnt):
            rank = node_id % device_cnt
            rank_sched = sched[rank]
            for batch_id in range(batch_cnt):
                rank_sched.append(Step(StepWork.RECV_ACT, node_id, batch_id))
                rank_sched.append(Step(StepWork.FORWARD, node_id, batch_id))
            for batch_id in range(batch_cnt):
                rank_sched.append(Step(StepWork.SEND_ACT, node_id, batch_id))

        for node_id in reversed(range(node_cnt)):
            rank = node_id % device_cnt
            rank_sched = sched[rank]
            for batch_id in range(batch_cnt):
                rank_sched.append(Step(StepWork.RECV_GRAD, node_id, batch_id))
                rank_sched.append(Step(StepWork.BACKWARD, node_id, batch_id))
            for batch_id in range(batch_cnt):
                rank_sched.append(Step(StepWork.SEND_GRAD, node_id, batch_id))

        for rank_sched in sched:
            if len(rank_sched) > 0:
                rank_sched.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

        return sched
