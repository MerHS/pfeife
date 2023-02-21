from typing import List, Tuple
from enum import Enum
from dataclasses import dataclass

import torch
from torch.distributed.rpc import PyRRef

from .pipe_graph import PipeGraph


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


class Scheduler:
    """
    Default Synchronous scheduler
    """

    def __init__(self, batch_cnt: int, graph: PipeGraph):
        self.batch_cnt = batch_cnt
        self.graph = graph

        # self.cluster: list of (indices of assigned node)
        # self.sched: list of List[Step]
        self.cluster, self.sched = self._build_sched()

    def _build_sched(self):
        # returns node clusters and steps
        raise NotImplementedError()

    def get_train_steps(self, worker_id: int) -> List[Step]:
        if worker_id >= len(self.sched):
            return []
        return self.sched[worker_id]

    def get_eval_steps(self, worker_id: int) -> List[Step]:
        # TODO: implement it
        pass


class SchedGPipe(Scheduler):
    """
    Equally slice the list of nodes into #worker pieces
    """

    def _build_sched(self):
        node_cnt = len(self.graph.internal_nodes)
        batch_cnt = self.batch_cnt
        worker_cnt = self.graph.worker_cnt

        worker_cnt = min(worker_cnt, node_cnt, batch_cnt)

        # rank: (Step, node_no, microbatch_no)
        sched: List[List[Step]] = []

        rank_clusters = []
        cls_len = node_cnt // worker_cnt
        cls_rem = node_cnt % worker_cnt
        start_pos = 0
        for i in range(worker_cnt):
            node_worker_len = cls_len + (1 if cls_rem > i else 0)
            rank_clusters.append(range(start_pos, start_pos + node_worker_len))
            start_pos += node_worker_len

        # calculate the width of trapozide
        for cluster in rank_clusters:
            rank_sched: List[Step] = []

            for batch_id in range(batch_cnt):
                for node_id in cluster:
                    rank_sched.append(Step(StepWork.RECV_ACT, node_id, batch_id))
                    rank_sched.append(Step(StepWork.FORWARD, node_id, batch_id))
                    rank_sched.append(Step(StepWork.SEND_ACT, node_id, batch_id))

            for batch_id in reversed(range(batch_cnt)):
                for node_id in reversed(cluster):
                    rank_sched.append(Step(StepWork.RECV_GRAD, node_id, batch_id))
                    rank_sched.append(Step(StepWork.BACKWARD, node_id, batch_id))
                    rank_sched.append(Step(StepWork.SEND_GRAD, node_id, batch_id))

            rank_sched.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

            sched.append(rank_sched)

        return rank_clusters, sched


def get_scheduler(sched_name: str = "gpipe"):
    if sched_name == "gpipe":
        return SchedGPipe
    else:
        raise NameError(f"Unknown scheduler: {sched_name}")
