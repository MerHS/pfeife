from typing import List, Tuple
from enum import Enum
from dataclasses import dataclass

from .utils import get_logger
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

        # self.cluster: list of (node indices for each rank)
        # self.sched: list of List[Step]
        self.sched = self._build_sched()

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


class SchedGPipe(Scheduler):
    """
    Equally slice the list of nodes into #worker pieces
    """

    def _build_sched(self):
        graph = self.graph

        nodes = graph.internal_nodes
        node_cnt = len(nodes)
        batch_cnt = self.batch_cnt
        worker_cnt = graph.worker_cnt

        worker_cnt = min(worker_cnt, node_cnt)

        # rank: (Step, node_no, microbatch_no)
        sched: List[List[Step]] = []

        rank_clusters = []
        cls_len = node_cnt // worker_cnt
        cls_rem = node_cnt % worker_cnt
        start_pos = 0

        for i in range(worker_cnt):
            node_worker_len = cls_len + (1 if cls_rem > i else 0)
            for node_id in range(start_pos, start_pos + node_worker_len):
                nodes[node_id].rank = i + 1
            rank_clusters.append(list(range(start_pos, start_pos + node_worker_len)))
            start_pos += node_worker_len

        # calculate the width of trapozide
        for cluster in rank_clusters:
            rank_sched: List[Step] = []

            for batch_id in range(batch_cnt):
                self._add_forward(rank_sched, cluster, batch_id)

            for batch_id in reversed(range(batch_cnt)):
                self._add_backward(rank_sched, reversed(cluster), batch_id)

            rank_sched.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

            sched.append(rank_sched)

        graph.input_node.rank = graph.internal_nodes[0].rank
        graph.output_node.rank = graph.internal_nodes[-1].rank

        return sched


class Sched1F1B(Scheduler):
    """
    Equally slice the list of nodes into #worker pieces
    """

    def _build_sched(self):
        graph = self.graph

        nodes = graph.internal_nodes
        node_cnt = len(nodes)
        batch_cnt = self.batch_cnt
        worker_cnt = graph.worker_cnt

        worker_cnt = min(worker_cnt, node_cnt)

        # rank: (Step, node_no, microbatch_no)
        sched: List[List[Step]] = []

        rank_clusters = []
        cls_len = node_cnt // worker_cnt
        cls_rem = node_cnt % worker_cnt
        start_pos = 0

        for i in range(worker_cnt):
            node_worker_len = cls_len + (1 if cls_rem > i else 0)
            for node_id in range(start_pos, start_pos + node_worker_len):
                nodes[node_id].rank = i + 1
            rank_clusters.append(list(range(start_pos, start_pos + node_worker_len)))
            start_pos += node_worker_len

        for worker_id, cluster in enumerate(rank_clusters):
            rank_sched: List[Step] = []

            if worker_id == worker_cnt - 1:
                for batch_id in range(batch_cnt):
                    self._add_forward(rank_sched, cluster, batch_id)
                    self._add_backward(rank_sched, cluster, batch_id)
            else:
                # warmup forward
                for batch_id in range(worker_cnt - worker_id):
                    self._add_forward(rank_sched, cluster, batch_id)

                for bw_batch_id in range(batch_cnt):
                    self._add_backward(rank_sched, cluster, bw_batch_id)

                    fw_batch_id = bw_batch_id + worker_cnt - worker_id

                    if fw_batch_id < batch_cnt:
                        self._add_forward(rank_sched, cluster, fw_batch_id)

            rank_sched.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

            sched.append(rank_sched)

        graph.input_node.rank = graph.internal_nodes[0].rank
        graph.output_node.rank = graph.internal_nodes[-1].rank

        return sched


class SchedBFS(Scheduler):
    """
    Equally slice the list of nodes into #worker pieces
    """

    def _build_sched(self):
        graph = self.graph

        nodes = graph.internal_nodes
        node_cnt = len(nodes)
        batch_cnt = self.batch_cnt
        worker_cnt = graph.worker_cnt

        worker_cnt = min(worker_cnt, node_cnt)

        # rank: (Step, node_no, microbatch_no)
        sched: List[List[Step]] = []

        rank_clusters = [[] for _ in range(worker_cnt)]

        for node_id in range(node_cnt):
            rank = (node_id % worker_cnt) + 1
            nodes[node_id].rank = rank
            rank_clusters[rank - 1].append(node_id)

        for cluster in rank_clusters:
            rank_sched: List[Step] = []

            for node_id in cluster:
                for batch_id in range(batch_cnt):
                    self._add_forward(rank_sched, node_id, batch_id)

            for node_id in reversed(cluster):
                for batch_id in reversed(range(batch_cnt)):
                    self._add_backward(rank_sched, node_id, batch_id)

            rank_sched.append(Step(StepWork.OPTIMIZER_STEP, -1, -1))

            sched.append(rank_sched)

        graph.input_node.rank = graph.internal_nodes[0].rank
        graph.output_node.rank = graph.internal_nodes[-1].rank

        return sched


_SCHED_MAP = {"gpipe": SchedGPipe, "1f1b": Sched1F1B, "bfs": SchedBFS}


def get_scheduler(sched_name: str = "gpipe"):
    name = sched_name.lower()

    if name in _SCHED_MAP:
        return _SCHED_MAP[name]
    else:
        raise NameError(f"Unknown scheduler: {sched_name}")
