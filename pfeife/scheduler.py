from typing import List, Tuple
from enum import Enum
from dataclass import dataclass

import torch

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


class Scheduler:
    """
    Default Synchronous scheduler
    """

    def __init__(self, batch_cnt: int, graph: PipeGraph):
        self.batch_cnt = batch_cnt
        self.graph = graph
        self.cluster, self.sched = self._build_sched()

    def _build_sched(self):
        # returns (worker_id: idx list of assigned node, worker_id: List[Step])
        raise NotImplementedError()

    def assign_train_steps_to_workers(self, modules: List[torch.nn.Module]):
        rank_clusters = self.cluster

        devices = []
        for worker_id, cluster in rank_clusters:
            worker = self.graph.workers[worker_id]
            worker_device = worker.rpc_sync().get_device()
            devices.append(worker_device)

            for node_id in cluster:
                node = self.graph.internal_nodes[node_id]
                node.device = worker_device

        input_node = self.graph.input_node
        input_node.device = devices[input_node.rank - 1]
        output_node = self.graph.output_node
        output_node.device = devices[output_node.rank - 1]

        for worker_id, (worker, cluster) in enumerate(
            zip(self.graph.workers, rank_clusters)
        ):
            worker.rpc_sync().set_graph(self.graph)
            worker.rpc_sync().set_scheduler_steps(self.get_train_steps(worker_id))
            for mod_id in cluster:
                module = modules[mod_id]
                worker.rpc_sync().set_module(mod_id, module)

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
