from typing import List, Dict
import operator

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from torch.distributed.rpc import PyRRef
from .rpc_worker import RPCWorker


class PipeNode:
    def __init__(self, rank: int):
        self.rank = rank  # 0: master/io, 1+: worker
        self.in_edges: List["PipeEdge"] = []
        self.out_edges: List["PipeEdge"] = []
        self.device = "cpu"

    def append_in(self, edge: "PipeEdge", idx: int = -1):
        self._append(self.in_edges, edge, idx)

    def append_out(self, edge: "PipeEdge", idx: int = -1):
        self._append(self.out_edges, edge, idx)

    def _append(self, edge_list, edge, idx):
        if idx < 0:
            edge_list.append(edge)
        else:
            if len(edge_list) <= idx:
                for _ in range(len(edge_list), idx + 1):
                    edge_list.append(None)
            edge_list[idx] = edge

    def set_device(self, device: str):
        self.device = device


class PipeIO(PipeNode):
    pass


class PipeWorker(PipeNode):
    def __init__(self, rank: int, worker=None):
        super().__init__(rank)
        self.worker = worker

    def set_worker(self, worker: RPCWorker):
        self.worker = worker


class PipeEdge:
    def __init__(self, start_node: PipeNode, item_idx=None):
        self.start_node = start_node
        self.end_nodes: List[PipeNode] = []
        self.idx = item_idx  # None: no getitem / int: getitem(idx)

    def connect(self, node: PipeNode):
        self.end_nodes.append(node)


class PipeGraph:
    def __init__(self, rpc_workers: List[PyRRef], gm: GraphModule = None):
        self.worker_cnt = len(rpc_workers)
        self.workers = rpc_workers

        if gm is not None:
            self.build_graph(gm)

    def build_graph(self, gm: GraphModule):
        self.input_node = PipeIO(0)
        self.output_node = PipeIO(0)
        self.internal_nodes: List[PipeWorker] = []  # topo-sorted list of nodes
        self.node_dict: Dict[str, PipeNode] = dict()
        self.edge_dict: Dict[str, PipeEdge] = dict()

        node_cnt = 0
        input_cnt = 0
        for node in gm.graph.nodes:
            node: Node
            if node.op == "placeholder":  # input
                edge = PipeEdge(self.input_node, input_cnt)
                input_cnt += 1
                self.edge_dict[node.name] = edge
                self.output_node.append_out(edge)
            elif node.op == "call_module":  # worker
                node_cnt += 1
                rank = (node_cnt // self.worker_cnt) + 1
                worker = PipeWorker(rank)

                self.internal_nodes.append(worker)
                self.node_dict[node.name] = worker
                for idx, arg in enumerate(node.args):
                    if arg.name in self.node_dict:
                        in_node = self.node_dict[arg.name]
                        edge = PipeEdge(in_node)
                        edge.connect(worker)
                        worker.append_in(edge, idx)
                        in_node.append_out(edge)
                    elif arg.name in self.edge_dict:
                        edge = self.edge_dict[arg.name]
                        edge.connect(worker)
                        worker.append_in(edge, idx)
                    else:
                        print(
                            f"cannot handle input argument {in_node.name} of node {node.name}"
                        )
            elif (
                node.op == "call_function" and node.target == operator.getitem
            ):  # getitem
                src = node.args[0]
                idx = node.args[1]
                assert src.name in self.node_dict
                src_node = self.node_dict[src.name]
                edge = PipeEdge(src_node, idx)
                src_node.append_out(edge)
                self.edge_dict[node.name] = edge
            elif node.op == "output":
                for idx, arg in enumerate(node.args):
                    if arg.name in self.node_dict:
                        in_node = self.node_dict[arg.name]
                        edge = PipeEdge(in_node)
                        edge.connect(self.output_node)
                        self.output_node.append_in(edge, idx)
                        in_node.append_out(edge)
                    elif arg.name in self.edge_dict:
                        edge = self.edge_dict[arg.name]
                        edge.connect(self.output_node)
                        self.output_node.append_in(edge, idx)
                    else:
                        print(f"cannot handle output node {arg.name}")
            else:
                print(f"cannot handle {node.op} from PipeGraph")

        self.input_node.rank = self.internal_nodes[0].rank
        self.output_node.rank = self.internal_nodes[-1].rank
