from typing import List, Dict
import operator 

import torch
from torch.fx import GraphModule, Node
from torch.distributed.rpc import PyRRef
from .rpc_worker import RPCWorker

class PipeNode:
    def __init__(self, idx: int):
        self.idx = idx
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

class PipeInput(PipeNode):
    pass

class PipeWorker(PipeNode):
    def __init__(self, idx, worker: RPCWorker):
        super().__init__(idx)
        self.worker = worker

class PipeOutput(PipeNode):
    pass

class PipeEdge:
    def __init__(self, start_node: PipeNode, item_idx = None):
        self.start_node = start_node
        self.end_node = None
        self.idx = item_idx # None: no getitem / int: getitem(idx)

    def connect(self, end_node: PipeNode):
        self.end_node = end_node


class PipeGraph:
    def __init__(self, gm: GraphModule, rpc_workers: List[PyRRef]):
        self.workers = rpc_workers
        self.input_node = PipeInput(0)
        self.output_node = PipeOutput(len(gm.graph.nodes) - 1)
        self.nodes: List[PipeNode] = [self.input_node] # topo-sorted list of nodes
        self.node_dict: Dict[str, PipeNode] = dict()
        self.edge_dict: Dict[str, PipeEdge] = dict()

        for node in gm.graph.nodes:
            node: Node
            if node.op == "placeholder": # input 
                self.node_dict[node.name] = node
            elif node.op == "call_module": # worker
                pass # TODO
            elif node.op == "call_function" and node.target == operator.getitem: # getitem
                src = node.args[0]
                idx = node.args[1]
                assert (src.name in self.node_dict)
                src_node = self.node_dict[src.name]
                edge = PipeEdge(src_node, idx)
                src_node.append_out(edge, idx)
                self.edge_dict[node.name] = edge
            elif node.op == "output":
                pass # TODO
            else: 
                print(f"cannot handle {node.op} from PipeGraph")

        self.nodes.append(self.output_node)


