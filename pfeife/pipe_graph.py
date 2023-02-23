import operator
from typing import Dict, List

from torch.fx import GraphModule, Node


class PipeNode:
    def __init__(self, idx: int, rank: int = 0, is_io=False):
        self.idx = idx
        self.rank = rank  # 0: unassigned/master, 1+: worker
        self.in_edges: List["PipeEdge"] = []
        self.out_edges: List["PipeEdge"] = []
        self.device = "cpu"
        self.is_io = is_io

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

    def to_str_tuple(self):
        if self.is_io:
            return f"[{self.idx}, {self.rank}]"
        else:
            return f"({self.idx}, {self.rank})"

    def to_str(self):
        s = self.to_str_tuple()
        indent = len(s) + 4
        for i, e in enumerate(self.in_edges):
            s += " <- " if i == 0 else ("\n" + " " * indent)
            s += f"{e.start_node.to_str_tuple()} <{e.idx}> "
        for i, e in enumerate(self.out_edges):
            if i == 0 and len(self.in_edges) > 0:
                s += "\n" + " " * (indent - 4)
            for j, n in enumerate(e.end_nodes):
                s += " -> " if (i + j) == 0 else ("\n" + " " * indent)
                s += f"{n.to_str_tuple()} <{e.idx}> "
        return s


class PipeEdge:
    def __init__(self, start_node: PipeNode, item_idx=None):
        self.start_node = start_node
        self.end_nodes: List[PipeNode] = []
        self.idx = item_idx  # None: no getitem / int: getitem(idx)

    def connect(self, node: PipeNode):
        self.end_nodes.append(node)


class PipeGraph:
    def __init__(self, worker_cnt: int, gm: GraphModule = None):
        self.worker_cnt = worker_cnt

        if gm is not None:
            self.build_graph(gm)

    def to_str(self):
        str_list = [self.input_node.to_str()]
        for n in self.internal_nodes:
            str_list.append(n.to_str())
        str_list.append(self.output_node.to_str())

        return "\n".join(str_list)

    def build_graph(self, gm: GraphModule):
        self.input_node = PipeNode(0, 0, is_io=True)
        self.output_node = PipeNode(0, 0, is_io=True)
        self.internal_nodes: List[PipeNode] = []  # topo-sorted list of nodes
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
                self.input_node.append_out(edge)
            elif node.op == "call_module":  # worker
                node_cnt += 1
                worker = PipeNode(node_cnt)

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
                args = node.args[0]
                if type(args) != tuple:
                    args = [args]

                for idx, arg in enumerate(args):
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

        self.output_node.idx = len(self.internal_nodes) + 1
