from typing import Dict, List, Optional, Set

import torch.fx as fx

from .computation_graph import CompNode, CompEdge, CompGraph
from ..device_bench import DeviceBench
from ..utils import fmem


class PipeNode:
    def __init__(self, idx: int, rank: int = 0):
        self.idx = idx
        self.rank = rank

        self.weight = 0
        self.act_mem = 0
        self.temp_mem = 0

        self.act_back = 0
        self.temp_back = 0

        self.comp_time = 0
        self.back_time = 0
        self.optim_time = 0

        self.send_weight = 0

        self.nodes: List[CompNode] = []
        self.params: Set[int] = set()

        self.placeholders: Set[fx.Node] = set()
        self.outputs: Set[fx.Node] = set()
        self.get_attrs: Set[fx.Node] = set()
        self.call_modules: Set[str] = set()

        self.in_edges: List["PipeEdge"] = []
        self.out_edges: List["PipeEdge"] = []

    def mem_diff(self):
        return self.weight + self.act_mem

    def compute_mem(self, fb_ratio):
        self.weight = 0
        self.act_mem = 0
        self.temp_mem = 0
        self.comp_time = 0
        self.optim_time = 0
        self.params.clear()
        self.placeholders.clear()
        self.get_attrs.clear()
        self.outputs.clear()

        max_temp = 0
        for cnode in self.nodes:
            self.weight += cnode.weight
            self.act_mem += cnode.act_mem
            if max_temp < self.act_mem + cnode.temp_mem:
                max_temp = self.act_mem + cnode.temp_mem
            self.comp_time += cnode.comp_time
            self.optim_time += cnode.optim_time

            self.params.update(cnode.use_params)
            self.placeholders.update(cnode.placeholders)
            self.get_attrs.update(cnode.get_attrs)
            self.call_modules.update(cnode.call_modules)
            self.outputs.update(cnode.outputs)

        self.temp_mem = max_temp

        # TODO: compute real back time and mem
        self.act_back = -self.act_mem
        self.temp_back = max(self.temp_mem - self.act_mem, 0)
        self.back_time = self.comp_time * fb_ratio

        self.send_weight = 0
        for edge in self.out_edges:
            edge.weight = 0
            for cedge in edge.comp_edges:
                edge.weight += cedge.weight
            self.send_weight += edge.weight

    def to_str(self):
        return f"Node [{self.idx}/{self.rank}] (weight: {fmem(self.weight)}, act_mem: {fmem(self.act_mem)}, temp_mem: {fmem(self.temp_mem)}, comp_time: {self.comp_time}, back_time: {self.back_time}, optim_time: {self.optim_time}, node len: {len(self.nodes)}, send weight: {fmem(self.send_weight)})"


class PipeEdge:
    def __init__(self, start_node: PipeNode, end_node: PipeNode):
        self.start_node: PipeNode = start_node
        self.end_node: PipeNode = end_node
        self.comp_edges: List[CompEdge] = []

        self.weight = 0
        self.send_time = 0

    def assign(self):
        self.start_node.out_edges.append(self)
        self.end_node.in_edges.append(self)

    def set_send_time(self, device_bench: Optional[DeviceBench] = None):
        """return send time in ms"""
        if device_bench is None:
            return 0

        dev1 = device_bench.devices[self.start_node.rank]
        dev2 = device_bench.devices[self.end_node.rank]
        if dev1 == dev2 or (dev1, dev2) not in device_bench.send_fn_map:
            return 0
        self.send_time = device_bench.send_time_rank(
            self.start_node.rank, self.end_node.rank, self.weight
        )
        return self.send_time

    def to_str(self, device_bench: Optional[DeviceBench] = None):
        rank1 = self.start_node.rank
        rank2 = self.end_node.rank
        if device_bench is not None:
            rank1 = device_bench.devices[rank1]
            rank2 = device_bench.devices[rank2]

        return f"Edge ({self.start_node.idx} -> {self.end_node.idx}) (weight: {fmem(self.weight)}, device: {rank1} -> {rank2}, weight: {fmem(self.weight)}, send time: {self.send_time})"


class PipeGraph:
    def __init__(self, comp_graph: CompGraph):
        self.comp_graph = comp_graph

        self.input_nodes: List[PipeNode] = []
        self.output_nodes: List[PipeNode] = []

        self.nodes: List[PipeNode] = []
        self.edges: List[PipeEdge] = []
        self.back_dict: Dict[CompNode, PipeNode] = dict()

        # weight + IO
        self.device_mem: List[int] = []

    def to_str(self):
        str_list = [self.input_node.to_str()]
        for n in self.internal_nodes:
            str_list.append(n.to_str())
        str_list.append(self.output_node.to_str())

        return "\n".join(str_list)

    def print_graph(self):
        print(f"GRAPH: {len(self.nodes)} nodes / {len(self.edges)} edges")
        print("==== nodes ====")
        for n in self.nodes:
            print(n.to_str())

        print("==== edges ====")
        for n in self.nodes:
            for e in n.out_edges:
                print(e.to_str())

    def topo_sort(self):
        sorted_nodes = []
        visited = set()

        def dfs(node: PipeNode):
            if node in visited:
                return
            visited.add(node)
            for e in node.out_edges:
                dfs(e.end_node)
            sorted_nodes.append(node)

        for n in self.nodes:
            dfs(n)

        self.nodes = sorted_nodes
        for idx, n in enumerate(self.nodes):
            n.idx = idx

        return self.nodes

    def get_peak_memory(self, sched):
        rank_peak_mem = []
        rank_weight = self.device_mem.copy()

        for weight, rank_sched in zip(rank_weight, sched):
            curr_mem = weight
            max_peak_mem = weight
            for idx, _, is_forward in rank_sched:
                pnode = self.nodes[idx]
                if is_forward:
                    # curr_mem += sum(e.weight for e in pnode.in_edges) + sum(
                    #     e.weight for e in pnode.out_edges
                    # )
                    peak_mem = curr_mem + pnode.temp_mem
                    curr_mem += pnode.act_mem
                else:
                    peak_mem = (
                        curr_mem
                        + (pnode.temp_mem - pnode.act_mem)
                        # + sum(e.weight for e in pnode.out_edges)  # grad
                        # + sum(e.weight for e in pnode.in_edges)  # input grad
                    )
                    curr_mem -= (
                        pnode.act_mem
                        # + sum(e.weight for e in pnode.in_edges)
                        # + sum(e.weight for e in pnode.out_edges)
                    )

                if peak_mem > max_peak_mem:
                    max_peak_mem = peak_mem

            rank_peak_mem.append(max_peak_mem)

        return rank_peak_mem

    def get_value(self, sched, batch_cnt: int, return_dist=False):
        """
        Valuate the pipe graph: returns (total_latency, List[Peak memory])

        We suppose that the schedule will follow synchronous 1f1b with
        the length of initial batch cluster is device_cnt, which consumes the least memory

        The total latency can be calculated by DAG's longest path algorithm
        edge_weight = node comp_time + communication latency
        """

        graph = dict()

        sink = (-1, -1, False)
        graph[sink] = []
        optim_times = [0 for _ in range(len(self.device_mem))]

        # dependency between nodes
        for pidx, pnode in enumerate(self.nodes):
            optim_times[pnode.rank] += pnode.optim_time

            for bidx in range(batch_cnt):
                # Forward
                f_vert = (pidx, bidx, True)
                b_vert = (pidx, bidx, False)

                f_edges = []
                b_edges = []

                for out_edge in pnode.out_edges:
                    end_node = out_edge.end_node
                    time = pnode.comp_time + out_edge.send_time
                    next_node = (end_node.idx, bidx, True)
                    f_edges.append((next_node, time))

                for in_edge in pnode.in_edges:
                    start_node = in_edge.start_node
                    time = pnode.back_time + in_edge.send_time
                    next_node = (start_node.idx, bidx, False)
                    b_edges.append((next_node, time))

                graph[f_vert] = f_edges
                graph[b_vert] = b_edges

        # dependency between schedules within a single device
        for dev_idx, rank_sched in enumerate(sched):
            for i in range(len(rank_sched) - 1):
                prev_v = rank_sched[i]
                next_v = rank_sched[i + 1]

                pnode = self.nodes[prev_v[0]]
                if prev_v[2]:
                    graph[prev_v].append((next_v, pnode.comp_time))
                else:
                    graph[prev_v].append((next_v, pnode.back_time))

            last_v = rank_sched[-1]
            pnode = self.nodes[last_v[0]]
            optim_time = optim_times[dev_idx]
            graph[last_v].append((sink, pnode.back_time + optim_time))

        # Step 1: Topological sorting
        visited = {node: False for node in graph}
        stack = []

        # for n, k in graph.items():
        #     print(n, k)

        def topological_sort(node, visited, stack):
            visited[node] = True
            if node in graph:
                for neighbor, _ in graph[node]:
                    if not visited[neighbor]:
                        topological_sort(neighbor, visited, stack)
            stack.insert(0, node)

        for node in graph:
            if not visited[node]:
                topological_sort(node, visited, stack)

        start_node = (0, 0, True)

        # Step 2: Initialize distances
        distances = {node: float("-inf") for node in graph}
        distances[start_node] = 0

        # Step 3: Traverse the graph
        predecessors = {}
        for node in stack:
            if node in graph:
                for neighbor, weight in graph[node]:
                    new_distance = distances[node] + weight
                    if new_distance > distances[neighbor]:
                        distances[neighbor] = new_distance
                        predecessors[neighbor] = node

        # Step 4: Find the longest path
        longest_distance = distances[sink]

        if return_dist:
            return longest_distance, distances
        else:
            return longest_distance
