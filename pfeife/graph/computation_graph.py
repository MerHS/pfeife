import os
import time
import pickle
import operator
import gc
from dataclasses import dataclass
from collections import namedtuple
from typing import List, Optional, Dict, Set, Tuple, Union, Any
from collections import defaultdict
from copy import deepcopy

import torch
import torch.fx as fx
from torch.fx.node import Node

from ..utils import (
    to_device,
    tree_map,
    tree_filter,
    fmem,
    list_params,
    tree_trav,
    module_set_attr,
)

GRAPH_FETCH_MEM_CAP = 0.6  # 60% of memory cap

ShapeMeta = namedtuple("ShapeMeta", ["size", "dtype", "bytes", "requires_grad"])


class DummyNode:
    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta


@dataclass
class ParamSet:
    """
    Parameter set for set_attr and call_module nodes
    """

    # target -> list of param_idx
    param_set: Dict[str, List[int]]

    # param_idx -> (param meta, main target)
    main_params: List[Tuple[ShapeMeta, str]]

    # moved target -> original target
    moved_targets: Dict[str, str]

    # (target, rank) -> (moved target, is_attr_or_module) (built from partition_graph)
    part_graph_target: Dict[Tuple[str, int], Tuple[str, bool]]

    # moved_target -> shared param_idx (built from partition graph)
    shared_idx: Dict[str, List[int]]

    # shared param_idx -> shared ranks
    shared_rank: Dict[int, Set[int]]


class CompNode:
    def __init__(self, node: Optional[Node], device, device_id=0):
        self.nodes: List[Node] = [node] if node else []
        self.device = device
        self.device_id = device_id

        # targets of placeholder, call_module and get_attr
        self.placeholders: Set[Node] = set()
        self.call_modules: Set[str] = set()
        self.get_attrs: Set[Node] = set()
        self.outputs: Set[Node] = set()

        self.use_params = set()
        self.weight = 0
        self.temp_mem = 0
        self.act_mem = 0
        self.comp_time = 0
        self.optim_time = 0

        self.in_edges: List[CompEdge] = []
        self.out_edges: List[CompEdge] = []

    def add_in_edge(self, edge: "CompEdge"):
        self.in_edges.append(edge)

    def add_out_edge(self, edge: "CompEdge"):
        self.out_edges.append(edge)

    def mem_diff(self):
        return self.weight + self.act_mem


class CompEdge:
    def __init__(
        self,
        start_node: CompNode,
        end_node: CompNode,
        start_idx: int,  # index of self.nodes
    ):
        self.start_node = start_node
        self.end_node = end_node
        self.start_idx = start_idx

        self.weight = 0
        self.shape = None


class CompGraph:
    def __init__(self):
        self.nodes: List[CompNode] = []
        self.edges: List[CompEdge] = []

        self.in_nodes: List[CompNode] = []
        self.out_nodes: List[Tuple[CompNode, int]] = []

        self.placeholders: List[fx.Node] = []
        self.get_attrs: List[fx.Node] = []

        self.output: fx.Node = None

        self.param_set = ParamSet(
            defaultdict(list),
            list(),
            dict(),
            dict(),
            defaultdict(list),
            defaultdict(set),
        )

        # rough backward time / forward time
        self.fb_ratio = 1

    def check_total_order(self):
        # check that the sequence of nodes forms total order
        linear = [n for n in self.nodes if len(n.in_edges) > 0]
        node_idx = {n: i for i, n in enumerate(linear)}
        for edge in self.edges:
            assert node_idx[edge.start_node] < node_idx[edge.end_node]

    def print_branches(self):
        node_idx = {n: i for i, n in enumerate(self.nodes)}
        chain_set = set()
        chain_heads = [(self.nodes[0], e) for e in self.nodes[0].out_edges]

        curr_chain = []
        chain_list = []
        while len(chain_heads) > 0:
            curr_node, curr_edge = chain_heads[0]
            chain_set.add(chain_heads[0])
            curr_chain.append(node_idx[curr_node])

            # print(f"{node_idx[curr_node]}-", end="")

            end_node = curr_edge.end_node
            chain_heads = chain_heads[1:]

            append_chain = False
            if len(end_node.out_edges) == 0:
                # print(f"{node_idx[end_node]}")
                pass
            elif len(end_node.in_edges) > 1:
                # print(f"^{node_idx[end_node]}")
                for e in end_node.out_edges:
                    next_pair = (end_node, e)
                    if next_pair not in chain_set:
                        chain_heads.append(next_pair)
                        chain_set.add(next_pair)
            elif len(end_node.out_edges) == 1:
                append_chain = True
                next_pair = (end_node, end_node.out_edges[0])
                if next_pair not in chain_set:
                    chain_heads.insert(0, next_pair)
                    chain_set.add(next_pair)
            else:
                for e in end_node.out_edges:
                    next_pair = (end_node, e)
                    if next_pair not in chain_set:
                        chain_heads.append(next_pair)
                        chain_set.add(next_pair)

            if not append_chain:
                curr_chain.append(node_idx[end_node])
                chain_list.append(curr_chain)
                curr_chain = []

        if len(curr_chain) > 0:
            chain_list.append(curr_chain)

        chain_list.sort()
        for chain in chain_list:
            print("-".join([str(c) for c in chain]))

    def traverse_linearize(self, mode="save_mem"):
        """
        mode:
        - None/"none": do not sort self.nodes
        - "save_mem": save memory by heuristic
        - "short_first": run short branches (#node is smallest) first
        - "long_first": run long branches (#node is largest) first
        """
        if mode is None or mode == "none":
            return

        node_idx = {n: i for i, n in enumerate(self.nodes)}
        run_order = []
        visit = set()

        def find_runnable():
            runnable = []
            for idx in range(len(self.nodes)):
                if idx in visit:
                    continue
                node = self.nodes[idx]
                can_run = True
                for edge in node.in_edges:
                    if node_idx[edge.start_node] not in visit:
                        can_run = False
                        break
                if can_run:
                    runnable.append([idx])
            for run_thread in runnable:
                curr_node_idx = run_thread[0]
                curr_node = self.nodes[curr_node_idx]
                while True:
                    next_nodes = set([e.end_node for e in curr_node.out_edges])
                    if len(next_nodes) > 1 or len(next_nodes) == 0:
                        break
                    next_node = next_nodes.pop()
                    next_prev_nodes = set(
                        [node_idx[e.start_node] for e in next_node.in_edges]
                    )
                    can_extend = True
                    for prev_idx in next_prev_nodes:
                        if prev_idx != curr_node_idx and prev_idx not in visit:
                            can_extend = False
                            break
                    if not can_extend:
                        break
                    curr_node_idx = node_idx[next_node]
                    curr_node = next_node
                    run_thread.append(curr_node_idx)

            return runnable

        while len(visit) < len(self.nodes):
            runnable = find_runnable()
            runnable = sorted(runnable, key=lambda x: len(x))

            taken = runnable[0]

            if mode == "save_mem":
                run_mems = []
                for run in runnable:
                    act = 0
                    temp_peak = 0
                    for idx in run:
                        node = self.nodes[idx]
                        temp_peak = max(temp_peak, node.temp_mem + act)
                        act += node.act_mem
                    run_mems.append((run, act, temp_peak))
                total_act = sum([m[1] for m in run_mems])
                sort_run = sorted(run_mems, key=lambda x: total_act - x[1] + x[2])
                taken = sort_run[0][0]

            elif mode == "short_first":
                pass
            else:
                taken = runnable[-1]

            for idx in taken:
                visit.add(idx)
                run_order.append(idx)

        new_nodes = []
        for idx in run_order:
            new_nodes.append(self.nodes[idx])
        self.nodes = new_nodes

    def redraw_graph(self, gm):
        fx_nodes = []
        fx_nodes.extend(self.placeholders)
        fx_nodes.extend(self.get_attrs)

        for node in self.nodes:
            fx_nodes.extend(node.nodes)
        fx_nodes.append(self.output)

        curr = gm.graph._root
        for n in fx_nodes:
            curr._next = n
            curr = n
        for n in reversed(fx_nodes[:-1]):
            curr._prev = n
            curr = n
        curr._prev = gm.graph._root
        gm.recompile()
        gm.graph.lint()

    def merge_nodes(self, prev: CompNode, next: CompNode):
        prev.nodes.extend(next.nodes)

        for k, v in self.node_dict.items():
            if v == next:
                self.node_dict[k] = prev

        self.nodes.remove(next)

    def build_edges(self):
        moved_targets = self.param_set.moved_targets
        edge_set: Set[Tuple[CompNode, CompNode, int]] = set()

        out_idx = {n: i for i, n in enumerate(self.output.args[0])}
        self.out_nodes = [None for _ in range(len(self.output.args[0]))]
        node_dict: Dict[Node, CompNode] = dict()

        for cnode in self.nodes:
            for node in cnode.nodes:
                node_dict[node] = cnode

        for cnode in self.nodes:
            for node_idx, node in enumerate(cnode.nodes):
                if node in out_idx:
                    cnode.outputs.add(node)
                    self.out_nodes[out_idx[node]] = (cnode, node_idx)

                if node.op == "call_module":
                    target = node.target
                    if target in moved_targets:
                        target = moved_targets[target]
                    cnode.call_modules.add(target)

                # Draw edge
                def _draw_edge(arg):
                    if not isinstance(arg, Node):
                        return

                    if arg in node_dict.keys():
                        source_node = node_dict[arg]
                        if source_node == cnode:  # self-reference
                            return

                        start_idx = source_node.nodes.index(arg)
                        edge = CompEdge(source_node, cnode, start_idx)
                        edge.weight = arg.meta["_weight"]
                        edge.shape = arg.meta["_shape"]

                        edge_t = (source_node, cnode, start_idx)
                        if edge_t not in edge_set:
                            edge_set.add(edge_t)
                            self.edges.append(edge)
                            source_node.add_out_edge(edge)
                            cnode.add_in_edge(edge)
                    elif arg.op == "placeholder":
                        cnode.placeholders.add(arg)
                    elif arg.op == "get_attr":
                        cnode.get_attrs.add(arg)

                tree_trav(_draw_edge, node.args)
                tree_trav(_draw_edge, node.kwargs)

        for cnode in self.nodes:
            cnode.use_params = set()
            for mod in cnode.call_modules:
                cnode.use_params.update(self.param_set.param_set[mod])
            for attr in cnode.get_attrs:
                target = attr.target
                if target in moved_targets:
                    target = moved_targets[target]
                cnode.use_params.update(self.param_set.param_set[target])

            cnode.weight = 0
            for idx in cnode.use_params:
                # TODO: non-adam optimizer
                # fp32: weight(4) + grad(4) + optimizer states of adam (2 * 4) = 16
                # fp16: fp16 weight(2) + fp16 grad(2) + optimizer states of adam (2 * 4)
                #       + fp32 master weight(4) = 16
                cnode.weight += self.param_set.main_params[idx][0].bytes * 4

    def gen_dummy_cgraph(self):
        """
        Alter every fx.Node to id(fx.Node).
        """
        cgraph = CompGraph()
        cgraph.nodes = [CompNode(None, c.device) for c in self.nodes]
        cgraph.edges = [CompEdge(None, None, e.start_idx) for e in self.edges]

        node_map = {n: c for n, c in zip(self.nodes, cgraph.nodes)}
        edge_map = {e: c for e, c in zip(self.edges, cgraph.edges)}

        def copy_meta(meta):
            return dict(_weight=meta["_weight"], _shape=meta["_shape"])

        fx_map = dict()
        for self_node in self.nodes:
            for n in self_node.nodes:
                fx_map[id(n)] = DummyNode(copy_meta(n.meta))
        for n in self.placeholders:
            fx_map[id(n)] = DummyNode(copy_meta(n.meta))
        for n in self.get_attrs:
            fx_map[id(n)] = DummyNode(copy_meta(n.meta))
        if self.output:
            fx_map[id(self.output)] = DummyNode(copy_meta(self.output.meta))

        for self_node, dummy_cnode in node_map.items():
            dummy_cnode.use_params = self_node.use_params.copy()
            dummy_cnode.weight = self_node.weight
            dummy_cnode.temp_mem = self_node.temp_mem
            dummy_cnode.act_mem = self_node.act_mem
            dummy_cnode.comp_time = self_node.comp_time
            dummy_cnode.optim_time = self_node.optim_time

            dummy_cnode.in_edges = [edge_map[e] for e in self_node.in_edges]
            dummy_cnode.out_edges = [edge_map[e] for e in self_node.out_edges]

            dummy_cnode.placeholders = {fx_map[id(n)] for n in self_node.placeholders}
            dummy_cnode.call_modules = self_node.call_modules.copy()
            dummy_cnode.get_attrs = {fx_map[id(n)] for n in self_node.get_attrs}
            dummy_cnode.outputs = {fx_map[id(n)] for n in self_node.outputs}

            dummy_cnode.nodes = [fx_map[id(n)] for n in self_node.nodes]

        for self_edge, dummy_edge in edge_map.items():
            # save only index
            dummy_edge.start_node = (
                self.nodes.index(self_edge.start_node) if self_edge.start_node else None
            )
            dummy_edge.end_node = (
                self.nodes.index(self_edge.end_node) if self_edge.end_node else None
            )

            dummy_edge.weight = self_edge.weight
            dummy_edge.shape = self_edge.shape

        cgraph.in_nodes = [node_map[n] for n in self.in_nodes]
        cgraph.out_nodes = [(node_map[n], idx) for n, idx in self.out_nodes]

        cgraph.placeholders = [fx_map[id(n)] for n in self.placeholders]
        cgraph.get_attrs = [fx_map[id(n)] for n in self.get_attrs]

        cgraph.output = fx_map[id(self.output)] if self.output else None

        cgraph.param_set = self.param_set
        cgraph.fb_ratio = self.fb_ratio

        return cgraph

    @staticmethod
    def read_comp_graph(graph_path):
        path = os.path.abspath(graph_path)
        with open(path, "rb") as f:
            cgraph = pickle.load(f)

        for edge in cgraph.edges:
            edge.start_node = (
                cgraph.nodes[edge.start_node] if edge.start_node is not None else None
            )
            edge.end_node = (
                cgraph.nodes[edge.end_node] if edge.end_node is not None else None
            )

        return cgraph

    def save_graph_data(self, graph_path):
        path = os.path.abspath(graph_path)
        cgraph = self.gen_dummy_cgraph()

        with open(path, "wb") as f:
            pickle.dump(cgraph, f)

    def load_graph_data(self, graph_path):
        graph_data = self.read_comp_graph(graph_path)

        for cnode, node_data in zip(self.nodes, graph_data.nodes):
            cnode.weight = node_data.weight
            cnode.temp_mem = node_data.temp_mem
            cnode.act_mem = node_data.act_mem
            cnode.comp_time = node_data.comp_time
            cnode.optim_time = node_data.optim_time

        for edge, edge_data in zip(self.edges, graph_data.edges):
            edge.weight = edge_data.weight
            edge.shape = edge_data.shape

        self.fb_ratio = graph_data.fb_ratio


class GraphCPUFetcher(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, graph: CompGraph):
        super().__init__(gm)
        self.comp_graph = graph

        self.prev_node: Optional[CompNode] = None
        self.curr_node = CompNode(None, "cpu")
        self.comp_graph.nodes.append(self.curr_node)

        self.in_get_item = False

        # Parameter -> (param idx of Module (if -1 -> attr), Node)
        self.param_rev: Dict[torch.nn.Parameter, List[Tuple[int, Node]]] = defaultdict(
            list
        )

    def move_node(self):
        self.prev_node = self.curr_node
        self.curr_node = CompNode(None, "cpu")
        self.comp_graph.nodes.append(self.curr_node)

    def check_dup_and_clone(self, target, submod):
        found = dict()
        for i, p in enumerate(submod.parameters()):
            if p in self.param_rev.keys():
                found[i] = p

        if len(found) > 0:
            # print(f"found shared weight from {target} to ", end="")
            for i, p in found.items():
                for idx, old_target in self.param_rev[p]:
                    print(f"{old_target}({idx}), ", end="")
            print()
            submod = deepcopy(submod)
            self.set_attr(target, submod)

        for i, p in enumerate(submod.parameters()):
            if i in found:
                self.param_rev[found[i]].append((i, target))
            else:
                self.param_rev[p].append((i, target))

        return submod

    def set_attr(self, target: str, attr):
        target_atoms = target.split(".")
        attr_itr = self.module
        for atom in target_atoms[:-1]:
            attr_itr = getattr(attr_itr, atom)
        setattr(attr_itr, target_atoms[-1], attr)

    def merge_prev_curr(self):
        self.comp_graph.merge_nodes(self.prev_node, self.curr_node)
        self.curr_node = CompNode(None, "cpu")
        self.comp_graph.nodes.append(self.curr_node)

    def run(self, *args, initial_env=None, enable_io_processing: bool = True):
        self.env = initial_env if initial_env is not None else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)
        self.args_iter = iter(args)

        def fetch_shapes(result) -> Optional[ShapeMeta]:
            if isinstance(result, torch.Tensor):
                return ShapeMeta(
                    result.shape,
                    result.dtype,
                    result.numel() * result.element_size(),
                    result.requires_grad,
                )

        for node in self.module.graph.nodes:
            if node in self.env:
                continue

            self.curr_node.nodes.append(node)
            self.comp_graph.node_dict[node] = self.curr_node

            # merge getitems to the prev node
            if node.op == "call_function" and node.target == operator.getitem:
                self.in_get_item = True
            else:
                if self.in_get_item:
                    self.merge_prev_curr()
                self.in_get_item = False

            try:
                with self._set_current_node(node):
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    assert isinstance(args, tuple)
                    assert isinstance(kwargs, dict)
                    result = getattr(self, node.op)(node.target, args, kwargs)

                    shapes = tree_filter(fetch_shapes, result)
                    node.meta["_weight"] = sum([s.bytes for s in shapes])
                    node.meta["_shape"] = shapes

                    self.env[node] = result
            except Exception as e:
                msg = f"While executing {node.format_node()}"
                msg = "{}\n\n{}".format(e.args[0], msg) if e.args else str(msg)
                msg += f"\nOriginal traceback:\n{node.stack_trace}"
                e.args = (msg,) + e.args[1:]
                if isinstance(e, KeyError):
                    raise RuntimeError(*e.args) from e
                raise

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == "output":
                output_val = self.env[node]
                return (
                    self.module.graph.process_outputs(output_val)
                    if enable_io_processing
                    else output_val
                )

            if self.curr_node.weight > 0:
                self.move_node()

    def call_module(self, target, args, kwargs):
        submod = self.fetch_attr(target)
        submod = self.check_dup_and_clone(target, submod)

        self.curr_node.submods.append(submod)

        params = 0
        for param in submod.parameters():
            params += param.untyped_storage().nbytes()
            # pre-popultae gradient
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

        for param in submod.buffers():
            params += param.untyped_storage().nbytes()

        # TODO: handle optimizer state
        params *= 4  # weight + grad + optimizer states of adam (2)

        ret = super().call_module(target, args, kwargs)

        self.curr_node.weight += params
        self.curr_node.act_mem += params  # assumed for test

        return ret


class GraphFetcher(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, graph: CompGraph, devices: List[str]):
        super().__init__(gm)
        self.comp_graph = graph

        self.device_id = 0
        self.devices = devices
        self.curr_device = devices[0]
        self.mem_cap = torch.cuda.get_device_properties(self.curr_device).total_memory

        self.prev_node: Optional[CompNode] = None
        self.curr_node = CompNode(None, self.curr_device, self.device_id)
        self.comp_graph.nodes.append(self.curr_node)

        # param_idx -> (param, main_target)
        self.param_list: List[Tuple[torch.nn.Parameter, str]] = []

        # parameter -> (param_idx, main target)
        self.param_idx: Dict[torch.nn.Parameter, Tuple[int, str]] = dict()

    def move_node(self):
        self.prev_node = self.curr_node
        self.curr_node = CompNode(None, self.curr_device, self.device_id)
        self.comp_graph.nodes.append(self.curr_node)

    def reset_inlined_devices(self, node):
        # check and change inlined device
        args = node.args
        kwargs = node.kwargs
        found_arg = False
        found_kwarg = False

        for a in args:
            if isinstance(a, torch.device):
                found_arg = True
                break

        for v in kwargs.values():
            if isinstance(v, torch.device):
                found_kwarg = True
                break

        if found_arg:
            arg_list = []
            for a in args:
                if isinstance(a, torch.device):
                    arg_list.append(torch.device(self.curr_device))
                else:
                    arg_list.append(a)
            args = tuple(arg_list)
            node.args = args

        if found_kwarg:
            kwargs = dict(**kwargs)
            for k, v in kwargs.items():
                if isinstance(v, torch.device):
                    kwargs[k] = torch.device(self.curr_device)
            node.kwargs = kwargs

        return args, kwargs

    def check_dup_and_clone(self, target, submod):
        param_set = self.comp_graph.param_set
        params = list_params(submod)

        should_move = False
        device_set = set()

        # if target is reused
        if target in param_set.param_set:
            for p in params:
                device_set.add(p.device)

            # should move the device
            if not all([d == self.curr_device for d in device_set]):
                should_move = True
        else:
            # first time -> record param set
            for p in params:
                # found shared weight
                if p in self.param_idx:
                    idx, _ = self.param_idx[p]
                    # if p.device != torch.device("cpu") and p.device != self.curr_device:
                    should_move = True
                else:
                    idx = len(self.param_list)
                    self.param_idx[p] = (idx, target)
                    self.param_list.append((p, target))
                    param_set.main_params.append(
                        (
                            ShapeMeta(
                                p.size(),
                                p.dtype,
                                p.numel() * p.element_size(),
                                p.requires_grad,
                            ),
                            target,
                        )
                    )
                param_set.param_set[target].append(idx)

        if should_move:
            clone_idx = len(param_set.moved_targets)
            moved = f"{target}_clone{clone_idx}"
            submod = deepcopy(submod)
            param_set.moved_targets[moved] = target

            module_set_attr(self.module, moved, submod)
            self.curr_fx_node.target = moved

            # print(f"target moved: {target} -> {moved}")

            target = moved

        return target, submod

    def merge_prev_curr(self):
        if self.prev_node is None:
            return

        self.comp_graph.merge_nodes(self.prev_node, self.curr_node)
        self.curr_node = CompNode(None, self.curr_device, self.device_id)
        self.comp_graph.nodes.append(self.curr_node)

    def move_device(self):
        self.device_id += 1
        self.curr_device = self.devices[self.device_id]
        self.mem_cap = torch.cuda.get_device_properties(self.curr_device).total_memory

    def to_device(self, args, kwargs):
        return to_device(args, self.curr_device), to_device(kwargs, self.curr_device)

    def append_node(self, node):
        graph = self.comp_graph
        if node.op == "placeholder":
            graph.placeholders.append(node)
        elif node.op == "get_attr":
            graph.get_attrs.append(node)
        elif node.op == "call_function":
            cnode = self.curr_node
            self.curr_node.nodes.append(node)
        elif node.op == "call_method":
            self.curr_node.nodes.append(node)
        elif node.op == "call_module":
            self.curr_node.nodes.append(node)
        elif node.op == "output":
            graph.output = node

    def remove_last_unused(self):
        graph = self.comp_graph
        if len(graph.nodes) > 0 and len(graph.nodes[-1].nodes) == 0:
            graph.nodes.pop()

    def run(self, *args, initial_env=None, enable_io_processing: bool = True):
        self.env = initial_env if initial_env is not None else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)
        self.args_iter = iter(args)

        def fetch_shapes(result) -> Optional[ShapeMeta]:
            if isinstance(result, torch.Tensor):
                return ShapeMeta(
                    result.shape,
                    result.dtype,
                    result.numel() * result.element_size(),
                    result.requires_grad,
                )

        for node in self.module.graph.nodes:
            # print(node.op, node.name, node.target, node.args, node.kwargs)

            if node in self.env:
                continue

            self.append_node(node)

            # Synchornize CUDA stream & check memory usage
            torch.cuda.synchronize(self.curr_device)
            mem_before = torch.cuda.memory_allocated(self.curr_device)

            try:
                with self._set_current_node(node):
                    self.reset_inlined_devices(node)
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    args, kwargs = self.to_device(args, kwargs)
                    assert isinstance(args, tuple)
                    assert isinstance(kwargs, dict)

                    self.curr_fx_node = node
                    result = getattr(self, node.op)(node.target, args, kwargs)

                    shapes = tree_filter(fetch_shapes, result)
                    node.meta["_weight"] = sum([s.bytes for s in shapes])
                    node.meta["_shape"] = shapes

                    self.env[node] = result
                    del result
            except Exception as e:
                msg = f"While executing {node.format_node()}"
                msg = "{}\n\n{}".format(e.args[0], msg) if e.args else str(msg)
                msg += f"\nOriginal traceback:\n{node.stack_trace}"
                e.args = (msg,) + e.args[1:]
                if isinstance(e, KeyError):
                    raise RuntimeError(*e.args) from e
                raise

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == "output":
                output_val = self.env[node]
                self.remove_last_unused()
                return (
                    self.module.graph.process_outputs(output_val)
                    if enable_io_processing
                    else output_val
                )

            torch.cuda.synchronize(self.curr_device)
            mem_after = torch.cuda.memory_allocated(self.curr_device)
            mem_resv = torch.cuda.memory_reserved(self.curr_device)

            torch.cuda.reset_max_memory_allocated()

            # TODO: not a 60% cap. sense the weight!
            if mem_resv > GRAPH_FETCH_MEM_CAP * self.mem_cap:
                self.move_device()
                print(
                    f"[fetch graph]: move device to {self.curr_device} at mem {fmem(mem_resv)}"
                )
                print(f"moved node: {node.op} {node.target} {node.name}")

            mem_diff = mem_after - mem_before

            if mem_diff > 0 and len(self.curr_node.nodes) > 0:
                self.move_node()

        self.remove_last_unused()

    def call_module(self, target, args, kwargs):
        submod = self.fetch_attr(target)
        target, submod = self.check_dup_and_clone(target, submod)

        submod = submod.to(device=self.curr_device)
        self.recent_submod = submod

        params = 0
        for param in submod.parameters():
            params += param.untyped_storage().nbytes()
            # pre-populate gradient
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

        for param in submod.buffers():
            params += param.untyped_storage().nbytes()

        ret = super().call_module(target, args, kwargs)

        return ret

    def get_attr(self, target, args, kwargs):
        assert isinstance(target, str)
        attr = self.fetch_attr(target)
        target, attr = self.check_dup_and_clone(target, attr)
        self.recent_submod = attr

        params = list_params(attr)

        if len(params) > 0:
            for p in params:
                # pre-populate gradient
                p.data = p.to(device=self.curr_device)
                p.grad = torch.zeros_like(p)
        else:
            attr = to_device(attr, self.curr_device)
            module_set_attr(self.module, target, attr)
            self.recent_attr = (attr, target)

        return attr


class GraphSingleDeviceFetcher(GraphFetcher):
    def __init__(self, gm: fx.GraphModule, graph: CompGraph, device):
        super().__init__(gm, graph, [device])
        self.recent_attr = None
        self.submod_cache = set()
        self.param_cache = set()
        self.attr_cache = set()

    def move_device(self):
        self.device_id += 1
        self.clear_device()

    def clear_device(self):
        # clear all memories
        for submod in self.submod_cache:
            submod.to(device="cpu")

        for p in self.param_cache:
            p.data = p.data.to(device="cpu")
            if p.grad is not None:
                p.grad = p.grad.to(device="cpu")

        for attr, target in self.attr_cache:
            module_set_attr(self.module, target, to_device(attr, "cpu"))

        self.submod_cache.clear()
        self.param_cache.clear()
        self.attr_cache.clear()

        def detach(x):
            if isinstance(x, torch.Tensor):
                return x.detach().requires_grad_(x.requires_grad)
            return x

        for k, v in self.env.items():
            self.env[k] = tree_map(detach, v)

        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.curr_device)

    def run(self, *args, initial_env=None, enable_io_processing: bool = True):
        result = super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

        self.clear_device()
        return result

    def call_module(self, target, args, kwargs):
        result = super().call_module(target, args, kwargs)
        self.submod_cache.add(self.recent_submod)
        self.param_cache.update(list_params(self.recent_submod))
        return result

    def get_attr(self, target, args, kwargs):
        result = super().get_attr(target, args, kwargs)
        params = list_params(self.recent_submod)
        if len(params) > 0:
            self.param_cache.update(params)
        else:
            self.attr_cache.add(self.recent_attr)
        return result


class MemoryLogger(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, graph: CompGraph):
        super().__init__(gm)
        self.graph = graph
        self.fwd_dict = {}
        self.placeholder_order = []

    def run_outer_node(self, nodes, device):
        for node in nodes:
            if node not in self.exec_input_attr:
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                args = to_device(args, device)
                kwargs = to_device(kwargs, device)
                self.env[node] = getattr(self, node.op)(node.target, args, kwargs)
                self.exec_input_attr.add(node)

    def run_output(self, node):
        if node not in self.exec_input_attr:
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            # args = to_device(args, device) # not move device
            # kwargs = to_device(kwargs, device)
            self.env[node] = getattr(self, node.op)(node.target, args, kwargs)
            self.exec_input_attr.add(node)

    def check_mem(self, device):
        torch.cuda.synchronize(device)
        mem_before = torch.cuda.memory_allocated(device)
        max_before = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_max_memory_allocated(device)
        return mem_before, max_before

    def run(self, *args, initial_env=None, enable_io_processing: bool = True):
        self.exec_input_attr = set()
        self.env = initial_env if initial_env is not None else {}
        self.placeholder_order = []

        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)
        self.args_iter = iter(args)

        # run placeholders first
        placeholders = []
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                with self._set_current_node(node):
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    self.env[node] = getattr(self, node.op)(node.target, args, kwargs)
                    self.exec_input_attr.add(node)
                    placeholders.append(node)

        devices = set()
        for cnode in self.graph.nodes:
            devices.add(cnode.device)

        for device in devices:
            torch.cuda.reset_max_memory_allocated(device)

        output = None
        last_device = self.graph.nodes[0].device
        ordered = set()

        for cnode_idx, cnode in enumerate(self.graph.nodes):
            curr_device = cnode.device

            # run placeholder / get_attr first
            # self.run_outer_node(cnode.placeholders, device)
            for node in cnode.placeholders:
                if node not in ordered:
                    self.placeholder_order.append(placeholders.index(node))
                    ordered.add(node)
            self.run_outer_node(cnode.get_attrs, device)

            # self.bench_cnode(cnode)

            device_set = (
                (curr_device, last_device)
                if curr_device != last_device
                else (curr_device,)
            )
            for device in device_set:
                mem_before, max_before = self.check_mem(device)
                self.fwd_dict[cnode_idx, device] = (mem_before, max_before)

            for node in cnode.nodes:
                with self._set_current_node(node):
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    args = to_device(args, curr_device)
                    kwargs = to_device(kwargs, curr_device)

                    assert isinstance(args, tuple)
                    assert isinstance(kwargs, dict)
                    self.env[node] = getattr(self, node.op)(node.target, args, kwargs)

                if self.garbage_collect_values:
                    for to_delete in self.user_to_last_uses.get(node, []):
                        del self.env[to_delete]

            last_device = curr_device

        node = self.graph.output
        self.run_output(node)
        output_val = self.env[node]

        output = (
            self.module.graph.process_outputs(output_val)
            if enable_io_processing
            else output_val
        )

        return output


class MemoryLocalDeviceLogger(MemoryLogger):
    def run(self, *args, initial_env=None, enable_io_processing: bool = True):
        self.exec_input_attr = set()
        self.env = initial_env if initial_env is not None else {}
        self.placeholder_order = []

        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)
        self.args_iter = iter(args)

        # run placeholders first
        placeholders = []
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                with self._set_current_node(node):
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    self.env[node] = getattr(self, node.op)(node.target, args, kwargs)
                    self.exec_input_attr.add(node)
                    placeholders.append(node)

        devices = set()
        for cnode in self.graph.nodes:
            devices.add(cnode.device)

        for device in devices:
            torch.cuda.reset_max_memory_allocated(device)

        output = None
        main_device = self.graph.nodes[0].device
        last_device_id = -1
        ordered = set()

        for _, cnode in enumerate(self.graph.nodes):
            curr_device_id = cnode.device_id

            if curr_device_id != last_device_id:
                self.move_device(curr_device_id, last_device_id)

            # run placeholder / get_attr first
            # self.run_outer_node(cnode.placeholders, device)
            for node in cnode.placeholders:
                if node not in ordered:
                    self.placeholder_order.append(placeholders.index(node))
                    ordered.add(node)

            mem_before, _ = self.check_mem(main_device)
            self.run_outer_node(cnode.get_attrs, device)

            for node in cnode.nodes:
                with self._set_current_node(node):
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    args = to_device(args, main_device)
                    kwargs = to_device(kwargs, main_device)

                    self.env[node] = getattr(self, node.op)(node.target, args, kwargs)

                if self.garbage_collect_values:
                    for to_delete in self.user_to_last_uses.get(node, []):
                        del self.env[to_delete]

            del args, kwargs

            mem_after, max_after = self.check_mem(main_device)

            cnode.act_mem = mem_after - mem_before
            cnode.temp_mem = max_after - mem_before

            last_device_id = curr_device_id

        node = self.graph.output
        self.run_output(node)
        output_val = self.env[node]

        output = (
            self.module.graph.process_outputs(output_val)
            if enable_io_processing
            else output_val
        )

        return output

    def move_device_node(self, node, device):
        if node.op == "call_module":
            submod = self.fetch_attr(node.target)
            submod.to(device=device)
        elif node.op == "get_attr":
            attr = self.fetch_attr(node.target)
            params = list_params(attr)
            for p in params:
                p.data = p.data.to(device=device)
                if p.grad is not None:
                    p.grad = p.grad.to(device=device)

            if len(params) == 0:
                attr = to_device(attr, device)
                module_set_attr(self.module, node.target, attr)

    def move_device(self, device_id, last_id):
        curr_nodes = []
        last_nodes = []
        for cnode in self.graph.nodes:
            if cnode.device_id == device_id:
                curr_nodes.append(cnode)
            elif cnode.device_id == last_id:
                last_nodes.append(cnode)

        for cnode in last_nodes:
            for node in cnode.nodes:
                self.move_device_node(node, "cpu")

        def detach(x):
            if isinstance(x, torch.Tensor):
                return x.detach().requires_grad_(x.requires_grad)
            return x

        for k, v in self.env.items():
            self.env[k] = tree_map(detach, v)

        for cnode in curr_nodes:
            for node in cnode.nodes:
                self.move_device_node(node, cnode.device)

        torch.cuda.empty_cache()
        torch.cuda.synchronize(curr_nodes[0].device)


class TimingChecker(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, graph: CompGraph, args_order):
        super().__init__(gm)
        self.graph = graph

        self.events = []
        self.args_order = args_order

    def run_outer_node(self, nodes, device):
        for node in nodes:
            if node not in self.exec_input_attr:
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                args = to_device(args, device)
                kwargs = to_device(kwargs, device)
                self.env[node] = getattr(self, node.op)(node.target, args, kwargs)
                self.exec_input_attr.add(node)
            else:
                # to device
                self.env[node] = to_device(self.env[node], device)

    def run_output(self, node):
        if node not in self.exec_input_attr:
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            self.env[node] = getattr(self, node.op)(node.target, args, kwargs)
            self.exec_input_attr.add(node)

    def run_timing(
        self, *args, initial_env=None, enable_io_processing: bool = True, no_event=False
    ):
        self.env = initial_env if initial_env is not None else {}
        self.exec_input_attr = set()

        # rearrange args by args_order
        args = [args[i] for i in self.args_order]

        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)
        self.args_iter = iter(args)

        start_event = []
        end_event = []

        # prefetch streams
        for cnode in self.graph.nodes:
            stream = torch.cuda.current_stream(cnode.device)
            s_event = torch.cuda.Event(enable_timing=True) if not no_event else None
            e_event = torch.cuda.Event(enable_timing=True) if not no_event else None
            moved = False

            for edge in cnode.in_edges:
                if edge.start_node.device != cnode.device:
                    moved = True
                    break

            start_event.append((s_event, stream, moved))
            end_event.append(e_event)

        self.start_time = time.perf_counter()

        for cnode_idx, cnode in enumerate(self.graph.nodes):
            self.run_outer_node(cnode.placeholders, cnode.device)
            self.run_outer_node(cnode.get_attrs, cnode.device)

            s_event, stream, moved = start_event[cnode_idx]
            recorded = False
            if not no_event and not moved:
                s_event.record(stream)

            for node in cnode.nodes:
                with self._set_current_node(node):
                    args, kwargs = self.fetch_args_kwargs_from_env(node)

                    # TODO: to_device/edge based event recording
                    # This line is too heavy
                    if moved:
                        args = to_device(args, cnode.device)
                        kwargs = to_device(kwargs, cnode.device)
                        if not no_event and not recorded:
                            s_event.record(stream)

                    self.env[node] = getattr(self, node.op)(node.target, args, kwargs)

                    # if moved -> record only single event
                    if not no_event and moved and not recorded:
                        recorded = True
                        e_event = end_event[cnode_idx]
                        e_event.record(stream)

                if self.garbage_collect_values:
                    for to_delete in self.user_to_last_uses.get(node, []):
                        del self.env[to_delete]

            if not no_event and not moved:
                e_event = end_event[cnode_idx]
                e_event.record(stream)

        node = self.graph.output
        self.run_output(node)
        output_val = self.env[node]

        output = (
            self.module.graph.process_outputs(output_val)
            if enable_io_processing
            else output_val
        )

        self.events.append((start_event, end_event))

        return output


class TimingLocalDeviceChecker(TimingChecker):
    def __init__(
        self, gm: fx.GraphModule, graph: CompGraph, cnodes: List[CompNode], args_order
    ):
        super().__init__(gm, graph, args_order)
        self.cnodes = cnodes

        self.events = []
        self.args_order = args_order

    def move_device_node(self, node, device):
        if node.op == "call_module":
            submod = self.fetch_attr(node.target)
            submod.to(device=device)
        elif node.op == "get_attr":
            attr = self.fetch_attr(node.target)
            params = list_params(attr)
            for p in params:
                p.data = p.data.to(device=device)
                if p.grad is not None:
                    p.grad = p.grad.to(device=device)

            if len(params) == 0:
                attr = to_device(attr, device)
                module_set_attr(self.module, node.target, attr)

    def move_cnode_devices(self):
        for cnode in self.graph.nodes:
            for node in cnode.nodes:
                self.move_device_node(node, "cpu")

        for cnode in self.cnodes:
            for node in cnode.nodes:
                self.move_device_node(node, cnode.device)

    def run_timing(
        self,
        *args,
        exec_input_attr=None,
        initial_env=None,
        input_processing: bool = True,
        output_processing=True,
        no_event=False,
    ):
        self.env = initial_env if initial_env is not None else {}
        self.exec_input_attr = exec_input_attr if exec_input_attr is not None else set()

        # rearrange args by args_order
        if input_processing:
            args = [args[i] for i in self.args_order]
            args = self.module.graph.process_inputs(*args)
            self.args_iter = iter(args)
        else:
            self.args_iter = iter([])

        start_event = []
        end_event = []

        # prefetch streams
        for cnode in self.cnodes:
            stream = torch.cuda.current_stream(cnode.device)
            s_event = torch.cuda.Event(enable_timing=True) if not no_event else None
            e_event = torch.cuda.Event(enable_timing=True) if not no_event else None

            start_event.append((s_event, stream))
            end_event.append(e_event)

        self.start_time = time.perf_counter()

        for cnode_idx, cnode in enumerate(self.cnodes):
            self.run_outer_node(cnode.placeholders, cnode.device)
            self.run_outer_node(cnode.get_attrs, cnode.device)

            s_event, stream = start_event[cnode_idx]
            recorded = False
            if not no_event:
                s_event.record(stream)

            for node in cnode.nodes:
                with self._set_current_node(node):
                    args, kwargs = self.fetch_args_kwargs_from_env(node)

                    # TODO: to_device/edge based event recording
                    # This line is too heavy
                    # if moved:
                    args = to_device(args, cnode.device)
                    kwargs = to_device(kwargs, cnode.device)
                    if not no_event and not recorded:
                        s_event.record(stream)

                    self.env[node] = getattr(self, node.op)(node.target, args, kwargs)

                if self.garbage_collect_values:
                    for to_delete in self.user_to_last_uses.get(node, []):
                        del self.env[to_delete]

            if not no_event:
                e_event = end_event[cnode_idx]
                e_event.record(stream)

        if output_processing:
            node = self.graph.output
            self.run_output(node)
            output_val = self.env[node]

            output = self.module.graph.process_outputs(output_val)
        else:
            output = self.env

        self.events.append((start_event, end_event))

        return output
