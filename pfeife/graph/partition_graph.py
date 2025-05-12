import inspect
import pickle
from copy import deepcopy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
import torch.fx as fx
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule

from ..utils import to_device_graphmodule
from .pipe_graph import PipeGraph, PipeNode
from .computation_graph import ShapeMeta


class Partition:
    def __init__(self, idx: int):
        self.idx: int = idx
        self.node_names: List[str] = []
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.attr_usage: Set[Node] = set()
        self.partitions_dependent_on: Set[int] = set()
        self.partition_dependents: Set[int] = set()
        self.graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
        self.graph_module: Optional[GraphModule] = None

        # input str -> partition_id (-1 if external input)
        self.input_map: Dict[str, int] = {}
        # output str -> partition_id (-1 if external output)
        self.output_map: Dict[str, Set[int]] = defaultdict(set)

        # inputs_idx -> list of tensors found by pytree traverse
        self.input_shapes: List[List[ShapeMeta]] = []
        self.output_shapes: List[List[ShapeMeta]] = []

        # original node to new node
        self.environment: Dict[torch.fx.node.Node, torch.fx.node.Node] = {}
        self.targets: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return (
            f"name: {self.idx},\n"
            f" nodes: {self.node_names},\n"
            f" inputs: {self.inputs},\n"
            f" outputs: {self.outputs},\n"
            f" partitions dependent on: {self.partitions_dependent_on},\n"
            f" partition dependents: {self.partition_dependents}"
        )


class PartitionNode:
    def __init__(self, partition: Partition):
        self.idx: int = partition.idx
        self.rank: int = -1
        self.inputs: List[str] = partition.inputs
        self.outputs: List[str] = partition.outputs
        self.input_map: Dict[str, int] = partition.input_map
        self.output_map: Dict[str, Set[int]] = partition.output_map
        self.input_shapes: List[List[ShapeMeta]] = partition.input_shapes
        self.output_shapes: List[List[ShapeMeta]] = partition.output_shapes
        self.graph_module: Optional[GraphModule] = partition.graph_module


class PartitionGraph:
    def __init__(self):
        self.nodes: Dict[int, PartitionNode] = {}
        # usage of each placeholder -> (partition_id, self.inputs index)
        self.inputs: List[List[Tuple[int, int]]] = []
        # outputs -> (partition_id, self.outputs index)
        self.outputs: List[Tuple[int, int]] = []

    def split_rank_gm(self, optimizer=None) -> Dict[int, "PartitionGraph"]:
        """
        Returns new PartitionGraphs with only the graph modules that are on the given rank
        """

        gms = dict()
        graph_dict = dict()  # unique for each rank

        # unload all graph modules. will be reloaded to GPU at set_partition_graph
        for idx, part_node in self.nodes.items():
            # part_node.graph_module.graph.print_tabular()
            gm_cpu = to_device_graphmodule(part_node.graph_module, "cpu")

            gms[idx] = gm_cpu
            part_node.graph_module = None

        # copy knocked-off partiton graph
        for idx, part_node in self.nodes.items():
            if part_node.rank not in graph_dict:
                graph_dict[part_node.rank] = deepcopy(self)

        # set gms and revert graph modules
        for idx, part_node in self.nodes.items():
            graph_dict[part_node.rank].nodes[idx].graph_module = gms[idx]
            part_node.graph_module = gms[idx]

        if optimizer is None:
            return graph_dict, None

        # if optimizer is not None, remove all parameters of graph modules from the optimizer
        # for each graph module, record its parameters' position in the optimizer
        optim_dict = dict()
        param_node_dict = dict()
        param_set = set()
        for idx, gm in gms.items():
            for param_id, param in enumerate(gm.parameters()):
                param_node_dict[param] = (idx, param_id)
                param_set.add(param)

        for param_group_id, param_group in enumerate(optimizer.param_groups):
            new_param_list = []
            for param in param_group["params"]:
                if param in param_set:
                    optim_dict[param_node_dict[param]] = param_group_id
                else:
                    new_param_list.append(param)
            param_group["params"] = new_param_list

        return graph_dict, optim_dict


def partition_module(m: GraphModule, split_map: Dict[Node, int]) -> PartitionGraph:
    """
    Creates a list of subgraphs out of main graph

    Args:
        m (GraphModule): Graph module to split
        split_map (Dict[Node, int]): Map from node to partition id (must be non-negative)

    Returns:
        PartitionGraph: Map from partition id to Partition
    """
    part_graph = PartitionGraph()

    partitions: Dict[int, Partition] = {}
    global_placeholder = part_graph.inputs
    global_outputs = part_graph.outputs

    output_node_map: List[Tuple[Node, int]] = []

    orig_nodes: Dict[str, Node] = {}
    base_attrs: Dict[str, Node] = {}
    placeholders: Dict[str, Tuple[List[Tuple[int, int]], Node]] = {}

    def record_cross_partition_use(
        def_node: torch.fx.node.Node,
        use_node: Optional[torch.fx.node.Node],
        is_output=False,
    ):  # noqa: B950
        def_idx = getattr(def_node, "_fx_partition", None)
        use_idx = getattr(use_node, "_fx_partition", None)
        if def_idx != use_idx:
            if def_idx is not None:
                def_partition = partitions[def_idx]
                if def_node.name not in def_partition.outputs:
                    def_partition.outputs.append(def_node.name)
                if use_idx is not None:
                    def_partition.partition_dependents.add(use_idx)
                if is_output:
                    def_partition.output_map[def_node.name].add(-1)
                    output_node_map.append((def_node, def_idx))

            if use_idx is not None:
                use_partition = partitions[use_idx]
                if def_node.name in base_attrs:
                    use_partition.attr_usage.add(base_attrs[def_node.name])
                    return
                if def_node.name not in use_partition.inputs:
                    use_partition.inputs.append(def_node.name)
                if def_idx is not None:
                    use_partition.partitions_dependent_on.add(def_idx)

    # split nodes into parititons
    for node in m.graph.nodes:
        orig_nodes[node.name] = node

        if node.op == "placeholder":
            # record global placeholder
            usage = []
            global_placeholder.append(usage)
            placeholders[node.name] = (usage, node)
            continue
        elif node.op == "get_attr":
            # move attr to all the user modules
            base_attrs[node.name] = node
            continue

        if node.op == "output":
            torch.fx.graph.map_arg(
                node.args[0], lambda n: record_cross_partition_use(n, None, True)
            )
            continue
        partition_idx = split_map[node]

        # add node to partitions
        partition = partitions.get(partition_idx)
        if partition is None:
            partition = Partition(partition_idx)
            partitions[partition_idx] = partition

        partition.node_names.append(node.name)
        node._fx_partition = partition_idx

        torch.fx.graph.map_arg(
            node.args, lambda def_node: record_cross_partition_use(def_node, node)
        )
        torch.fx.graph.map_arg(
            node.kwargs, lambda def_node: record_cross_partition_use(def_node, node)
        )  # noqa: B950

    for partition_id, partition in partitions.items():
        # add placeholders to parititons
        for idx, input in enumerate(partition.inputs):
            if input in placeholders:
                usage, node = placeholders[input]
                default_value = (
                    node.args[0] if len(node.args) > 0 else inspect.Signature.empty
                )
                placeholder = partition.graph.placeholder(
                    input,
                    type_expr=node.type,
                    default_value=default_value,
                )
                usage.append((partition_id, idx))
                partition.input_map[node.name] = -1
            else:
                placeholder = partition.graph.placeholder(
                    input,
                    type_expr=orig_nodes[input].type,
                )
            placeholder.meta = orig_nodes[input].meta.copy()
            partition.environment[orig_nodes[input]] = placeholder

        # add attributes to parititons
        ## sort attr_usage because of consistency between dp ranks
        attr_usage = list(partition.attr_usage)
        attr_usage.sort(key=lambda x: x.name)
        for attr_node in attr_usage:
            target_atoms = attr_node.target.split(".")
            target_attr = m
            for atom in target_atoms:
                if not hasattr(target_attr, atom):
                    raise RuntimeError(f"Operator target {node.target} not found!")
                target_attr = getattr(target_attr, atom)
            # target = target_atoms[-1]
            target = "_".join(target_atoms)
            partition.targets[target] = target_attr

            # TODO: record buffer?
            get_attr = partition.graph.get_attr(target)
            get_attr.meta = attr_node.meta.copy()
            orig_nodes[get_attr.name] = attr_node
            partition.environment[attr_node] = get_attr

    # Transform nodes and collect targets for partition's submodule
    for node in m.graph.nodes:
        if hasattr(node, "_fx_partition"):
            partition = partitions[node._fx_partition]

            # swap out old graph nodes in kw/args with references to new nodes in this submodule
            environment = partition.environment
            gathered_args = torch.fx.graph.map_arg(node.args, lambda n: environment[n])
            gathered_kwargs = torch.fx.graph.map_arg(
                node.kwargs, lambda n: environment[n]
            )

            if node.op not in ["call_module", "get_attr"]:
                target = node.target
            else:
                target_atoms = node.target.split(".")
                target_attr = m
                for atom in target_atoms:
                    if not hasattr(target_attr, atom):
                        raise RuntimeError(f"Operator target {node.target} not found!")
                    target_attr = getattr(target_attr, atom)
                # target = target_atoms[-1]
                target = "_".join(target_atoms)
                partition.targets[target] = target_attr

            assert isinstance(gathered_args, tuple)
            assert isinstance(gathered_kwargs, dict)
            new_node = partition.graph.create_node(
                op=node.op,
                target=target,
                args=gathered_args,
                kwargs=gathered_kwargs,
                type_expr=node.type,
            )
            new_node.meta = node.meta.copy()
            partition.environment[node] = new_node

    for partition in partitions.values():
        # Set correct output values
        output_vals = tuple(
            partition.environment[orig_nodes[name]] for name in partition.outputs
        )
        output_vals = output_vals[0] if len(output_vals) == 1 else output_vals  # type: ignore[assignment]
        partition.graph.output(output_vals)

        # Construct GraphModule for this partition
        partition.graph_module = GraphModule(partition.targets, partition.graph)

        # Connnect output edges
        for dependent in partition.partition_dependents:
            dep_inputs = partitions[dependent].inputs
            for input_name in dep_inputs:
                if input_name in partition.outputs:
                    partition.output_map[input_name].add(dependent)

        for dependent in partition.partitions_dependent_on:
            dep_outputs = partitions[dependent].outputs
            for output_name in dep_outputs:
                if output_name in partition.inputs:
                    partition.input_map[output_name] = dependent

        for name in partition.inputs:
            node = orig_nodes[name]
            partition.input_shapes.append(node.meta["_shape"])

        for name in partition.outputs:
            node = orig_nodes[name]
            partition.output_shapes.append(node.meta["_shape"])

    for out_node, part_idx in output_node_map:
        out_idx = partitions[part_idx].outputs.index(out_node.name)
        global_outputs.append((part_idx, out_idx))

    for idx, partition in partitions.items():
        part_graph.nodes[idx] = PartitionNode(partition)

    return part_graph


def module_set_attr(module, target: str, attr):
    target_atoms = target.split(".")
    attr_itr = module
    for atom in target_atoms[:-1]:
        attr_itr = getattr(attr_itr, atom)
    setattr(attr_itr, target_atoms[-1], attr)


def module_fetch_attr(module, target):
    target_atoms = target.split(".")
    attr_itr = module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def _rearrange_graph(gm: GraphModule, pgraph: PipeGraph):
    nodes: List[fx.Node] = []
    split_map: Dict[Node, int] = {}
    cgraph = pgraph.comp_graph

    # copy params for shared memory
    param_set = pgraph.comp_graph.param_set
    pnode_params: Dict[int, Set[PipeNode]] = defaultdict(set)  # param_id -> pnode
    part_graph_target = param_set.part_graph_target
    shared_idx = param_set.shared_idx
    shared_rank = param_set.shared_rank

    for idx, pnode in enumerate(pgraph.nodes):
        for param in pnode.params:
            pnode_params[param].add(pnode)

    for param_id, pnodes in pnode_params.items():
        if len(pnodes) == 1:
            continue

        rank_dict: Dict[int, List[PipeNode]] = defaultdict(list)
        for pnode in pnodes:
            rank_dict[pnode.rank].append(pnode)

        if len(rank_dict) == 1:
            continue

        for _, pnodes in rank_dict.items():
            for pnode in pnodes:
                for attr in pnode.get_attrs:
                    target = attr.target
                    if param_id in param_set.param_set[target]:
                        new_target = f"{target}_device_{pnode.rank}"
                        part_graph_target[(target, pnode.rank)] = (new_target, True)
                        param_set.param_set[new_target] = param_set.param_set[target]
                        shared_idx[new_target].append(param_id)
                        shared_rank[param_id].add(pnode.rank)
                for mod in pnode.call_modules:
                    if param_id in param_set.param_set[mod]:
                        new_target = f"{mod}_device_{pnode.rank}"
                        part_graph_target[(mod, pnode.rank)] = (new_target, False)
                        param_set.param_set[new_target] = param_set.param_set[mod]
                        shared_idx[new_target].append(param_id)
                        shared_rank[param_id].add(pnode.rank)

    targets_to_remove = []

    for (target, rank), (new_target, is_attr) in part_graph_target.items():
        print(f"shared weight moved: {target} / {rank} -> {new_target} / {is_attr}")
        value = module_fetch_attr(gm, target)
        new_value = deepcopy(value)
        module_set_attr(gm, new_target, new_value)

        targets_to_remove.append(target)

        new_attr_node = gm.graph.get_attr(new_target) if is_attr else None

        # update attr
        for pnode in pgraph.nodes:
            if pnode.rank != rank:
                continue

            attr_node = None
            for attr in pnode.get_attrs:
                if attr.target == target:
                    attr_node = attr
                    break
            if attr_node is None:
                continue

            pnode.get_attrs.remove(attr_node)
            pnode.get_attrs.add(new_attr_node)

            for cnode in pnode.nodes:
                cnode_nodes = []
                for node in cnode.nodes:
                    if attr_node in node.args:
                        node.args = tuple(
                            new_attr_node if arg == attr_node else arg
                            for arg in node.args
                        )
                    if attr_node in node.kwargs.values():
                        node.kwargs = {
                            k: new_attr_node if v == attr_node else v
                            for k, v in node.kwargs.items()
                        }
                    cnode_nodes.append(node)
                cnode.nodes = cnode_nodes
                cnode.get_attrs.remove(attr_node)
                cnode.get_attrs.add(new_attr_node)

        # update call_modules
        for pnode in pgraph.nodes:
            if pnode.rank != rank:
                continue

            if target in pnode.call_modules:
                pnode.call_modules.remove(target)
                pnode.call_modules.add(new_target)

                for cnode in pnode.nodes:
                    if target in cnode.call_modules:
                        cnode.call_modules.remove(target)
                        cnode.call_modules.add(new_target)

                        for node in cnode.nodes:
                            if node.op == "call_module" and node.target == target:
                                node.target = new_target

    for target in targets_to_remove:
        module_set_attr(gm, target, None)  # remove old target

    # re-draw the graph
    nodes.extend(cgraph.placeholders)
    for place in cgraph.placeholders:
        split_map[place] = pgraph.nodes[0].idx

    getattr_set = set()
    for pnode in pgraph.nodes:
        # sort pnode.get_attrs
        get_attr_list = list(pnode.get_attrs)
        get_attr_list.sort(key=lambda x: x.target)
        for attr in get_attr_list:
            if attr not in getattr_set:
                nodes.append(attr)
                getattr_set.add(attr)
        # nodes.extend(pnode.get_attrs)
        for cnode in pnode.nodes:
            nodes.extend(cnode.nodes)

        for node in pnode.get_attrs:
            split_map[node] = pnode.idx
        for cnode in pnode.nodes:
            for node in cnode.nodes:
                split_map[node] = pnode.idx

    nodes.append(cgraph.output)
    split_map[cgraph.output] = pgraph.nodes[-1].idx

    curr = gm.graph._root
    for n in nodes:
        curr._next = n
        curr = n
    for n in reversed(nodes[:-1]):
        curr._prev = n
        curr = n
    curr._prev = gm.graph._root

    # gm.graph.nodes = nodes
    gm.recompile()

    return split_map


def reset_graph_device(gm: GraphModule, device: torch.device):
    should_recompile = False

    for n in gm.graph.nodes:
        moved_kv = []
        for k, v in n.kwargs.items():
            if isinstance(v, torch.device):
                moved_kv.append((k, v))
        if len(moved_kv) > 0:
            values = dict(**n.kwargs)
            for k, v in moved_kv:
                values[k] = device
            n.kwargs = values
            should_recompile = True

        has_arg = False
        for val in n.args:
            if isinstance(val, torch.device):
                has_arg = True
                break
        if has_arg:
            args = []
            for a in n.args:
                if isinstance(a, torch.device):
                    args.append(device)
                else:
                    args.append(a)
            n.args = tuple(args)
            should_recompile = True

    if should_recompile:
        gm.recompile()


def partition_module_by_pgraph(
    gm: GraphModule, pgraph: PipeGraph, devices=None
) -> PartitionGraph:
    split_map: Dict[Node, int] = _rearrange_graph(gm, pgraph)

    part_graph = partition_module(gm, split_map)

    for pnode in pgraph.nodes:
        idx = pnode.idx
        part_node = part_graph.nodes[idx]
        part_node.rank = pnode.rank

        gm = part_node.graph_module
        if devices is not None:
            reset_graph_device(gm, devices[pnode.rank])

        # print(f">>>>>>>>>>>>>>partition {idx} rank: {pnode.rank}<<<<<<<<<<<<<<<")
        # gm.print_readable()

    return part_graph
