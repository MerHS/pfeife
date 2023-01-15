from dataclasses import dataclass, field
from typing import Any, List, Optional
import logging

import torch
import torch.fx as fx
from torch.fx.graph_module import GraphModule

from .utils import get_logger


@dataclass
class Bucket:
    size: int = 0
    params: List[str] = field(default_factory=list)
    nodes: List[fx.Node] = field(default_factory=list)

    # param_ids is just used for unit testing
    param_ids: List = field(default_factory=list)


class GraphSplitter:
    def split(self, gm: GraphModule):
        raise NotImplementedError()


class ParamSplit(GraphSplitter):
    def __init__(self, split_cnt=2):
        super().__init__()
        self.split_cnt = split_cnt

    def _ignore_parameter(self, parameter):
        # TODO: handle it. What's this for?
        return hasattr(parameter, "_ddp_ignored") and parameter._ddp_ignored

    def split(self, gm: GraphModule):
        total_bytes = 0
        split_cnt = self.split_cnt

        for node in gm.graph.nodes:
            if node.op == "call_module":
                target = gm.get_submodule(node.target)
                for name, p in target.named_parameters():
                    param = target.get_parameter(name)
                    if p.requires_grad and not self._ignore_parameter(param):
                        total_bytes += p.untyped_storage().nbytes()
            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if maybe_param.requires_grad and not self._ignore_parameter(
                    maybe_param
                ):
                    total_bytes += maybe_param.untyped_storage().nbytes()

        bucket_bytes = total_bytes // split_cnt

        # 1: compute the partition map according to bucket logic
        buckets = [Bucket()]  # (size, param_names)
        for node in gm.graph.nodes:
            if node.op in ("output", "placeholder"):
                continue

            buck = buckets[len(buckets) - 1]

            if buck.size >= bucket_bytes and len(buckets) < split_cnt:
                buckets.append(Bucket())

            if node.op == "call_module":
                target = gm.get_submodule(node.target)
                for name, p in target.named_parameters():
                    param = target.get_parameter(name)
                    if p.requires_grad and not self._ignore_parameter(param):
                        buck.size += p.untyped_storage().nbytes()
                        buck.params.append(f"{node.target}_{name}")
                        buck.param_ids.append(id(param))
            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if maybe_param.requires_grad and not self._ignore_parameter(
                    maybe_param
                ):
                    buck.size += maybe_param.untyped_storage().nbytes()
                    buck.params.append(node.target)
                    buck.param_ids.append(id(maybe_param))

            # All nodes have to be mapped to a bucket, even if they don't have their own params
            # Ignored params still end up in buckets, we just don't count them towards the capacity
            buck.nodes.append(node)

        # stash buckets for testing/debugging purposes
        self.buckets = buckets

        # 2: partition the graphmodule according to bucket capacity
        partition_map = {}
        for idx, b in enumerate(buckets):
            for node in b.nodes:
                node.meta["part_idx"] = idx
                partition_map[node] = idx

        split_gm = fx.passes.split_module.split_module(
            gm, None, lambda node: partition_map[node]
        )

        log = get_logger()

        if log.level == logging.INFO:
            debug_str = (
                f"\n---orig graph---\n{gm.graph}\n"
                + f"\n---split graph---\n{split_gm.graph}\n"
            )
            for name, module in split_gm.named_modules():
                if "." not in name and len(name):
                    # only print the submod graphs, not their children
                    debug_str += f"\n---{name} graph---\n{module.graph}\n"
            debug_str += "\n---------------\n"
            log.info(debug_str)

        return split_gm
