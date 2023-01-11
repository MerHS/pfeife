import logging
import os
import copy
from typing import Any, List, Optional

import torch
import torch.fx.traceback as fx_traceback
from torch import fx
from torch.fx.node import Node
# from torch.distributed.pipeline.sync import Pipe

from torch._dynamo.optimizations import BACKENDS
from .distributed import Bucket
from torch._inductor.compile_fx import compile_fx
from .graph_drawer import FxGraphDrawer

from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses import UnsupportedFakeTensorException
from torch.utils._pytree import tree_flatten

log = logging.getLogger(__name__)
PIPE_BACKEND = ['inductor']

def wrap_fake_exception(fn):
    try:
        return fn()
    except UnsupportedFakeTensorException as e:
        from torch._dynamo.exc import unimplemented

        msg = f"Unsupported: {e.reason} with fake tensor propagation."
        log.warning(msg)
        raise unimplemented(msg) from e

def fake_mode_from_tensors(inputs: List[Any]):
    """
    Takes a list of anything, unflattened is fine, returns a fake_mode
    if any are fake. All fake modes on all fake tensors must be identical.
    Returns None if no fake_mode is fine
    """
    flat_inputs, _ = tree_flatten(inputs)
    fake_mode = None
    for flat_input in flat_inputs:
        if isinstance(flat_input, torch._subclasses.FakeTensor):
            if fake_mode is None:
                fake_mode = flat_input.fake_mode
            else:
                assert fake_mode is flat_input.fake_mode
    return fake_mode

def deepcopy_to_fake_tensor(obj, fake_mode):
    with torch._subclasses.fake_tensor.FakeCopyMode(fake_mode):
        return wrap_fake_exception(lambda: copy.deepcopy(obj))


class PipelineOptimizer:
    """
    Very basic & hand-written 2-GPU auto-pipeliner
    """
    def __init__(
        self,
        backend_compile_fn,
        first_bucket_cap: Optional[int] = None,
    ):
        self.backend_compile_fn = backend_compile_fn

    def _ignore_parameter(self, parameter):
        return hasattr(parameter, "_ddp_ignored") and parameter._ddp_ignored

    def compile_fn(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        fake_mode = fake_mode_from_tensors(example_inputs)
        if fake_mode is None:
            fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        gpu_n = 2
        total_bytes = 0

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

        bucket_bytes = total_bytes // gpu_n

        # 1: compute the partition map according to bucket logic
        buckets = [Bucket()]  # (size, param_names)
        for node in gm.graph.nodes:
            if node.op in ("output", "placeholder"):
                continue
            
            buck = buckets[len(buckets) - 1]

            if buck.size >= bucket_bytes and len(buckets) < gpu_n:
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

        if len(buckets) == 1:
            # bypass split/fuse logic if there is only one bucket
            return self.backend_compile_fn(gm, example_inputs)

        # 2: partition the graphmodule according to bucket capacity
        partition_map = {}
        for idx, b in enumerate(buckets):
            for node in b.nodes:
                node.meta["part_idx"] = idx
                partition_map[node] = idx

        # print node
        if log.level == logging.INFO:
            for node in gm.graph.nodes:
                if "part_idx" in node.meta:
                    node.meta["print_meta"] = { "gpu": node.meta["part_idx"] }

            g = FxGraphDrawer(gm, "Pipelined")
            dot = g.get_dot_graph()

            with open(f"{os.path.abspath(os.path.dirname(__file__))}/pipelined.svg", "wb") as f:
                f.write(dot.create_svg())

        split_gm = fx.passes.split_module.split_module(
            gm, None, lambda node: partition_map[node]
        )

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

        # 3: compile each of the partitioned submodules using the user-provided compiler
        class SubmodCompiler(torch.fx.interpreter.Interpreter):
            def __init__(self, module, compiler):
                super().__init__(module)
                self.compiler = compiler
                self.submodule_idx = 0

            def compile_submod(self, submod, args, kwargs):
                """
                Compile the submodule,
                using a wrapper to make sure its output is always a tuple,
                which is required by AotAutograd based compilers
                """
                assert len(kwargs) == 0, "We assume only args for these modules"

                class WrapperModule(torch.nn.Module):
                    def __init__(self, compiled_submod, unwrap_singleton_tuple):
                        super().__init__()
                        self.compiled_submod = compiled_submod
                        self.unwrap_singleton_tuple = unwrap_singleton_tuple

                    def forward(self, *args):
                        x = self.compiled_submod(*args)
                        # TODO(whc)
                        # for some reason the isinstance check is necessary if I split one node per submod
                        # - even though I supposedly wrapped the output in a tuple in those cases, the real
                        # compiled module was still returning a tensor
                        if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                            return x[0]
                        return x

                unwrap_singleton_tuple = False
                for sn in submod.graph.nodes:
                    if sn.op == "output":
                        if not isinstance(sn.args[0], tuple):
                            unwrap_singleton_tuple = True
                            sn.args = (sn.args,)
                submod.recompile()

                wrapper = WrapperModule(
                    self.compiler(submod, args),
                    unwrap_singleton_tuple,
                )
                return wrapper

            # Note:
            #
            # The way distributed works today around fake tensors can be somehwat confusing.
            # Some of these codepaths are shared in both runtime, and compile time. The presence
            # of a fake_mode, read off of fake tensor inputs, dictates how we will operate.
            #
            # A few things to keep in mind:
            #
            # 1) We invoke `compile_submod` with a real module. The output of that gets stored
            # on the graph via `self.module.add_submodule(n.target, compiled_submod_real)`.
            #
            # 2) When running a call_module targeted node, if we have a fake_mode, we fakify the
            # module we got from self.fetch_attr(n.target). Regardless of fake_mode, we then execute it.
            #
            # 3) Fake tensors should always be around during compile time.
            #
            # 4) Fake tensors should never be around at runtime.
            #
            # 5) We end up with a compilation mode that takes a real submodule and fake tensors,
            # to match what aot_autograd exepcts. See Note: [Fake Modules and AOTAutograd]
            def run_node(self, n: Node) -> Any:
                with fx_traceback.append_stack_trace(n.stack_trace):
                    # TODO: 

                    args, kwargs = self.fetch_args_kwargs_from_env(n)
                    new_args = []
                    assert fake_mode
                    for arg in args:
                        if isinstance(arg, torch.Tensor) and not isinstance(
                            arg, torch._subclasses.FakeTensor
                        ):
                            new_args.append(fake_mode.from_tensor(arg))
                        else:
                            new_args.append(arg)

                    assert isinstance(args, tuple)
                    assert isinstance(kwargs, dict)

                    # modify the currently running FX graph
                    # maybe this isn't sound in general, but only changing the target of a node might be ok?
                    if n.op == "call_module":
                        idx = self.submodule_idx
                        device = f'cuda:{idx}'
                        real_mod = self.fetch_attr(n.target)
                        real_mod = real_mod.to(device)

                        if fake_mode:
                            curr_submod = deepcopy_to_fake_tensor(real_mod, fake_mode)
                        else:
                            curr_submod = real_mod

                        compiled_submod_real = self.compile_submod(
                            real_mod, new_args, kwargs
                        )
                        compiled_submod_real = compiled_submod_real.to(device)

                        self.module.delete_submodule(n.target)
                        n.target = "compiled_" + n.target
                        self.module.add_submodule(n.target, compiled_submod_real)

                        # TODO: prepend nodes of .to('cuda:n') for every args & kwargs
                        to_args = []
                        for arg in n.args:
                            new_arg = arg
                            if isinstance(arg, Node):
                                # TODO: check arg is raw tensor
                                new_arg = Node(arg.graph, f'{arg.name}_moved', 'call_method', 'to', (arg, device), dict())
                                n.prepend(new_arg)
                            to_args.append(new_arg)
                        n.args = tuple(to_args)

                        self.submodule_idx += 1
                        return curr_submod(*new_args, **kwargs)
                        
                    return getattr(self, n.op)(n.target, new_args, kwargs)

        submod_compiler = SubmodCompiler(split_gm, self.backend_compile_fn)
        submod_compiler.run(*example_inputs)
        split_gm.recompile()

        log.info("\n---final graph---\n" + str(split_gm.graph) + "\n---------------\n")

        return split_gm

def pipeline_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    backend = PIPE_BACKEND[0]
    if backend == 'inductor':
        torch._inductor.config.triton.cudagraphs = False
        backend = compile_fx
    else:
        backend = BACKENDS[backend]
    opt = PipelineOptimizer(backend)
    return opt.compile_fn(gm, example_inputs)
