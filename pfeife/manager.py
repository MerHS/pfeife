from typing import List, Any
import copy

import torch
import torch.nn as nn
import torch.fx as fx
import torch.optim as optim
from torch.fx import Node
import torch.fx.traceback as fx_traceback
import torch.distributed.rpc as rpc

import torch._dynamo as dynamo
from torch._subclasses import UnsupportedFakeTensorException
from torch.utils._pytree import tree_flatten

from .graph_split import ParamSplit
from .utils import get_logger
from .batch import send_value, split_args_kwargs
from .trace import graph_break, StepTrace, RPCStage, RPCProxyModule
from .scheduler import SchedGPipe, Sched1F1B

log = get_logger()


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


class SubmodCompiler(torch.fx.interpreter.Interpreter):
    # compile each of the partitioned submodules using the user-provided compiler
    def __init__(self, manager: "PipeManager", module, compiler, fake_mode):
        super().__init__(module)
        self.manager = manager
        self.compiler = compiler
        self.submodule_idx = 0
        self.fake_mode = fake_mode

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
            fake_mode = self.fake_mode

            args, kwargs = self.fetch_args_kwargs_from_env(n)
            new_args = []
            assert self.fake_mode

            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            # modify the currently running FX graph
            # maybe this isn't sound in general, but only changing the target of a node might be ok?
            if n.op == "call_module":
                idx = self.submodule_idx

                # TODO: set next_device
                device_max = self.manager.pipe_split
                device = f"cuda:{idx}"
                next_device = f"cuda:{idx+1}"

                for arg in args:
                    if torch.is_tensor(arg):
                        arg = arg.to("cpu")

                    if isinstance(arg, torch.Tensor) and not isinstance(
                        arg, torch._subclasses.FakeTensor
                    ):
                        new_args.append(fake_mode.from_tensor(arg))
                    else:
                        new_args.append(arg)

                real_mod = self.fetch_attr(n.target)
                real_mod = real_mod.to("cpu")

                if fake_mode:
                    curr_submod = deepcopy_to_fake_tensor(real_mod, fake_mode)
                else:
                    curr_submod = real_mod

                stage = self.manager.add_stage(idx, real_mod, self.compiler)
                stage_mod = RPCProxyModule(stage)

                self.module.delete_submodule(n.target)
                n.target = "compiled_" + n.target
                self.module.add_submodule(n.target, stage_mod)

                # break graph
                if device_max > idx + 1:
                    send_arg = Node(
                        n.graph,
                        f"{n.name}_sent",
                        "call_function",
                        graph_break,
                        (n, next_device),
                        dict(),
                    )
                    n.replace_all_uses_with(send_arg)
                    send_arg.args = (n, next_device)
                    n.append(send_arg)

                self.submodule_idx += 1
                return curr_submod(*new_args, **kwargs)
            else:
                for arg in args:
                    if isinstance(arg, torch.Tensor) and not isinstance(
                        arg, torch._subclasses.FakeTensor
                    ):
                        new_args.append(fake_mode.from_tensor(arg))
                    else:
                        new_args.append(arg)

            return getattr(self, n.op)(n.target, new_args, kwargs)


class TraceGenerator(nn.Module):
    def __init__(self, manager: "PipeManager", gm: fx.GraphModule):
        super().__init__()
        self.manager = manager
        self.gm = gm

    def forward(self, *args, **kwargs):
        trace = StepTrace(args, kwargs, self.gm, self.manager.rpc_stages)
        return [trace]


class PipeManager:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer="adam",
        dynamo_backend="aot_eager",
        scheduler="gpipe",
        pipe_split=2,
        batch_split=4,
        graph_splitter=None,
    ):
        super().__init__()
        self.orig_model = model
        self.pipe_split = pipe_split
        self.batch_split = batch_split
        self.loss_fn = loss_fn

        sched_cls = Sched1F1B if scheduler == "1f1b" else SchedGPipe
        self.sched = sched_cls(device_cnt=pipe_split, batch_cnt=batch_split)

        self.splitter = (
            ParamSplit(pipe_split) if graph_splitter is None else graph_splitter
        )
        self.dynamo_backend = dynamo_backend
        self.rpc_stages = []

        self._compile()

        # TODO: late initialize optimizer after dynamo compilation?
        optimizer_cls = optim.Adam if optimizer == "adam" else optimizer
        self.optimizer = optimizer_cls(self.opt_model.parameters())

    def _compile(self):
        def compile_fn(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            fake_mode = fake_mode_from_tensors(example_inputs)
            if fake_mode is None:
                fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

            split_gm = self.splitter.split(gm)

            submod_compiler = SubmodCompiler(
                self, split_gm, self.dynamo_backend, fake_mode
            )
            submod_compiler.run(*example_inputs)
            split_gm.recompile()

            log.info(
                "\n---final graph---\n" + str(split_gm.graph) + "\n---------------\n"
            )
            print(split_gm.graph)

            trace_gen = TraceGenerator(self, split_gm)

            return trace_gen

        def pipeline_compiler(
            gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
        ):
            return compile_fn(gm, example_inputs)

        dynamo_ctx = dynamo.optimize(pipeline_compiler)
        self.opt_model = dynamo_ctx(self.orig_model)

    def add_stage(self, idx, stage_mod, compiler):
        stage = rpc.remote(f"worker_{idx+1}", RPCStage, args=(idx, stage_mod, compiler))
        self.rpc_stages.append(stage)
        return stage

    def train(self):
        self.opt_model.train()

    def eval(self):
        self.opt_model.eval()

    def run(self, target, *args, **kwargs):
        """
        1. optimizer.zero_grad()
        2. forward & backward & optimizer step
        3. return loss
        """

        for stage in self.rpc_stages:
            stage.rpc_sync().reset_cache()

        self.optimizer.zero_grad()

        # TODO: should we move args to gpu 0 before splitting?
        args = send_value(args, "cuda:0")
        kwargs = send_value(kwargs, "cuda:0")

        def pmap(v):
            if torch.is_tensor(v):
                return v.shape
            else:
                return v

        # TODO: check last device
        target = send_value(target, f"cuda:{self.pipe_split - 1}")

        args, kwargs = split_args_kwargs(args, kwargs, self.batch_split)
        targets, _ = split_args_kwargs([target], dict(), self.batch_split)

        traces: List[StepTrace] = []
        for micro_args, micro_kwargs in zip(args, kwargs):
            traces.append(self.opt_model(*micro_args, **micro_kwargs))

        # returns losses
        losses = self.sched.exec(traces, targets, self.loss_fn)
        loss = sum([l for l in losses if torch.is_tensor(l)])

        self.optimizer.step()

        return loss
