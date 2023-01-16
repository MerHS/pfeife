import threading
from typing import Callable, Optional, List
from inspect import Signature, Parameter

import torch
import torch.fx as fx
from torch.fx import Node
from torch.utils._pytree import tree_flatten, tree_map

from torch._dynamo.optimizations import BACKENDS
from torch._inductor.compile_fx import compile_fx


def graph_break(value, device):
    # def tensor_map(v):
    #     if torch.is_tensor(v):
    #         # return v.to(device)
    #         return v.detach().to(device).requires_grad_(True).clone()
    #     else:
    #         return v

    # return tree_map(tensor_map, value)
    return value


class TraceInterpreter(fx.Interpreter):
    def __init__(self, module, args, kwargs, rpc_stages):
        super().__init__(module)
        self.pc = 0
        self.env = {}
        self.node_list = list(self.module.graph.nodes)
        self.rpc_stages = rpc_stages
        self.curr_batch = 0
        self.curr_stage = 0

        # process args and kwargs
        parameters = []
        # if enable_io_processing:
        #     args = self.module.graph.process_inputs(*args)
        for node in self.module.graph.nodes:
            if node.op != "placeholder":
                continue
            default = next(iter(node.args)) if node.args else Parameter.empty
            parameters.append(
                Parameter(node.name, Parameter.POSITIONAL_OR_KEYWORD, default=default)
            )
        sig = Signature(parameters)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        self.args = bound_args.args
        self.args_iter = iter(self.args)
        self.last_leaf = None

    def run_until(
        self, predicate: Callable[[Node], bool], curr_batch
    ) -> Optional[Node]:
        result = None
        ignore_this = True
        self.curr_batch = curr_batch

        while self.pc < len(self.node_list):
            node = self.node_list[self.pc]

            if not ignore_this and predicate(node):
                return result

            ignore_this = False

            result = self.run_node(node)

        return result

    def run_node(self, node):
        self.env[node] = super().run_node(node)
        self.pc += 1
        return self.env[node]

    def run_next(self):
        node = self.node_list[self.pc]
        return self.run_node(node)

    def call_function(self, target, args, kwargs):
        if target == graph_break:
            value, device = args

            # value is future
            def tensor_wait(v):
                if isinstance(v, torch.futures.Future):
                    return v.wait()
                else:
                    return v

            return tree_map(tensor_wait, value)

        return target(*args, **kwargs)

    def call_module(self, target, args, kwargs):
        stage = self.rpc_stages[self.curr_stage]

        args = [a.cpu() for a in args]

        result = stage.rpc_async().forward(self.curr_batch, *args)
        self.curr_stage += 1

        return result


class RPCProxyModule(torch.nn.Module):
    def __init__(self, stage_ref):
        super().__init__()
        self.stage_ref = stage_ref

    def forward(self, *args):
        # NOOP
        return args


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


class RPCStage:
    # TODO: stage_no-based, not rank
    def __init__(self, rank, stage_mod, compiler):
        # TODO: should we use lock?
        self.lock = threading.Lock()
        self.rank = rank
        self.device = f"cuda:{rank}"
        self.stage_mod = stage_mod.to(self.device)
        self.is_compiled = False
        self.compiler = compiler

        if type(compiler) == str:
            if compiler == "inductor":
                torch._inductor.config.triton.cudagraphs = False
                self.compiler = compile_fx
            else:
                self.compiler = BACKENDS[compiler]

        self.reset_cache()

    def reset_cache(self):
        self.fw_results = dict()
        self.fw_inputs = dict()

    def compile_submod(self, args):
        def to_device(v):
            if torch.is_tensor(v):
                return v.to(self.device)
            else:
                return v

        args = tree_map(to_device, args)
        submod = self.stage_mod

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

        self.stage_mod = wrapper

    def forward(self, batch_no, *args):
        if self.rank == 0:
            args = [a.to(self.device) for a in args]
        else:
            fw_args = [a.detach().to(self.device).requires_grad_(True) for a in args]
            self.fw_inputs[batch_no] = fw_args
            args = [a.clone() for a in fw_args]

        with self.lock:
            if not self.is_compiled:
                self.compile_submod(args)
                self.is_compiled = True
            result = self.stage_mod(*args)
            self.fw_results[batch_no] = result

        result_cpu = [r.detach().cpu() for r in result]

        return result_cpu

    def backward(self, batch_no, grads):
        grads = [a.to(self.device) for a in grads]

        with self.lock:
            torch.autograd.backward(self.fw_results[batch_no], grads)

        next_grads = []

        if self.rank != 0:
            back_input = self.fw_inputs[batch_no]
            next_grads = [g.grad.cpu() for g in back_input]

        return next_grads

    def backward_last(self, batch_no, target, loss_fn):
        res = self.fw_results[batch_no][0]
        target = target.to(self.device)

        with self.lock:
            loss = loss_fn(res, target)
            torch.autograd.backward(loss, None)

        next_grads = []
        if self.rank != 0:
            back_input = self.fw_inputs[batch_no]
            next_grads = [g.grad.cpu() for g in back_input]

        return next_grads


class StepTrace:
    """
    Save the result of each submodule for backward
    # TODO: use rpc
    # TODO: find grad from a u-net like model
    """

    def __init__(self, init_args, init_kwargs, gm: fx.GraphModule, rpc_stages):
        self.interpreter = TraceInterpreter(gm, init_args, init_kwargs, rpc_stages)
        self.rpc_stages = rpc_stages

    def forward_step(self, batch_no):
        result = self.interpreter.run_until(lambda n: n.target == graph_break, batch_no)
        return result

    def backward_step(self, batch_no):
        # torch.autograd.backward(last_result, grad_tensors=self.last_grad)
        pass
