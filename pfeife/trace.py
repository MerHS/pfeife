from typing import Callable, Optional, List
from inspect import Signature, Parameter

import torch
import torch.fx as fx
from torch.fx import Node
from torch.utils._pytree import tree_flatten, tree_map


def graph_break(value, device):
    def tensor_map(v):
        if torch.is_tensor(v):
            # return v.to(device)
            return v.detach().to(device).requires_grad_(True).clone()
        else:
            return v

    return tree_map(tensor_map, value)


class TraceInterpreter(fx.Interpreter):
    def __init__(self, module, args, kwargs):
        super().__init__(module)
        self.pc = 0
        self.env = {}
        self.node_list = list(self.module.graph.nodes)
        self.target_device = "cuda:0"

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

    def run_until(self, predicate: Callable[[Node], bool]) -> Optional[Node]:
        result = None
        while self.pc < len(self.node_list):
            node = self.node_list[self.pc]

            if predicate(node):
                return result

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

            def tensor_detach(v):
                if torch.is_tensor(v):
                    return v.detach().to(self.target_device).requires_grad_(True)
                else:
                    return v

            def tensor_clone(v):
                if torch.is_tensor(v):
                    return v.clone()
                else:
                    return v

            leaf = tree_map(tensor_detach, value)
            self.last_leaf = leaf

            return tree_map(tensor_clone, leaf)

        return target(*args, **kwargs)


class StepTrace:
    """
    Save the result of each submodule for backward
    # TODO: use rpc
    """

    def __init__(self, init_args, init_kwargs, gm: fx.GraphModule):
        self.gm = gm
        self.interpreter = TraceInterpreter(gm, init_args, init_kwargs)
        self.forward_results = []
        self.forward_sents = []
        self.last_result = None
        self.last_grad = None

    def forward_step(self):
        result = self.interpreter.run_until(lambda n: n.target == graph_break)
        self.forward_results.append(result)
        self.last_result = result
        return result

    def forward_send(self, device):
        self.interpreter.target_device = device
        sent = self.interpreter.run_next()

        self.forward_sents.append(self.interpreter.last_leaf)
        self.last_result = sent
        return sent

    def calc_loss(self, target, loss_fn):
        loss = loss_fn(self.last_result[0], target)
        self.forward_results[len(self.forward_results) - 1] = loss
        return loss

    def backward_step(self):
        last_result = self.forward_results.pop()
        torch.autograd.backward(last_result, grad_tensors=self.last_grad)

    def backward_grad_send(self, device):
        def map_grad(v):
            if torch.is_tensor(v) and v.requires_grad:
                return v.grad.to(device)
            else:
                return None

        grads = tree_map(map_grad, self.forward_sents.pop())
        self.last_grad = grads
