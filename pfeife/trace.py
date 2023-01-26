import threading
from typing import Callable, Optional, List
from inspect import Signature, Parameter

import torch
import torch.fx as fx
import torch.optim as optim
from torch.fx import Node
from torch.distributed.rpc import PyRRef

from torch._dynamo.optimizations import BACKENDS
from torch._inductor.compile_fx import compile_fx
from torch.utils._pytree import tree_flatten, tree_map


from .utils import to_device


def graph_break(value, device):
    # def tensor_map(v):
    #     if torch.is_tensor(v):
    #         # return v.to(device)
    #         return v.detach().to(device).requires_grad_(True).clone()
    #     else:
    #         return v

    # return tree_map(tensor_map, value)
    return value


class StepTrace:
    """
    Save the result of each submodule for backward
    # TODO: find grad from a u-net like model
    """

    def __init__(self, interpreter: TraceInterpreter):
        self.interpreter = interpreter
        self.rpc_stages = self.interpreter.rpc_stages
        self.back_stage = len(self.rpc_stages) - 1
        self.batch_no = self.interpreter.batch_no
        self.last_grads = None
        self.last_step = None

    def forward_step(self):
        if self.last_step is not None:
            self.last_step.wait()

        self.last_step = self.interpreter.run_until(lambda n: n.target == graph_break)

    def forward_loss(self, target, loss_fn):
        if self.last_step is not None:
            for v in self.last_step:
                v.wait()

        target = to_device(target, "cpu")
        loss = (
            self.rpc_stages[self.back_stage]
            .rpc_async()
            .forward_loss(self.batch_no, target, loss_fn)
        )

        return loss

    def backward_step(self):
        grads = self.last_grads
        grads = grads.wait() if grads is not None else grads

        self.last_grads = (
            self.rpc_stages[self.back_stage].rpc_async().backward(self.batch_no, grads)
        )
        self.back_stage -= 1

        return self.last_grads
