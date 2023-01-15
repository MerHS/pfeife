import torch
import torch.fx as fx


class Trace:
    """
    Save the result of each submodule for backward
    """

    def __init__(self, init_args, init_kwargs, gm: fx.GraphModule):
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        self.gm = gm
        self.curr_device = 0

    def forward_step(self):
        pass

    def calc_loss(self, target, loss_fn):
        pass

    def backward_step(self):
        pass
