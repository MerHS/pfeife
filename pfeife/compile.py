import torch
from torch._dynamo.optimizations import BACKENDS
from torch._inductor.compile_fx import compile_fx


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


def compile_module(compiler, submod, args):
    if compiler == "inductor":
        torch._inductor.config.triton.cudagraphs = False
        compiler = compile_fx
    else:
        compiler = BACKENDS[compiler]

    unwrap_singleton_tuple = False
    for sn in submod.graph.nodes:
        if sn.op == "output":
            if not isinstance(sn.args[0], tuple):
                unwrap_singleton_tuple = True
                sn.args = (sn.args,)
    submod.recompile()

    wrapper = WrapperModule(
        compiler(submod, args),
        unwrap_singleton_tuple,
    )

    return wrapper
