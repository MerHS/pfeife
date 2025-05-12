import torch
from torch import _TorchCompileInductorWrapper
from torch._dynamo.eval_frame import get_compiler_fn
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode

class WrapperModule(torch.nn.Module):
    def __init__(self, compiled_submod, unwrap_singleton_tuple):
        super().__init__()
        self.compiled_submod = compiled_submod
        self.unwrap_singleton_tuple = unwrap_singleton_tuple

    def forward(self, *args):
        x = self.compiled_submod(*args)
        if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
            return x[0]
        return x


def compile_module(compiler, submod, inputs):
    unwrap_singleton_tuple = False
    for sn in submod.graph.nodes:
        if sn.op == "output":
            if not isinstance(sn.args[0], tuple):
                unwrap_singleton_tuple = True
                sn.args = (sn.args,)
    submod.recompile()

    # fn = torch.compile(submod, backend=compiler)
    if compiler == 'inductor':
        compiler = _TorchCompileInductorWrapper("max-autotune", None, False)
    cpx_fn = get_compiler_fn(compiler)

    # fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    # new_args = []
    # for arg in inputs:
    #     if isinstance(arg, torch.Tensor) and not isinstance(
    #         arg, torch._subclasses.FakeTensor
    #     ):
    #         new_args.append(
    #             torch._dynamo.utils.to_fake_tensor(arg, fake_mode)
    #         )
    #     else:
            # new_args.append(arg)

    # fake_mode = detect_fake_mode(inputs)
    # print(f"fake_mode: {fake_mode}")
    # if fake_mode is None:
    #     fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

    # if fake_mode:
        # submod = deepcopy_to_fake_tensor(submod, inputs)

    fn = cpx_fn(submod, inputs)

    wrapper = WrapperModule(
        fn,
        unwrap_singleton_tuple,
    )

    return wrapper
