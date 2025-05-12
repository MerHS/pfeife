import importlib
import os
import random
import sys
import time
import numbers
from os.path import abspath, exists
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules.dropout import _DropoutNd as DropoutOrigin
from torch.nn.modules.batchnorm import _BatchNorm as BatchNormOrigin
from torch.nn import LayerNorm as LayerNormOrigin


class _NormBase(torch.nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, **factory_kwargs)
            )
            self.register_buffer(
                "running_var", torch.ones(num_features, **factory_kwargs)
            )
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        return input


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input):
        return input


class _DropoutNd(torch.nn.Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)

    def forward(self, input):
        return input


def replace_batch_norm(model):
    for name, module in model.named_children():
        if isinstance(module, BatchNormOrigin):
            bn = _BatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
            )
            setattr(model, name, bn)
        # elif isinstance(module, LayerNormOrigin):
        #     ln = LayerNorm(
        #         module.normalized_shape,
        #         module.eps,
        #         module.elementwise_affine,
        #         module.bias is not None,
        #     )
        #     setattr(model, name, ln)
        elif isinstance(module, DropoutOrigin):
            dropout = _DropoutNd(module.p, module.inplace)
            setattr(model, name, dropout)
        else:
            replace_batch_norm(module)


def setup_torchbench_cwd():
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(dir_path)
    original_dir = abspath(os.getcwd())

    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    for torchbench_dir in (
        "./torchbenchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
    ):
        if exists(torchbench_dir):
            break

    if exists(torchbench_dir):
        torchbench_dir = abspath(torchbench_dir)
        os.chdir(torchbench_dir)
        sys.path.append(torchbench_dir)

    return original_dir


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


def timed(model, model_iter_fn, example_inputs, times=1):
    synchronize()

    reset_rng_state()
    perfs = []
    for _ in range(times):
        t0 = time.perf_counter()
        _ = model_iter_fn(model, example_inputs, collect_outputs=False)
        synchronize()
        t1 = time.perf_counter()
        perfs.append(t1 - t0)

    return perfs


def get_model(model_name, batch_size=None):
    setup_torchbench_cwd()
    module = importlib.import_module(f"torchbenchmark.models.{model_name}")
    benchmark_cls = getattr(module, "Model", None)
    bm = benchmark_cls(test="train", device="cpu", batch_size=batch_size)
    model, inputs = bm.get_module()

    return model, inputs
