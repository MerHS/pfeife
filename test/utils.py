import importlib
import os
import random
import sys
import time
from os.path import abspath, exists

import numpy as np
import torch


def setup_torchbench_cwd():
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
    torch.cuda.synchronize()


def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


def timed(model, model_iter_fn, example_inputs, times=1, return_result=False):
    synchronize()

    reset_rng_state()
    t0 = time.perf_counter()
    # Dont collect outputs to correctly measure timing
    for _ in range(times):
        result = model_iter_fn(model, example_inputs, collect_outputs=False)
        synchronize()
    t1 = time.perf_counter()
    return (t1 - t0, result) if return_result else t1 - t0


def get_model(model_name, batch_size=None):
    setup_torchbench_cwd()
    module = importlib.import_module(f"torchbenchmark.models.{model_name}")
    benchmark_cls = getattr(module, "Model", None)
    bm = benchmark_cls(test="train", device="cpu", jit=False, batch_size=batch_size)
    model, inputs = bm.get_module()

    return model, inputs
