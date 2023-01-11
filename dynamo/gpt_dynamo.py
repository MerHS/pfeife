import argparse
import logging
import os
from functools import partial

import random
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch._dynamo as dynamo
import torch.utils._pytree as pytree
from torch.distributed.pipeline.sync import Pipe

from common import timed
from dist_util import model_iter_fn, get_model

import os
log = logging.getLogger(__name__)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

class PipeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod1 = nn.Linear(50, 100).to('cuda:0')
        self.mod2 = nn.Linear(100, 50).to('cuda:1')
    
    def forward(self, x):
        x = self.mod1(x)
        x = x.to('cuda:1')
        x = self.mod2(x)
        return x

class PipeChunk(nn.Module):
    def __init__(self):
        super().__init__()
        mod1 = nn.Linear(50, 100).to('cuda:0')
        mod2 = nn.Linear(100, 50).to('cuda:1')
        seq = nn.Sequential(mod1, mod2)
        self.seq = Pipe(seq, chunks=2)
    
    def forward(self, x):
        x = self.seq(x)
        x = x.local_value()
        return x

def run_model(args, model, inputs, key):
    rank = int(os.getenv("RANK", 0))
    
    if args.device == "cuda":
        # needed for FSDP
        torch.cuda.set_device(rank)

    dev_rank = f"{args.device}:{rank}"
    if not args.torchbench_model.startswith('pipe'):
        model = model.to(dev_rank)

    def move_tensor(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(dev_rank)
        return maybe_tensor

    inputs = pytree.tree_map(move_tensor, inputs)

    if args.dynamo:
        dynamo.reset()
        if args.verbose:
            dynamo.config.verbose = True
            dynamo.config.log_level = logging.INFO

        backend = args.dynamo

        def print_compile(gm, ex):
            print(
                f"print_compile:\n{str(gm.graph)}\n-----------------------------------------"
            )
            return gm

        dynamo_ctx = dynamo.optimize(
            print_compile if args.dynamo == "print" else backend
        )
        model = dynamo_ctx(model)

    # warmup
    _ = timed(model, model_iter_fn, inputs, times=3, return_result=False)
    t_total = timed(
        model, model_iter_fn, inputs, times=args.repeat, return_result=False
    )

    return t_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dynamo",
        default=None,
        help="if set to a str, uses dynamo[str] backend. else, eager",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--trace_file", default="profile.json", help="Run the profiler")
    parser.add_argument("--repeat", default=10, type=int, help="Repeats for timing run")

    model_arg = parser.add_mutually_exclusive_group(required=True)
    model_arg.add_argument(
        "--torchbench_model", help="name of torchbench model, e.g. hf_Bert"
    )
    model_arg.add_argument(
        "--toy_model", action="store_true", help="use toy model instead"
    )
    args = parser.parse_args()

    model_name = args.torchbench_model
    if args.torchbench_model == 'pipe':
        model = PipeModule()
        inputs = (torch.rand(20, 50), )
    elif args.torchbench_model == 'pipechunk':
        model = PipeChunk()
        inputs = (torch.rand(20, 50), )
    else:
        if args.toy_model:
            model_name = "ToyModel"
        model, inputs = get_model(args)

    fn = partial(run_model, args, model, inputs)

    world_size = os.getenv("WORLD_SIZE", 1)
    t_total = fn(f"{model_name}_{world_size}")
    print(f"mean latency {t_total / args.repeat} across {args.repeat} runs")
