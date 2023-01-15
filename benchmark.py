import argparse
import logging
import os
from functools import partial

import torch
import torch._dynamo as dynamo

from pfeife import run_master, PipeManager
from pfeife.utils import get_logger
from test.utils import get_model, timed

log = get_logger()


class DummyOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass


def run_model(args, model, inputs):
    # def move_tensor(maybe_tensor):
    #     if torch.is_tensor(maybe_tensor):
    #         return maybe_tensor.to(dev_rank)
    #     return maybe_tensor
    # inputs = pytree.tree_map(move_tensor, inputs)

    dynamo.reset()
    if args.verbose:
        dynamo.config.verbose = True
        dynamo.config.log_level = logging.DEBUG
        log.setLevel(logging.INFO)

    backend = args.backend
    if args.backend == "print":

        def print_compile(gm, ex):
            print(
                f"print_compile:\n{str(gm.graph)}\n-----------------------------------------"
            )
            return gm

        backend = print_compile

    def model_iter_fn(model, example_inputs, collect_outputs=False):
        target = torch.rand(args.batch_size, 10)
        outputs = model.run(target, *example_inputs)

        if collect_outputs:
            return outputs

    def loss_fn(pred, target):
        return pred.sum()

    pipe = PipeManager(
        model,
        loss_fn=loss_fn,
        dynamo_backend=backend,
        pipe_split=args.pipe_split,
        batch_split=args.batch_split,
    )
    pipe.train()

    # warmup
    _ = timed(pipe, model_iter_fn, inputs, times=3, return_result=False)
    t_total = timed(pipe, model_iter_fn, inputs, times=args.repeat, return_result=False)

    print(f"mean latency {t_total / args.repeat} across {args.repeat} runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="eager",
        help="if set to a str, uses dynamo[str] backend. else, eager",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--pipe_split", type=int, default=2)
    parser.add_argument("--batch_split", type=int, default=4)
    parser.add_argument("--repeat", default=5, type=int, help="Repeats for timing run")
    parser.add_argument(
        "--model",
        default="timm_vision_transformer",
        help="name of torchbench model, e.g. hf_Bert",
    )

    args = parser.parse_args()

    model_name = args.model

    print(f"================ run {model_name} ================")

    model, inputs = get_model(args.model, args.batch_size)
    print(f"input shape: {inputs[0].shape}")
    args.batch_size = inputs[0].shape[0]

    run_master(run_model, (args, model, inputs), pipe_split=args.pipe_split)
