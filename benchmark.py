import argparse
import logging
import os
from datetime import datetime
from test.utils import get_model, timed

import torch
import torch._dynamo as dynamo
from torch.profiler import ProfilerActivity, profile

from pfeife import PipeManager, run_master
from pfeife.loss import SumLoss
from pfeife.utils import get_logger
from pfeife.option import PipeOption
from pfeife.compile import compile_module

log = get_logger()

now = datetime.now()
current_time = now.strftime("%y%m%d-%H%M%S")
dir_path = os.path.dirname(os.path.realpath(__file__))
result_path = f"{dir_path}/result/pipe-test-{current_time}"


def profile_model(model_iter_fn, model, inputs):
    print(f"save to: {result_path}")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=2, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            result_path, worker_name="worker_1"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(5):
            model_iter_fn(model, inputs)
            prof.step()


def run_model(args, model, inputs, option):
    dynamo.reset()

    if not args.no_pipe:
        model = PipeManager(model, loss_fn=SumLoss(), option=option)

        def model_iter_fn(model, example_inputs, collect_outputs=False):
            target = torch.rand(args.batch_size, 10)
            outputs = model.run(target, *example_inputs)

            if collect_outputs:
                return outputs

    else:
        model = model.cuda()

        optim = torch.optim.Adam(model.parameters())
        model = torch.compile(model, backend=option.compiler)

        def model_iter_fn(model, example_inputs, collect_outputs=False):
            inputs = [x.cuda() for x in example_inputs]
            outputs = model(*inputs)
            loss = outputs.sum()
            loss.backward()
            optim.zero_grad()
            optim.step()
            if collect_outputs:
                return outputs

    model.train()

    # warmup
    if args.profile:
        profile_model(model_iter_fn, model, inputs)
    else:
        _ = timed(model, model_iter_fn, inputs, times=3, return_result=False)
        t_total = timed(
            model, model_iter_fn, inputs, times=args.repeat, return_result=False
        )

        print(f"mean latency {t_total / args.repeat} across {args.repeat} runs")


def run_valid(args, model, inputs, option):
    dynamo.reset()

    model.train()

    pipe = PipeManager(model, loss_fn=SumLoss(), option=option)
    target = torch.rand(args.batch_size, 10)
    pipe_outputs = pipe.run(target, *inputs)

    optim = torch.optim.Adam(model.parameters(), **option.optimizer_kwargs)
    optim.zero_grad()
    cpu_outputs = model(*inputs).sum()
    cpu_outputs.backward()
    optim.step()

    print(f"first pipe output (sum): {pipe_outputs}")
    print(f"cpu output (sum): {cpu_outputs}")

    pipe_param, pipe_grad = pipe.rpc_workers[0].rpc_sync().test_param_and_grad()
    cpu_param = list(model.parameters())[0]

    print(f"pipe param[0]: {pipe_param.reshape(-1)[:5]}")
    print(f"cpu param[0]: {cpu_param.reshape(-1)[:5]}")

    print(f"pipe param[0] grad: {pipe_grad.reshape(-1)[:5]}")
    print(f"cpu param[0] grad: {cpu_param.grad.reshape(-1)[:5]}")

    pipe_outputs2 = pipe.run(target, *inputs)
    cpu_outputs2 = model(*inputs).sum()
    print(f"second pipe output (sum): {pipe_outputs2}")
    print(f"second cpu output (sum): {cpu_outputs2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="aot_eager",
        help="if set to a str, uses dynamo[str] backend. else, aot_eager",
    )
    parser.add_argument("--no_pipe", action="store_true")
    parser.add_argument("--scheduler", default="gpipe")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--check_valid", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device_cnt", type=int, default=2)
    parser.add_argument("--pipe_split", type=int, default=2)
    parser.add_argument("--batch_split", type=int, default=4)
    parser.add_argument("--repeat", default=5, type=int, help="Repeats for timing run")
    parser.add_argument(
        "--model",
        default="timm_vision_transformer",
        help="name of torchbench model, e.g. hf_Bert",
    )

    args = parser.parse_args()

    option = PipeOption(
        compiler=args.backend,
        scheduler=args.scheduler,
        device_cnt=args.device_cnt,
        stage_cnt=args.pipe_split,
        batch_cnt=args.batch_split,
        verbosity="debug" if args.verbose else "info",
    )

    model_name = args.model

    print(f"================ run {model_name} ================")

    model, inputs = get_model(args.model, args.batch_size)
    model = model.to("cpu")
    inputs = [i.to("cpu") for i in inputs]

    print(f"input shape: {inputs[0].shape}")
    if args.batch_size is None:
        args.batch_size = inputs[0].shape[0]

    if args.check_valid:
        run_master(run_valid, args=(args, model, inputs, option), option=option)
    elif args.no_pipe:
        run_model(args, model, inputs, option)
    else:
        run_master(run_model, args=(args, model, inputs, option), option=option)
