import argparse
import os
import random
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.backends.cudnn as cudnn
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity, profile

import torch.cuda.nvtx as nvtx

from pfeife.local.manager import PipeManager
from pfeife.loss import SumLoss, MaskedLMOutputLoss, RescaleLoss, LossWrapper
from pfeife.utils import get_logger, fmem
from pfeife.option import PipeOption
from pfeife.batch import split_args_kwargs
from pfeife.utils import tree_map

from test.utils import get_model, timed, replace_batch_norm

log = get_logger()

now = datetime.now()
current_time = now.strftime("%y%m%d-%H%M%S")
dir_path = os.path.dirname(os.path.realpath(__file__))
result_path = f"{dir_path}/result/pipe-test-{current_time}"


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 10)

        self.linear.weight = self.linear3.weight
        self.linear.bias = self.linear3.bias

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
    file_name = f"memory_{current_time}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_name}.json.gz")

    # Construct the memory timeline file.
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    for dev in devices:
        # prof.export_memory_timeline(f"{file_name}_{dev}.html", device=dev)
        prof.export_memory_timeline(f"{file_name}.html")


def profile_model(model, inputs, model_iter_fn, args):
    print(f"save to: {result_path}")
    if args.collect_memory:
        handler = trace_handler
    else:
        handler = torch.profiler.tensorboard_trace_handler(
            result_path, worker_name="worker_1"
        )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=3),
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(args.repeat):
            with record_function(f"## iter {i} ##"):
                model_iter_fn(model, inputs)
            prof.step()


def detach(x):
    if torch.is_tensor(x):
        return x.detach()
    else:
        return x


def prepare_model(args):
    cwd = os.getcwd()

    if args.model == "toy":
        model = TestModel()
        inputs = [torch.rand(args.batch_size * args.batch_cnt, 10)]
        loss_fn = SumLoss()
    elif args.model.startswith("timm:"):
        import timm

        # TODO: should we set the size of inputs instead Imagenet?
        model_name = args.model[5:]
        model = timm.create_model(model_name, pretrained=False)
        inputs = [torch.rand(args.batch_size * args.batch_cnt, 3, 224, 224)]
        loss_fn = SumLoss()
    else:
        model, inputs = get_model(args.model, args.batch_size * args.batch_cnt)
        if args.model.startswith("hf_"):
            loss_fn = MaskedLMOutputLoss()
        elif args.model == "Super_SloMo":

            class SecondLoss(LossWrapper):
                def forward(self, input, target, model=None):
                    return input[1]

            loss_fn = SecondLoss()
        else:
            loss_fn = SumLoss()

    os.chdir(cwd)

    return model, inputs, loss_fn


def run_valid_local(args, option):
    set_seed()

    option.optimizer_kwargs["lr"] = 1e-5

    model, inputs, loss_fn = prepare_model(args)

    replace_batch_norm(model)

    dynamo.reset()

    model.train()
    local_model = deepcopy(model)

    if args.model.startswith("hf_"):
        loss_fn = MaskedLMOutputLoss()
    else:
        loss_fn = SumLoss()

    loss_fn = RescaleLoss(loss_fn, 1 / 1000)

    inputs = tree_map(detach, inputs)
    local_inputs = tree_map(detach, inputs)
    local_inputs, _ = split_args_kwargs(local_inputs, dict(), args.batch_cnt)

    pipe = PipeManager(model, loss_fn=loss_fn, option=option)
    target = torch.rand(args.batch_size, 10)
    pipe_outputs = pipe.run(target, *inputs)

    optim = torch.optim.Adam(local_model.parameters(), **option.optimizer_kwargs)
    optim.zero_grad()

    cpu_outputs = 0
    for v in local_inputs:
        loss = loss_fn(local_model(*v), None)
        cpu_outputs += loss
    cpu_outputs.backward()
    optim.step()

    print(f"first pipe output (sum): {pipe_outputs}")
    print(f"first vanilla output (sum): {cpu_outputs}")
    print(f"diff percentage: {(pipe_outputs - cpu_outputs) / cpu_outputs * 100:.3f}%\n")

    # pipe_param, pipe_grad = pipe.workers[0].test_param_and_grad()
    # cpu_param = list(local_model.parameters())[0]

    # print(f"pipe param[0][:5]: {pipe_param.reshape(-1)[:5].detach().cpu().numpy()}")
    # print(f"vanilla param[0][:5]: {cpu_param.reshape(-1)[:5].detach().numpy()}\n")

    # print(f"pipe param[0] grad[:5]: {pipe_grad.reshape(-1)[:5].detach().cpu().numpy()}")
    # print(
    #     f"vanilla param[0] grad[:5]: {cpu_param.grad.reshape(-1)[:5].detach().numpy()}\n"
    # )

    for n in range(5):
        inputs = tree_map(detach, inputs)
        local_inputs = tree_map(detach, inputs)
        local_inputs, _ = split_args_kwargs(local_inputs, dict(), args.batch_cnt)

        pipe_outputs = pipe.run(target, *inputs)

        cpu_outputs = 0
        optim.zero_grad()
        for v in local_inputs:
            cpu_outputs += loss_fn(local_model(*v), None)
        cpu_outputs.backward()
        optim.step()

        print(f"#{n} next pipe output (sum): {pipe_outputs}")
        print(f"#{n} next vanilla output (sum): {cpu_outputs}")
        print(
            f"#{n} diff percentage: {(pipe_outputs - cpu_outputs) / cpu_outputs * 100:.3f}%"
        )

        # pipe_param, pipe_grad = pipe.workers[0].test_param_and_grad()
        # cpu_param = list(local_model.parameters())[0]

        # print(f"pipe param[0][:5]: {pipe_param.reshape(-1)[:5].detach().cpu().numpy()}")
        # print(f"vanilla param[0][:5]: {cpu_param.reshape(-1)[:5].detach().numpy()}\n")

        # print(
        #     f"pipe param[0] grad[:5]: {pipe_grad.reshape(-1)[:5].detach().cpu().numpy()}"
        # )
        # print(
        #     f"vanilla param[0] grad[:5]: {cpu_param.grad.reshape(-1)[:5].detach().numpy()}\n"
        # )


def run_model(args, option):
    # Disable TF32
    # torch.set_float32_matmul_precision("high")

    model, inputs, loss_fn = prepare_model(args)

    # if torch.is_tensor(inputs[0]):
    #     print(f"input shape: {inputs[0].shape}")

    dynamo.reset()

    if args.no_pipe:
        model = model.cuda() if not args.cpu else model
        optim = torch.optim.Adam(model.parameters())
        if option.compiler is not None:
            model = torch.compile(model, backend=option.compiler)

        def model_iter_fn(model, example_inputs, collect_outputs=False):
            inputs = [x.clone() for x in example_inputs]
            if not args.cpu:
                inputs = [x.cuda() for x in inputs]

            outputs = model(*inputs)
            optim.zero_grad()
            # loss = outputs.sum()
            loss = loss_fn(outputs, None)
            loss.backward()
            optim.step()
            if collect_outputs:
                return outputs

    else:
        model = PipeManager(model, loss_fn=loss_fn, option=option)

        def model_iter_fn(model, example_inputs, collect_outputs=False):
            target = torch.rand(args.batch_size, 10)
            inputs = [x.clone() for x in example_inputs]
            outputs = model.run(target, *inputs)

            if collect_outputs:
                return outputs

    model.train()

    if args.profile:
        profile_model(model, inputs, model_iter_fn, args)
    else:
        _run_bench(args, option, model, inputs, model_iter_fn)


def _run_bench(args, option, model, inputs, model_iter_fn):
    # warmup
    _ = timed(model, model_iter_fn, inputs, times=3)

    if isinstance(model, PipeManager):
        model.clear_record()
        model.init_runner()

    alloc = []
    if not args.cpu:
        for device in option.device_map:
            # time.sleep(2)  # sleep 2 seconds to avoid the effect of previous run
            torch.cuda.synchronize(device)
            alloc.append(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.cudart().cudaProfilerStart()

    times = timed(model, model_iter_fn, inputs, times=args.repeat)
    t_total = sum(times)
    t_std = np.std(times)

    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"mean latency {t_total / args.repeat} / std.dev. {t_std} across {args.repeat} runs"
    )

    if isinstance(model, PipeManager):
        model.print_record()

    if not args.cpu:
        for i, device in enumerate(option.device_map):
            torch.cuda.synchronize(device)

            dev_max = torch.cuda.max_memory_allocated(device)
            dev_reserved = torch.cuda.max_memory_reserved(device)

            print(
                f"[Device {device}] idle alloc: {fmem(alloc[i])}, max alloc: {fmem(dev_max)}, alloc diff: {fmem(dev_max - alloc[i])}, reserved: {fmem(dev_reserved)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_pipe", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check_valid", action="store_true")
    parser.add_argument("--repeat", default=30, type=int, help="Repeats for timing run")
    parser.add_argument(
        "--model",
        default="toy",
        help="name of torchbench model, e.g. hf_Bert",
    )
    parser.add_argument("--collect_memory", action="store_true")

    PipeOption.add_arguments(parser)
    args = parser.parse_args()

    if args.device_bench:
        args.device_bench = os.path.join(os.path.dirname(__file__), args.device_bench)

    option = PipeOption.from_args(args)

    model_name = args.model
    print(f"================ run {model_name} ================")

    if args.collect_memory and not args.profile:
        torch.cuda.memory._record_memory_history()

    if args.check_valid:
        run_valid_local(args, option)
    else:
        run_model(args, option)

    if args.collect_memory and not args.profile:
        torch.cuda.memory._dump_snapshot(f"memory_{current_time}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"memory snapshot saved to memory_{current_time}.pickle")
