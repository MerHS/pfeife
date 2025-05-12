import argparse
import os
import random
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity, profile


import torch.cuda.nvtx as nvtx

from pfeife.mp import initialize_pfeife, get_state
from pfeife.mp.partial.module import PartialRunner, PartialModule, partial_compiler
from pfeife.loss import SumLoss, MaskedLMOutputLoss, RescaleLoss, LossWrapper
from pfeife.utils import get_logger, fmem
from pfeife.option import PipeOption
from pfeife.batch import split_args_kwargs
from pfeife.utils import tree_map, set_seed, to_device

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
    handler = torch.profiler.tensorboard_trace_handler(result_path)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=2, active=3),
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(args.repeat):
            with record_function(f"## iter {i} ##"):
                model_iter_fn(model, inputs)
            prof.step()

    # if args.collect_memory:
    #     devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    #     for dev in devices:
    #         prof.export_memory_timeline(f"memory_{current_time}_{dev}.html", device=dev)


def detach(x):
    if torch.is_tensor(x):
        return x.detach()
    else:
        return x


class SecondLoss(LossWrapper):
    def forward(self, input, target, model=None):
        return input[1]


class FastNLPBertLoss(LossWrapper):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target, model=None):
        return self.loss_fn(input["pred_start"], input["pred_end"])


class ContrastiveLossWithTemperature(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLossWithTemperature, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        # Ensure batch sizes are equal
        assert image_embeddings.size(0) == text_embeddings.size(
            0
        ), "Batch sizes of image and text embeddings should be the same"

        # Compute the similarity between image and text embeddings
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature

        # Compute the labels for the positive pairs
        labels = torch.arange(logits.size(0)).to(image_embeddings.device)

        # Compute the contrastive loss
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2


class ClipLoss(LossWrapper):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target, model=None):
        image_embedding = input.image_embeds
        text_embedding = input.text_embeds

        loss = self.loss_fn(image_embedding, text_embedding)
        return loss


LOSS_DICT = {"Super_SloMo": SecondLoss}


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
        model_name = args.model.strip()
        if model_name == "fastNLP_Bert":
            from fastNLP.core.losses import CMRC2018Loss

            loss_fn = FastNLPBertLoss(CMRC2018Loss())
        elif model_name == "hf_clip":
            loss_fn = ClipLoss(ContrastiveLossWithTemperature())
        elif model_name in LOSS_DICT:
            loss_fn = LOSS_DICT[model_name]()
        elif model_name.startswith("hf_"):
            loss_fn = MaskedLMOutputLoss()
        else:
            loss_fn = SumLoss()

    os.chdir(cwd)

    return model, inputs, loss_fn


def run_valid_local(args, option):
    set_seed(option.bench_seed)

    option.optimizer_kwargs["lr"] = 1e-5

    model, inputs, loss_fn = prepare_model(args)

    if type(inputs) is list or type(inputs) is tuple:
        vargs = inputs
        kwargs = dict()
    else:
        vargs = tuple()
        kwargs = inputs

    replace_batch_norm(model)

    dynamo.reset()

    model.train()
    local_model = deepcopy(model)

    loss_fn = RescaleLoss(loss_fn, 1 / 1000)
    # loss_fn = NormalizeLoss(loss_fn)

    vargs = tree_map(detach, vargs)
    kwargs = tree_map(detach, kwargs)
    local_vargs = tree_map(detach, vargs)
    local_kwargs = tree_map(detach, kwargs)
    local_vargs, local_kwargs = split_args_kwargs(
        local_vargs, local_kwargs, args.batch_cnt
    )

    net_state = get_state()
    # model.to(net_state.device)
    model = torch.compile(model, backend=partial_compiler)
    runner = PartialRunner(
        option, torch.optim.Adam(model.parameters(), **option.optimizer_kwargs)
    )
    target = torch.rand(args.batch_size, 10)

    def exec_fn(*args, **kwargs):
        output = model(*args, **kwargs)
        loss = loss_fn(output, target)
        loss.backward()
        return loss

    runner.set_exec_fn(exec_fn)
    pipe_outputs = sum(runner.step(*vargs, **kwargs))

    optim = torch.optim.Adam(local_model.parameters(), **option.optimizer_kwargs)

    cpu_outputs = 0
    optim.zero_grad()
    losses = []
    for va, kw in zip(local_vargs, local_kwargs):
        output = local_model(*va, **kw)
        loss = loss_fn(output, target)
        losses.append(output)
        print(f"local loss: {loss.item()}")
        cpu_outputs += loss
    cpu_outputs.backward()

    # param_sum = 0
    # grad_sum = 0
    # for param in local_model.parameters():
    #     param_sum += param.data.sum().item()
    #     grad_sum += param.grad.sum().item()
    # print(f"local param_sum: {param_sum}, grad_sum: {grad_sum}")

    optim.step()

    print(f"first pipe output (sum): {pipe_outputs}")
    print(f"first vanilla output (sum): {cpu_outputs}")
    print(f"diff percentage: {(pipe_outputs - cpu_outputs) / cpu_outputs * 100:.3f}%\n")

    # runner.gather_params()
    # pipe_params = list(model.parameters())
    # local_params = list(local_model.parameters())
    # pipe_params = [
    #     pipe_params[0].detach().cpu().numpy(),
    #     pipe_params[-1].detach().cpu().numpy(),
    # ]
    # local_params = [local_params[0].detach().numpy(), local_params[-1].detach().numpy()]

    # print(f"pipe param[0][:5]: {pipe_params[0].reshape(-1)[:5]}")
    # print(f"vanilla param[0][:5]: {local_params[0].reshape(-1)[:5]}")
    # print(f"pipe param[-1][:5]: {pipe_params[1].reshape(-1)[:5]}")
    # print(f"vanilla param[-1][:5]: {local_params[1].reshape(-1)[:5]}")

    for n in range(5):
        vargs = tree_map(detach, vargs)
        kwargs = tree_map(detach, kwargs)
        local_vargs = tree_map(detach, vargs)
        local_kwargs = tree_map(detach, kwargs)
        local_vargs, local_kwargs = split_args_kwargs(
            local_vargs, local_kwargs, args.batch_cnt
        )

        pipe_outputs = sum(runner.step(*vargs, **kwargs))

        cpu_outputs = 0
        optim.zero_grad()
        losses = []
        for va, kw in zip(local_vargs, local_kwargs):
            output = local_model(*va, **kw)
            loss = loss_fn(output, target)
            losses.append(output)
            print(f"local loss: {loss.item()}")
            cpu_outputs += loss
        cpu_outputs.backward()

        # param_sum = 0
        # grad_sum = 0
        # for param in local_model.parameters():
        #     param_sum += param.data.sum().item()
        #     grad_sum += param.grad.sum().item()
        # print(f"local param_sum: {param_sum}, grad_sum: {grad_sum}")

        optim.step()

        print(f"#{n} next pipe output (sum): {pipe_outputs}")
        print(f"#{n} next vanilla output (sum): {cpu_outputs}")
        print(
            f"#{n} diff percentage: {(pipe_outputs - cpu_outputs) / cpu_outputs * 100:.3f}%"
        )

        # runner.gather_params()
        # pipe_params = list(model.parameters())
        # local_params = list(local_model.parameters())
        # pipe_params = [
        #     pipe_params[0].detach().cpu().numpy(),
        #     pipe_params[-1].detach().cpu().numpy(),
        # ]
        # local_params = [
        #     local_params[0].detach().numpy(),
        #     local_params[-1].detach().numpy(),
        # ]

        # print(f"pipe param[0][:5]: {pipe_params[0].reshape(-1)[:5]}")
        # print(f"vanilla param[0][:5]: {local_params[0].reshape(-1)[:5]}")
        # print(f"pipe param[-1][:5]: {pipe_params[1].reshape(-1)[:5]}")
        # print(f"vanilla param[-1][:5]: {local_params[1].reshape(-1)[:5]}")


def run_model(args, option):
    # Disable TF32
    # torch.set_float32_matmul_precision("high")

    model, inputs, loss_fn = prepare_model(args)

    # if torch.is_tensor(inputs[0]):
    #     print(f"input shape: {inputs[0].shape}")

    dynamo.reset()

    optimizer = torch.optim.Adam(model.parameters(), **option.optimizer_kwargs)
    # optimizer = None
    runner = PartialRunner(option, optimizer)
    model = torch.compile(model, backend=partial_compiler)
    scaler = torch.GradScaler(enabled=option.mixed_precision is not None)

    def model_iter_fn(model, example_inputs, collect_outputs=False):
        inputs = example_inputs
        if type(inputs) is list or type(inputs) is tuple:
            vargs = inputs
            kwargs = dict()
        else:
            vargs = tuple()
            kwargs = inputs

        def exec_fn(*vargs, **kwargs):
            with torch.autocast(
                dtype=option.mixed_precision,
                device_type="cuda",
                enabled=option.mixed_precision is not None,
            ):
                target = torch.rand(args.batch_size, 10)
                output = model(*vargs, **kwargs)
                loss = loss_fn(output, target)

            scaler.scale(loss).backward()
            return loss

        runner.set_exec_fn(exec_fn)
        runner.step(*vargs, **kwargs)

    model.train()

    if args.profile:
        profile_model(model, inputs, model_iter_fn, args)
    else:
        _run_bench(args, model, inputs, model_iter_fn)


def _run_bench(args, model, inputs, model_iter_fn):
    # warmup
    _ = timed(model, model_iter_fn, inputs, times=3)

    # if isinstance(model, PipeManager):
    #     model.clear_record()

    # alloc = []
    # if not args.cpu:
    #     for device in option.device_map:
    #         # time.sleep(2)  # sleep 2 seconds to avoid the effect of previous run
    #         torch.cuda.synchronize(device)
    #         alloc.append(torch.cuda.memory_allocated(device))
    #         torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.cudart().cudaProfilerStart()

    times = timed(model, model_iter_fn, inputs, times=args.repeat)
    t_total = sum(times)
    t_std = np.std(times)

    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"mean latency {t_total / args.repeat} / std.dev. {t_std} across {args.repeat} runs"
    )

    # if isinstance(model, PipeManager):
    #     model.print_record()

    # if not args.cpu:
    #     for i, device in enumerate(option.device_map):
    #         torch.cuda.synchronize(device)

    #         dev_max = torch.cuda.max_memory_allocated(device)
    #         dev_reserved = torch.cuda.max_memory_reserved(device)

    #         print(
    #             f"[Device {device}] idle alloc: {fmem(alloc[i])}, max alloc: {fmem(dev_max)}, alloc diff: {fmem(dev_max - alloc[i])}, reserved: {fmem(dev_reserved)}"
    #         )


if __name__ == "__main__":
    initialize_pfeife()
    parser = argparse.ArgumentParser()

    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check_valid", action="store_true")
    parser.add_argument("--repeat", default=30, type=int, help="Repeats for timing run")
    parser.add_argument(
        "--model",
        default="toy",
        help="name of torchbench model, e.g. hf_Bert",
    )

    PipeOption.add_arguments(parser)
    args = parser.parse_args()

    if args.device_bench:
        args.device_bench = os.path.join(os.path.dirname(__file__), args.device_bench)

    option = PipeOption.from_args(args)
    if option.bench_seed == 0:
        option.bench_seed = 42

    model_name = args.model
    print(f"================ run {model_name} ================")

    if args.check_valid:
        run_valid_local(args, option)
    else:
        run_model(args, option)
