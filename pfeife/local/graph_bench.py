import os
import time
from contextlib import nullcontext
from typing import List

import torch
import torch.fx as fx
from torch.amp import GradScaler

from ..utils import tree_map, tree_filter_tensor, fmem, module_set_attr
from ..option import PipeOption, get_optimizer_cls
from ..graph.computation_graph import (
    CompGraph,
    GraphCPUFetcher,
    GraphFetcher,
    GraphSingleDeviceFetcher,
    MemoryLocalDeviceLogger,
    TimingLocalDeviceChecker,
    MemoryLogger,
    TimingChecker,
)

TIME_WARM_ITER = 2
TIME_CHECK_ITER = 5


def measure_optimizer(
    gm: fx.GraphModule,
    graph: CompGraph,
    devices: List[torch.device],
    option: PipeOption,
):
    params = list(gm.parameters())
    mem_caps = [torch.cuda.get_device_properties(d).total_memory for d in devices]

    # move all params to cpu first
    device_map = dict()
    for p in params:
        device_map[p] = p.device
        p.data = p.data.cpu()
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad = p.grad.cpu()

    # move parameters until each GPU reaches 50% of memory
    params_list = []
    curr_params = []
    params_list.append(curr_params)
    curr_device = devices[0]
    device_idx = 0

    param_len = len(params)
    for pidx, p in enumerate(params):
        if p.device != torch.device("cpu"):
            continue
        p_size = p.data.numel() * p.data.element_size()
        if p_size * 2 + torch.cuda.memory_allocated(curr_device) >= 0.3 * mem_caps[0]:
            assert (
                len(curr_params) > 0
            ), f"single param size is too big. (reaches > 25%): {p.data.shape}"
            print(
                f"optim moved {pidx} / {param_len} - mem {fmem(p_size)} + {fmem(torch.cuda.memory_allocated(curr_device))} -> {fmem(mem_caps[0])}"
            )
            device_idx += 1
            curr_device = devices[device_idx]
            curr_params = []
            params_list.append(curr_params)

        curr_params.append(p)
        p.data = p.data.to(device=curr_device)
        p.grad = p.grad.to(device=curr_device)

    print(f"total params: {len(params)}, moved params: {[len(p) for p in params_list]}")
    # make optimizer
    optim_cls = get_optimizer_cls(option.optimizer_type)
    optims = [optim_cls(p, lr=0.0) for p in params_list]

    # measure optimizer runtime
    optim_runtimes = [0 for _ in range(len(optims))]
    optim_events = [
        [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(5)
        ]
        for _ in range(len(optims))
    ]
    # warmup
    for _ in range(3):
        for idx, optim in enumerate(optims):
            device = devices[idx]
            optim.step()

    for iter_idx in range(5):
        for idx, optim in enumerate(optims):
            device = devices[idx]
            stream = torch.cuda.current_stream(device)
            start_ev, end_ev = optim_events[idx][iter_idx]

            start_ev.record(stream)
            optim.step()
            end_ev.record(stream)
            optim_events

    for device in devices:
        torch.cuda.synchronize(device)
    for idx, events in enumerate(optim_events):
        for start_ev, end_ev in events:
            optim_runtimes[idx] += start_ev.elapsed_time(end_ev)

    optim_time = sum([r / 5 for r in optim_runtimes])
    total_mems = sum([cnode.weight for cnode in graph.nodes])

    for cnode in graph.nodes:
        cnode.optim_time = optim_time * cnode.weight / total_mems

    print(f"optim time: {optim_time:.2f} ms")

    # restore devices
    del optims
    for p in params:
        p.data = p.data.to(device=device_map[p])
        p.grad = p.grad.to(device=device_map[p])


def gen_comp_graph_cpu(gm: fx.GraphModule, args: List[torch.Tensor]):
    graph = CompGraph()
    fetcher = GraphCPUFetcher(gm, graph)

    # TODO: is it safe?
    def dummy_inputs(v):
        if torch.is_tensor(v):
            return v.clone()
        else:
            return v

    inputs = tree_map(dummy_inputs, args)
    output = fetcher.run(*inputs)

    graph.build_edges(fetcher.param_rev)

    for cnode in graph.nodes:
        cnode.comp_time = 1.0

    return graph, output


def restore_graph(gm, graph):
    # restore original target
    moved = graph.param_set.moved_targets
    for node in gm.graph.nodes:
        if node.target in moved:
            module_set_attr(gm, node.target, None)
            node.target = moved[node.target]
    gm.recompile()

    for i, cnode in enumerate(graph.nodes):
        modules = list(cnode.call_modules)
        for call_m in modules:
            if call_m in moved:
                cnode.call_modules.remove(call_m)
                cnode.call_modules.add(moved[call_m])


def gen_amp_context(option):
    if option.mixed_precision:
        context = torch.autocast(dtype=option.mixed_precision, device_type="cuda")
    else:
        context = nullcontext()
    return context


def gen_comp_graph(
    gm: fx.GraphModule,
    args: List[torch.Tensor],
    devices: List[str | int],
    option: PipeOption,
):
    graph_bench_path = option.graph_bench
    devices = [torch.device(d) for d in devices]
    graph = CompGraph()
    fetcher = GraphFetcher(gm, graph, devices)
    scaler = GradScaler(enabled=option.mixed_precision is not None)

    first_dev = devices[0]

    # TODO: is it safe?
    def dummy_inputs(v):
        if torch.is_tensor(v):
            return v.to(device=first_dev).detach()
        else:
            return v

    with gen_amp_context(option):
        inputs = tree_map(dummy_inputs, args)

        # Iteration 1: Draw graph and populate gradients and temporaries
        output = fetcher.run(*inputs)

        graph.build_edges()

        if graph_bench_path is not None:
            path = os.path.abspath(graph_bench_path)
            if os.path.exists(path):
                graph.load_graph_data(path)
                restore_graph(gm, graph)
                return graph, output

        gm.recompile()
    del output
    del fetcher

    # TODO: remove this
    # graph.print_branches()
    # for i, p in enumerate(graph.nodes):
    #     print(f"{i}: {'/'.join([n.op + '|' + str(n)[:10] for n in p.nodes])}")
    # gm.print_readable()

    # Iteration 2: Fetch memory usage
    last_idx = len(graph.nodes)
    last_device = graph.nodes[-1].device
    mem_logger = MemoryLogger(gm, graph)

    with gen_amp_context(option):
        output = mem_logger.run(*inputs)

        mem_logger.fwd_dict[last_idx, last_device] = mem_logger.check_mem(last_device)

        for cnode_idx, cnode in enumerate(graph.nodes):
            fwd_before = mem_logger.fwd_dict[cnode_idx, cnode.device]
            fwd_after = mem_logger.fwd_dict[cnode_idx + 1, cnode.device]

            cnode.act_mem = fwd_after[0] - fwd_before[0]
            cnode.temp_mem = fwd_after[1] - fwd_before[0]

        args_order = mem_logger.placeholder_order
        del output
        del mem_logger

        graph.traverse_linearize(option.linearizer)
        if option.linearizer and option.linearizer != "none":
            graph.redraw_graph(gm)

    # Iteration 3: Fetch timing
    timing = TimingChecker(gm, graph, args_order)

    # 3 times mean
    devices_set = set([cnode.device for cnode in graph.nodes])

    for cnode in graph.nodes:
        cnode.comp_time = 0

    for device in devices_set:
        torch.cuda.synchronize(device)

    for _ in range(TIME_WARM_ITER):
        with gen_amp_context(option):
            output = timing.run_timing(*inputs)
            loss = 0
            for out in output:
                # sum only one leaf tensors
                if (
                    torch.is_tensor(out)
                    and out.device == last_device
                    and out.requires_grad
                    and out.grad_fn is not None
                ):
                    loss += out.sum()
                    break
        if torch.is_tensor(loss):
            scaler.scale(loss).backward()
            # loss.backward()

    timing.events = []

    # for device in devices:
    #     torch.cuda.synchronize(device)

    for _ in range(TIME_CHECK_ITER):
        with gen_amp_context(option):
            output = timing.run_timing(*inputs)
            loss = 0
            for out in output:
                # sum only one leaf tensors
                if (
                    torch.is_tensor(out)
                    and out.device == last_device
                    and out.requires_grad
                    and out.grad_fn is not None
                ):
                    loss += out.sum()
                    break
        if torch.is_tensor(loss):
            scaler.scale(loss).backward()

    for device in devices_set:
        torch.cuda.synchronize(device)

    for start_event, end_event in timing.events:
        for cnode_idx, cnode in enumerate(graph.nodes):
            s_event = start_event[cnode_idx][0]
            e_event = end_event[cnode_idx]
            cnode.comp_time += s_event.elapsed_time(e_event)

    cumul_forward_time = 0
    for cnode in graph.nodes:
        cnode.comp_time = cnode.comp_time / TIME_CHECK_ITER
        cumul_forward_time += cnode.comp_time

    # estimate rough ratio of backward / forward times
    def run_fb():
        for device in devices_set:
            torch.cuda.synchronize(device)

        with gen_amp_context(option):
            output = timing.run_timing(*inputs, no_event=True)

            for device in devices:
                torch.cuda.synchronize(device)

            mid = time.perf_counter()

            loss = 0
            for out in output:
                # sum only one leaf tensors
                if (
                    torch.is_tensor(out)
                    and out.device == last_device
                    and out.requires_grad
                    and out.grad_fn is not None
                ):
                    loss += out.sum()
                    break

        if torch.is_tensor(loss):
            scaler.scale(loss).backward()

        for device in devices_set:
            torch.cuda.synchronize(device)

        end = time.perf_counter()
        start = timing.start_time
        return start, mid, end

    for _ in range(2):
        run_fb()

    fwd, bwd = 0, 0
    for _ in range(3):
        start, mid, end = run_fb()
        fwd += mid - start
        bwd += end - mid

    forward_time = fwd / 3 * 1000
    back_time = bwd / 3 * 1000
    print(f"forward: {forward_time:.4f} ms, backward: {back_time:.4f} ms")

    graph.fb_ratio = back_time / forward_time

    if graph.fb_ratio < 1:
        graph.fb_ratio = 1.0

    del loss

    # Iteration 4 (Optional): measure optimizer step time
    if not option.no_bench_optimizer:
        measure_optimizer(gm, graph, devices, option)

    # print each fx node and comp time
    # gm.graph.print_tabular()
    # for i, cnode in enumerate(graph.nodes):
    #     print(f"{i} - {cnode.device} - {cnode.comp_time:.4f}")
    #     for node in cnode.nodes:
    #         print(f"  - {node.op} {node.name} {node.target}")

    with gen_amp_context(option):
        if graph_bench_path is not None:
            graph.save_graph_data(graph_bench_path)

        restore_graph(gm, graph)

        output = timing.run_timing(*inputs)

    # clear gradients
    for param in gm.parameters():
        param.grad = None

    return graph, output


def gen_local_comp_graph(
    gm: fx.GraphModule,
    args: List[torch.Tensor],
    device: str | int,
    option: PipeOption,
):
    graph_bench_path = option.graph_bench
    graph = CompGraph()
    fetcher = GraphSingleDeviceFetcher(gm, graph, device)
    scaler = GradScaler(enabled=option.mixed_precision is not None)

    def dummy_inputs(v):
        if torch.is_tensor(v):
            return v.to(device=device).detach()
        else:
            return v

    with gen_amp_context(option):
        inputs = tree_map(dummy_inputs, args)

        # Iteration 1: Draw graph and populate gradients and temporaries
        output = fetcher.run(*inputs)

        graph.build_edges()

        if graph_bench_path is not None:
            path = os.path.abspath(graph_bench_path)
            if os.path.exists(path):
                graph.load_graph_data(path)
                restore_graph(gm, graph)
                return graph, output

        gm.recompile()
    del output
    del fetcher

    # Iteration 2: Fetch memory usage
    mem_logger = MemoryLocalDeviceLogger(gm, graph)

    with gen_amp_context(option):
        output = mem_logger.run(*inputs)

        args_order = mem_logger.placeholder_order
        del output
        del mem_logger

        graph.traverse_linearize(option.linearizer)
        if option.linearizer and option.linearizer != "none":
            graph.redraw_graph(gm)

    # Iteration 3: Fetch timing
    # 3 times mean
    device_cnodes = [[]]
    last_dev_id = graph.nodes[0].device_id

    for cnode in graph.nodes:
        cnode.comp_time = 0
        if cnode.device_id != last_dev_id:
            device_cnodes.append([])
            last_dev_id = cnode.device_id
        device_cnodes[-1].append(cnode)

    torch.cuda.synchronize(device)
    total_fwd_time = 0
    total_bwd_time = 0
    envs = None
    exec_input_attr = set()

    def detach_val(t):
        if torch.is_tensor(t):
            return t.detach().requires_grad_(t.requires_grad)
        else:
            return t

    for graph_idx, cnodes in enumerate(device_cnodes):
        timing = TimingLocalDeviceChecker(gm, graph, cnodes, args_order)
        timing.move_cnode_devices()

        is_last = graph_idx == len(device_cnodes) - 1
        output = None
        loss = None
        initial_env = envs

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        for _ in range(TIME_WARM_ITER):
            if initial_env is not None:
                envs = tree_map(detach_val, initial_env)

            with gen_amp_context(option):
                if initial_env is None:
                    output = timing.run_timing(
                        *inputs,
                        exec_input_attr=exec_input_attr.copy(),
                        output_processing=is_last,
                    )
                else:
                    output = timing.run_timing(
                        *inputs,
                        exec_input_attr=exec_input_attr.copy(),
                        initial_env=envs,
                        output_processing=is_last,
                    )

                loss = 0
                for out in tree_filter_tensor(output):
                    if out.requires_grad and out.grad_fn is not None:
                        loss += out.sum()

            if torch.is_tensor(loss):
                scaler.scale(loss).backward()

        timing.events = []

        for _ in range(TIME_CHECK_ITER):
            if initial_env is not None:
                envs = tree_map(detach_val, initial_env)

            with gen_amp_context(option):
                if initial_env is None:
                    output = timing.run_timing(
                        *inputs,
                        exec_input_attr=exec_input_attr.copy(),
                        output_processing=is_last,
                    )
                else:
                    output = timing.run_timing(
                        *inputs,
                        exec_input_attr=exec_input_attr.copy(),
                        initial_env=envs,
                        output_processing=is_last,
                    )

                loss = 0
                for out in tree_filter_tensor(output):
                    if out.requires_grad and out.grad_fn is not None:
                        loss += out.sum()

            if torch.is_tensor(loss):
                scaler.scale(loss).backward()

        torch.cuda.synchronize(device)

        for start_event, end_event in timing.events:
            for cnode_idx, cnode in enumerate(cnodes):
                s_event = start_event[cnode_idx][0]
                e_event = end_event[cnode_idx]
                cnode.comp_time += s_event.elapsed_time(e_event)

        cumul_forward_time = 0
        for cnode in cnodes:
            cnode.comp_time = cnode.comp_time / TIME_CHECK_ITER
            cumul_forward_time += cnode.comp_time

        def run_fb(initial_env):
            torch.cuda.synchronize(device)

            with gen_amp_context(option):
                if initial_env is not None:
                    envs = tree_map(detach_val, initial_env)

                if initial_env is None:
                    output = timing.run_timing(
                        *inputs,
                        exec_input_attr=exec_input_attr.copy(),
                        output_processing=is_last,
                        no_event=True,
                    )
                else:
                    output = timing.run_timing(
                        *inputs,
                        exec_input_attr=exec_input_attr.copy(),
                        initial_env=envs,
                        output_processing=is_last,
                        no_event=True,
                    )

                torch.cuda.synchronize(device)

                mid = time.perf_counter()

                loss = 0
                for out in output:
                    for out in tree_filter_tensor(output):
                        if out.requires_grad and out.grad_fn is not None:
                            loss += out.sum()

            if torch.is_tensor(loss):
                scaler.scale(loss).backward()

            torch.cuda.synchronize(device)

            end = time.perf_counter()
            start = timing.start_time
            return start, mid, end

        for _ in range(2):
            run_fb(initial_env)

        fwd, bwd = 0, 0
        for _ in range(3):
            start, mid, end = run_fb(initial_env)
            fwd += mid - start
            bwd += end - mid

        total_fwd_time += fwd / 3 * 1000
        total_bwd_time += bwd / 3 * 1000

        envs = tree_map(detach_val, output)
        exec_input_attr = timing.exec_input_attr

    print(
        f"total forward: {total_fwd_time:.4f} ms, total backward: {total_bwd_time:.4f} ms"
    )

    graph.fb_ratio = total_bwd_time / total_fwd_time

    if graph.fb_ratio < 1:
        graph.fb_ratio = 1.0

    del loss

    with gen_amp_context(option):
        if graph_bench_path is not None:
            graph.save_graph_data(graph_bench_path)

        restore_graph(gm, graph)

    output = envs

    # clear gradients
    for param in gm.parameters():
        param.grad = None

    return graph, output
