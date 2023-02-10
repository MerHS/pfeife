import os

import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from .option import PipeOption
from .utils import set_logger_level


def get_rpc_name(rank):
    return "master" if rank == 0 else f"worker_{rank}"


def run_master(main_fn, option: PipeOption, args):
    """
    main_fn: main training function
    args: iterable of arguments for main_fn
    """
    # TODO: set world size w/ DDP + exact gpu map
    # TODO: refactor split_cnt
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")

    pipe_split = option.device_cnt

    assert pipe_split >= 2, "the number of pipeline stages should be bigger than 1"

    world_size = pipe_split + 1

    mp.spawn(run_local, args=(main_fn, args, option), nprocs=world_size, join=True)


def run_local(rank, main_fn, args, option):
    world_size = option.device_cnt + 1
    proc_name = get_rpc_name(rank)
    # dist_group = "master" if rank == 0 else "worker"

    device_map = dict()
    main_device = "cpu" if rank == 0 else f"cuda:{rank - 1}"

    for next_rank in range(1, world_size):
        if next_rank == rank:
            continue
        device_map[get_rpc_name(next_rank)] = {main_device: f"cuda:{next_rank - 1}"}
        # if rank == 0:
        #     device_map[get_rpc_name(next_rank)]["cuda:0"] = f"cuda:{next_rank}"

    print(f"device map for {rank}: {device_map}")

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=512, device_maps=device_map
    )

    dist.init_process_group("nccl", world_size=world_size, rank=rank)

    rpc.init_rpc(
        proc_name,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )

    set_logger_level(option.verbosity)

    if rank == 0:
        main_fn(*args)

    rpc.shutdown()
