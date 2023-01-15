import os

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp


def run_master(main_fn, args, *, pipe_split=2):
    """
    main_fn: main training function
    args: iterable of arguments for main_fn
    """
    # TODO: set world size w/ DDP + exact gpu map
    # TODO: refactor split_cnt
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")

    world_size = pipe_split

    mp.spawn(run_local, args=(main_fn, args, pipe_split), nprocs=world_size, join=True)


def run_local(rank, main_fn, args, pipe_split):
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=512)

    world_size = pipe_split

    # TODO: for DDP
    # backend = "nccl" if args.cuda else "gloo"
    # torch.distributed.init_process_group(
    #     backend=backend, rank=rank, world_size=world_size
    # )

    if rank == 0:
        rpc.init_rpc(
            f"worker_{rank}", rank=rank, world_size=world_size, rpc_backend_options=options
        )
        main_fn(*args)
    else:
        rpc.init_rpc(
            f"worker_{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )

    rpc.shutdown()
