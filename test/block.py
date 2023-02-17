import os
import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SIZE = 100


class Logger:
    def __init__(self, rank):
        self.rank = rank
        self.st = time.time()

    def log(self, text):
        name = "Main" if self.rank == 0 else "Worker"
        t = time.time()
        print(f"[{name}]({t - self.st:5.3f}) {text}")


def run_worker(rank, use_nccl):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if use_nccl:
        dist.init_process_group("nccl", world_size=2, rank=rank)
        dev0 = "cuda:0"
        dev1 = "cuda:1"
    else:
        dist.init_process_group("gloo", world_size=2, rank=rank)
        dev0 = "cpu"
        dev1 = "cpu"

    logger = Logger(rank)

    if rank == 0:
        ones = torch.ones(SIZE, SIZE, device=dev0)

        time.sleep(3)

        logger.log(f"send {ones.sum()}")
        dist.send(ones, 1)

        logger.log("finish")
    else:
        buf1 = torch.rand(SIZE, SIZE, device=dev1)

        # time.sleep(3)

        logger.log("recv")
        dist.recv(buf1, 0)

        logger.log(f"received {buf1.sum()}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nccl", action="store_true")
    args = parser.parse_args()

    print(f"Use NCCL: {args.nccl}")

    mp.spawn(run_worker, nprocs=2, join=True, args=(args.nccl,))
