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


def run_worker(use_nccl):
    rank = int(os.environ["RANK"])

    if use_nccl:
        dist.init_process_group("nccl")
        dev0 = "cuda:0"
    else:
        dist.init_process_group("gloo")
        dev0 = "cpu"

    logger = Logger(rank)

    if rank == 0:
        ones = torch.ones(SIZE, SIZE, device=dev0)

        logger.log(f"isend 1 / {ones.sum()}")
        f1 = dist.isend(ones, 1, tag=1)

        logger.log(f"wait")
        f1.wait()

        logger.log("finished")
    else:
        buf = torch.rand(SIZE, SIZE, device=dev0)

        logger.log("irecv 1")
        f1 = dist.irecv(buf, 0, tag=1)

        logger.log(f"wait")
        f1.wait()

        logger.log(f"recv {buf.sum()}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nccl", action="store_true")
    args = parser.parse_args()

    print(f"Use NCCL: {args.nccl}")

    run_worker(args.nccl)
