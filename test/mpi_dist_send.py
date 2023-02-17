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


def run_worker(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("mpi")
    dev0 = "cpu"
    dev1 = "cpu"

    logger = Logger(rank)

    if rank == 0:
        ones = torch.ones(SIZE, SIZE, device=dev0)
        twos = torch.ones(SIZE * 2, SIZE * 2, device=dev0) * 2

        time.sleep(3)

        logger.log(f"isend 2 / {twos.sum()}")
        f2 = dist.isend(twos, 1, tag=2)
        logger.log(f"isend 2 ends")

        time.sleep(5)

        logger.log(f"isend 1 / {ones.sum()}")
        f1 = dist.isend(ones, 1, tag=1)
        logger.log(f"isend 1 ends")

        logger.log("waiting...")

        f1.wait()
        f2.wait()

        logger.log("finish")
    else:
        buf1 = torch.rand(SIZE, SIZE, device=dev1)
        buf2 = torch.rand(SIZE * 2, SIZE * 2, device=dev1)

        logger.log("irecv 1")
        f1 = dist.irecv(buf1, 0, tag=1)

        logger.log("irecv 2")
        f2 = dist.irecv(buf2, 0, tag=2)

        logger.log("waiting...")

        f2.wait()
        logger.log(f"received {buf2.sum()}")

        f1.wait()
        logger.log(f"received {buf1.sum()}")


if __name__ == "__main__":
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    run_worker(rank)
