import os
import time
import threading

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

SIZE = 5000

z = None


class MyModule:
    def __init__(self, rank, device):
        super().__init__()
        self.rank = rank
        self.device = device
        self.buffer = torch.empty(SIZE, SIZE, device=device)

    def recv(self):
        print("recv")
        fut = dist.irecv(self.buffer, 0)
        print("wait")
        fut.wait()
        print(self.buffer.sum())

    def send(self):
        dist.isend(z, 1)


def run_main(base_time):
    global z
    m1 = rpc.remote("worker1", MyModule, args=(1, "cuda:1"))
    m0 = rpc.remote("worker0", MyModule, args=(1, "cuda:0"))

    z = torch.rand(SIZE, SIZE, device="cuda:0")

    m1.rpc_async().recv()
    time.sleep(10)
    m0.rpc_async().send()


def run_worker(rank, base_time):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    proc_name = f"worker{rank}"

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    dist.init_process_group("nccl", world_size=2, rank=rank)
    rpc.init_rpc(proc_name, rank=rank, world_size=2, rpc_backend_options=options)

    if rank == 0:
        run_main(base_time)

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    base_time = time.time()
    mp.spawn(run_worker, nprocs=2, join=True, args=(base_time,))
