import argparse 

import torch
import torch.distributed.autograd as autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn

import os
import time


class MyModule(nn.Module):
    def __init__(self, device, comm_mode):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(1000, 1000).to(device)
        self.comm_mode = comm_mode

    def forward(self, x):
        # x.to() is a no-op if x is already on self.device
        y = self.linear(x.to(self.device))
        return y.cpu() if self.comm_mode == "cpu" else y

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


def measure(comm_mode, dev0, dev1):
    # local module on "worker0/cuda:0"
    lm = MyModule(f"cuda:{dev0}", comm_mode)
    # remote module on "worker1/cuda:1"
    rm = rpc.remote("worker1", MyModule, args=(f"cuda:{dev1}", comm_mode))
    # prepare random inputs
    x = torch.randn(1000, 1000).cuda(0)

    tik = time.time()
    for _ in range(10):
        with autograd.context() as ctx:
            y = rm.rpc_sync().forward(lm(x))
            autograd.backward(ctx, [y.sum()])
    # synchronize on "cuda:0" to make sure that all pending CUDA ops are
    # included in the measurements
    torch.cuda.current_stream(f"cuda:{dev0}").synchronize()
    tok = time.time()
    print(f"({comm_mode} {dev0} => {dev1}) RPC total execution time: {tok - tik}")


def run_worker(rank, dev0, dev1):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        print(f"{dev0} to {dev1}")
        options.set_device_map("worker1", {dev0: dev1})
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=2,
            rpc_backend_options=options
        )
        measure("cpu", dev0, dev1)
        measure("cuda", dev0, dev1)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=2,
            rpc_backend_options=options
        )

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "devs",
        nargs="+",
        type=int
    )
    args = parser.parse_args()
    world_size = 2
    mp.spawn(run_worker, nprocs=world_size, join=True, args=args.devs)