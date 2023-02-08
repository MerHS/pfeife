import argparse
import os
import time
import threading

import torch
import torch.distributed.autograd as autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn


class MyModule:
    def __init__(self):
        super().__init__()
        self.tick = 0
        self.barrier = False
        self.lock = threading.Lock()

    def call_tick(self):
        # busy loop
        while self.barrier is False:
            with self.lock:
                self.tick += 1
        return self.tick

    def release(self):
        val = 2
        with self.lock:
            ticks = []
            ticks.append(self.tick)
            for _ in range(10):
                # meaningless loop
                for _ in range(50000):
                    val *= 1.000001
                ticks.append(self.tick)
            self.barrier = True
        return ticks


def run_main():
    # create a shared pointer of an instance of MyModule which is placed in a process "worker1"
    print("create remote module")
    remote_module = rpc.remote("worker1", MyModule)

    # asynchronously call a method 'call_tick'. returns a waitable object 'Future'.
    print("call remote call_tick")
    fut = remote_module.rpc_async().call_tick()

    # wait 2 seconds
    print("wait 2 seconds")
    time.sleep(2)

    # release the busy loop of call_tick
    print("release the barrier")
    release_ticks = remote_module.rpc_sync().release()

    # Wait until the future is resolved. returns the result of 'call_tick'
    print("wait until the future is resolved")
    call_tick = fut.wait()

    print(f"call_tick: {call_tick}\nrelease_ticks: ")
    for t in release_ticks:
        print(t)


def run_worker(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=2, rpc_backend_options=options)

    if rank == 0:
        run_main()

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    mp.spawn(run_worker, nprocs=2, join=True)
