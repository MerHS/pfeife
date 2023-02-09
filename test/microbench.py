import os
import time
import threading

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

SIZE = 5000


class MyModule:
    def __init__(self, device, next_rpc):
        super().__init__()
        self.device = device
        self.w1 = torch.rand(SIZE, SIZE, device=device, requires_grad=False)
        self.w2 = torch.rand(SIZE, SIZE, device=device, requires_grad=False)
        self.next = next_rpc
        self.wait_time = 0
        self.start_time = 0
        self.end_time = 0
        self.count = 0

    def matmul(self, x):
        start_time = time.time()

        w1 = self.w1
        w2 = self.w2
        y = x.mm(w1).mm(w2).mm(w1).mm(w2).mm(w1).mm(w2).mm(w1).mm(w2)

        torch.cuda.current_stream(self.device).synchronize()
        end_time = time.time()

        if self.next is not None:
            send = self.next.rpc_async().matmul(y)
            send.wait()
            self.wait_time = time.time()

        self.start_time = start_time
        self.end_time = end_time

    def comm_matmul(self, x):
        start_time = time.time()

        # can communication and matmul be overlapped?
        if self.next is not None:
            send = self.next.rpc_async().comm_matmul(x)
            self.wait_time = time.time()

        w1 = self.w1
        w2 = self.w2
        y = x.mm(w1).mm(w2).mm(w1).mm(w2).mm(w1).mm(w2).mm(w1).mm(w2)

        torch.cuda.current_stream(self.device).synchronize()
        end_time = time.time()

        if self.next is not None:
            send.wait()

        self.start_time = start_time
        self.end_time = end_time

    def get_time(self):
        return (
            self.start_time / self.count,
            self.end_time / self.count,
            self.wait_time / self.count,
        )


def run_once(base_time):
    print("Run once, send & matmul {SIZE}x{SIZE}")
    m3 = rpc.remote("worker3", MyModule, args=("cuda:3", None))
    m2 = rpc.remote("worker2", MyModule, args=("cuda:2", m3))
    m1 = rpc.remote("worker1", MyModule, args=("cuda:1", m2))

    x = torch.rand(SIZE, SIZE, device="cuda:0", requires_grad=False)
    torch.cuda.current_stream("cuda:0").synchronize()
    torch.cuda.synchronize()

    start_time = time.time()
    wait = m1.rpc_async().matmul(x)

    wait.wait()
    end_time = time.time()

    (s1, e1, w1) = m1.rpc_sync().get_time()
    (s2, e2, w2) = m2.rpc_sync().get_time()
    (s3, e3, w3) = m3.rpc_sync().get_time()

    print("+++ Starting from script start time +++")

    print(
        f"[Worker 0] Start: {start_time - base_time:8.5f}, End: {end_time - base_time:8.5f}"
    )
    print(
        f"[Worker 1] Start: {s1 - base_time:8.5f}, End: {e1 - base_time:8.5f}, Wait: {w1 - base_time:8.5f}"
    )
    print(
        f"[Worker 2] Start: {s2 - base_time:8.5f}, End: {e2 - base_time:8.5f}, Wait: {w2 - base_time:8.5f}"
    )
    print(f"[Worker 3] Start: {s3 - base_time:8.5f}, End: {e3 - base_time:8.5f}")

    print("\n+++ Latency (matmul -> send) +++")
    print(
        f"[Worker 0] Start: {start_time - start_time:8.5f}, End: {end_time - start_time:8.5f}"
    )
    print(
        f"[Worker 1] Receive latency (0->1): {s1 - start_time:8.5f}, matmul latency: {e1 - s1:8.5f}"
    )
    print(
        f"[Worker 2] Receive latency (1->2): {s2 - e1:8.5f}, matmul latency: {e2 - s2:8.5f}"
    )
    print(
        f"[Worker 3] Receive latency (2->3): {s3 - e2:8.5f}, matmul latency: {e3 - s3:8.5f}"
    )

    start_time = time.time()
    wait = m1.rpc_async().comm_matmul(x)

    wait.wait()
    end_time = time.time()

    (s1, e1, w1) = m1.rpc_sync().get_time()
    (s2, e2, w2) = m2.rpc_sync().get_time()
    (s3, e3, w3) = m3.rpc_sync().get_time()

    print("\n+++ Latency (send / matmul overlap) +++")
    print(
        f"[Worker 0] Start: {start_time - start_time:8.5f}, End: {end_time - start_time:8.5f}"
    )
    print(
        f"[Worker 1] Receive latency (0->1): {s1 - start_time:8.5f}, async send: {w1 - s1:8.5f}, matmul latency: {e1 - w1:8.5f}"
    )
    print(
        f"[Worker 2] Receive latency (1->2): {s2 - s1:8.5f}, async send: {w2 - s2:8.5f}, matmul latency: {e2 - w2:8.5f}"
    )
    print(
        f"[Worker 3] Receive latency (2->3): {s3 - s2:8.5f}, matmul latency: {e3 - s3:8.5f}"
    )


def run_worker(rank, base_time):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    curr_device = f"cuda:{rank}"
    device_map = {
        "worker0": {curr_device: "cuda:0"},
        "worker1": {curr_device: "cuda:1"},
        "worker2": {curr_device: "cuda:2"},
        "worker3": {curr_device: "cuda:3"},
    }

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=128, device_maps=device_map
    )

    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=4, rpc_backend_options=options)

    if rank == 0:
        run_once(base_time)

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    base_time = time.time()
    mp.spawn(run_worker, nprocs=4, join=True, args=(base_time,))
