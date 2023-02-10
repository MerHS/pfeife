import os
import time
import threading

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

SIZE = 5000


class MyModule:
    def __init__(self, rank, device, next_rpc):
        super().__init__()
        self.rank = rank
        self.device = device
        self.w1 = torch.rand(SIZE, SIZE, device=device, requires_grad=False)
        self.w2 = torch.rand(SIZE, SIZE, device=device, requires_grad=False)
        self.next = next_rpc
        self.wait_time = 0
        self.start_time = 0
        self.end_time = 0
        self.count = 0

        self.input = torch.empty(SIZE, SIZE, device=device, requires_grad=False)

    def matmul(self, base_time):
        dist.recv(self.input, self.rank - 1)
        x = self.input
        torch.cuda.current_stream(self.device).synchronize()

        start_time = time.time()

        w1 = self.w1
        w2 = self.w2

        y = x.mm(w1).mm(w2).mm(w1).mm(w2)
        y = y / y.max()
        y = y.mm(w1).mm(w2).mm(w1).mm(w2)
        y = y / y.max()

        torch.cuda.current_stream(self.device).synchronize()
        end_time = time.time()

        if self.next is not None:
            mm = self.next.rpc_async().matmul(end_time)
            dist.send(y, self.rank + 1)
            # send.wait()
            mm.wait()
            self.wait_time += time.time() - start_time

        self.start_time += start_time - base_time
        self.end_time += end_time - start_time
        self.count += 1
        self.y = y

    def reset_time(self):
        self.start_time = 0
        self.end_time = 0
        self.wait_time = 0
        self.count = 0

    def get_time(self):
        return (
            self.start_time / self.count,
            self.end_time / self.count,
            self.wait_time / self.count,
        )

    def get_weights(self):
        return (self.w1, self.w2, self.y)


def calc(x, w1, w2):
    y = x.mm(w1).mm(w2).mm(w1).mm(w2)
    y = y / y.max()
    y = y.mm(w1).mm(w2).mm(w1).mm(w2)
    y = y / y.max()
    return y


def run_main(base_time):
    print(f"Repeat 20 times, matmul -> send {SIZE}x{SIZE}")
    m3 = rpc.remote("worker3", MyModule, args=(3, "cuda:3", None))
    m2 = rpc.remote("worker2", MyModule, args=(2, "cuda:2", m3))
    m1 = rpc.remote("worker1", MyModule, args=(1, "cuda:1", m2))

    x = torch.rand(SIZE, SIZE, device="cuda:0", requires_grad=False)
    torch.cuda.current_stream("cuda:0").synchronize()
    torch.cuda.synchronize()

    # warmup
    print("warmup")
    for _ in range(5):
        start_time = time.time()
        wait = m1.rpc_async().matmul(start_time)
        dist.send(x, 1)
        wait.wait()

    m1.rpc_sync().reset_time()
    m2.rpc_sync().reset_time()
    m3.rpc_sync().reset_time()

    # bench
    print("start benchmark")
    base_start = time.time()
    for _ in range(20):
        start_time = time.time()
        wait = m1.rpc_async().matmul(start_time)
        dist.send(x, 1)
        wait.wait()

    base_end = time.time()

    (s1, e1, w1) = m1.rpc_sync().get_time()
    (s2, e2, w2) = m2.rpc_sync().get_time()
    (s3, e3, w3) = m3.rpc_sync().get_time()

    print("\n+++ Latency +++")
    print(f"[Master] 20 times Total: {base_end - base_start:8.5f}")
    print(
        f"[Worker 1] Receive latency (nccl 0->1): {s1:8.5f}, matmul latency: {e1:8.5f}"
    )
    print(
        f"[Worker 2] Receive latency (nccl 1->2): {s2:8.5f}, matmul latency: {e2:8.5f}"
    )
    print(
        f"[Worker 3] Receive latency (nccl 2->3): {s3:8.5f}, matmul latency: {e3:8.5f}"
    )

    (w1, w11, y1) = m1.rpc_sync().get_weights()
    (w2, w22, y2) = m2.rpc_sync().get_weights()
    (w3, w33, y3) = m3.rpc_sync().get_weights()

    print("\n+++ Validity +++")
    y11 = calc(x, w1, w11)
    y22 = calc(y11, w2, w22)
    y33 = calc(y22, w3, w33)
    print(f"[{(y1 == y11).all()}] y1 remote: {y1[0][0:5]}, local: {y11[0][0:5]}")
    print(f"[{(y2 == y22).all()}] y2 remote: {y2[0][0:5]}, local: {y22[0][0:5]}")
    print(f"[{(y3 == y33).all()}] y3 remote: {y3[0][0:5]}, local: {y33[0][0:5]}")


def run_worker(rank, base_time):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    curr_device = f"cuda:{rank}"
    proc_name = f"worker{rank}"

    device_map = {
        "worker0": {curr_device: "cuda:0"},
        "worker1": {curr_device: "cuda:1"},
        "worker2": {curr_device: "cuda:2"},
        "worker3": {curr_device: "cuda:3"},
    }

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=128, device_maps=device_map
    )

    dist.init_process_group("nccl", world_size=4, rank=rank)
    rpc.init_rpc(proc_name, rank=rank, world_size=4, rpc_backend_options=options)

    if rank == 0:
        run_main(base_time)

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    base_time = time.time()
    mp.spawn(run_worker, nprocs=4, join=True, args=(base_time,))
