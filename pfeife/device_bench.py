import os
import collections
import itertools
import pickle
import time
import threading
import argparse
from queue import Queue
from functools import partial

import torch
import numpy as np
from scipy.optimize import curve_fit
import torch.distributed as dist
from tqdm import tqdm

BYTE_DENOM = 1024 * 1024
BENCH_CNT = 10
DEFAULT_OFFSET = 0  # ms


def send_fn(x, c, m, x_break):
    x = x / BYTE_DENOM
    if x <= x_break:
        return c
    else:
        return m * (x - x_break) + c


def biased_relu(x, c, m, x_break):
    return np.piecewise(
        x,
        [x <= x_break, x > x_break],
        [lambda x: c, lambda x: m * (x - x_break) + c],
    )


class DeviceBench:
    def __init__(self, devices, offset=DEFAULT_OFFSET):
        self.devices = devices
        self.offset = offset

        self.bench_result = collections.defaultdict(list)
        self.send_fn_map = dict()
        self.send_fn_params = dict()

    def send_time(self, dev1, dev2, weight):
        if (dev1, dev2) in self.send_fn_map:
            return self.send_fn_map[(dev1, dev2)](weight) + self.offset
        elif (dev2, dev1) in self.send_fn_map:
            return self.send_fn_map[(dev2, dev1)](weight) + self.offset
        else:
            return 0

    def send_time_rank(self, rank1, rank2, weight):
        dev1 = self.devices[rank1]
        dev2 = self.devices[rank2]
        return self.send_time(dev1, dev2, weight)

    def _run_single(self, rank1, rank2, b, w, h):
        dev1 = self.devices[rank1]
        dev2 = self.devices[rank2]

        stream = torch.cuda.current_stream(dev1)
        test_loads = [
            torch.ones(b, w, h, device=dev1, dtype=torch.float32) for _ in range(5)
        ]
        main_loads = [
            torch.ones(b, w, h, device=dev1, dtype=torch.float32)
            for _ in range(BENCH_CNT)
        ]

        test_loads2 = [
            torch.zeros(b, w, h, device=dev2, dtype=torch.float32) for _ in range(5)
        ]
        main_loads2 = [
            torch.zeros(b, w, h, device=dev2, dtype=torch.float32)
            for _ in range(BENCH_CNT)
        ]

        elapsed = 0
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(dev1)
        torch.cuda.synchronize(dev2)

        for i in range(5):
            test_loads2[i].copy_(test_loads[i])

        torch.cuda.synchronize(dev1)
        torch.cuda.synchronize(dev2)

        e1.record(stream)
        for i in range(BENCH_CNT):
            main_loads2[i].copy_(main_loads[i])
        e2.record(stream)

        torch.cuda.synchronize(dev1)
        torch.cuda.synchronize(dev2)
        elapsed = e1.elapsed_time(e2)

        return elapsed / BENCH_CNT

    def _load_file(self, file_path):
        if file_path is not None and os.path.exists(file_path):
            with open(file_path, "rb") as fp:
                self.send_fn_params = pickle.load(fp)

            for k, (c, m, x_break) in self.send_fn_params.items():
                self.send_fn_map[k] = partial(send_fn, c=c, m=m, x_break=x_break)

            # loaded
            return True
        return False

    def run_bench(self, print_result=False, file_path=None):
        if file_path is not None:
            if self._load_file(file_path):
                return

        heights = [100, 400, 800, 3200]
        widths = [100, 400, 800, 3200]
        batches = [1, 2, 4]

        devices = self.devices
        rank_pair = []
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                rank_pair.append((i, j))

        bench_len = len(rank_pair) * len(batches) * len(widths) * len(heights)
        tqdm_bar = tqdm(
            total=bench_len, desc="Benchmarking device P2P communication...", ncols=80
        )

        self.bench_result = collections.defaultdict(list)
        for rank1, rank2 in rank_pair:
            for b, w, h in itertools.product(batches, widths, heights):
                result = self._run_single(rank1, rank2, b, w, h)
                self.bench_result[(rank1, rank2)].append((b, w, h, result))
                tqdm_bar.update(1)

        tqdm_bar.close()

        for k, v in self.bench_result.items():
            x = []
            y = []
            for b, w, h, t in v:
                x.append(4 * b * w * h / BYTE_DENOM)
                y.append(t)

            params, _ = curve_fit(
                biased_relu,
                np.array(x),
                np.array(y),
                p0=[0.001, 0.1, 0.1],
            )

            c, m, x_break = params
            print(f"{k}: {c:.5f} + {m:.5f} * (x - {x_break:.5f})")
            self.send_fn_params[k] = (c, m, x_break)
            self.send_fn_map[k] = partial(send_fn, c=c, m=m, x_break=x_break)

        if print_result:
            # heights = [1000, 2000, 4000, 8000, 16000]
            # widths = [1000, 2000, 4000, 8000, 16000]
            # for rank1, rank2 in rank_pair:
            #     for w, h in itertools.product(widths, heights):
            #         result = self._run_single(rank1, rank2, 1, w, h)
            #         self.bench_result[(rank1, rank2)].append((1, w, h, result))

            for k, v in self.bench_result.items():
                print(k)
                print("batch\twidth\theight\tbytes(MB)\ttime(us)\ttime_diff(us)")
                f = self.send_fn_map[k]
                for b, w, h, t in v:
                    mul = 4 * b * w * h
                    print(
                        f"{b}\t{w}\t{h}\t{mul/BYTE_DENOM:.5f}\t{t*1000:.5f}\t{(t - f(mul))*1000:.5f}"
                    )

        if file_path is not None:
            with open(file_path, "wb") as fp:
                pickle.dump(self.send_fn_params, fp)


class NCCLDeviceBench(DeviceBench):
    def __init__(self, devices, offset=DEFAULT_OFFSET, rank=0, curr_device=None):
        super().__init__(devices, offset)
        self.rank = rank
        self.curr_device = curr_device
        self.result_tensor = torch.zeros(1, device=self.curr_device)

    def run_bench(self, print_result=False, file_path=None):
        # override print_result
        print_result = print_result and self.rank == 0
        if self.rank != 0 and file_path is not None:
            if self._load_file(file_path):
                return

        file_path = file_path if self.rank == 0 else None
        return super().run_bench(print_result, file_path)

    def _run_single(self, rank1, rank2, b, w, h):
        if self.rank == rank1 or self.rank == rank2:
            result = self._run_rank(rank1, rank2, b, w, h)
        else:
            result = 0

        self.result_tensor = torch.zeros(1, device=self.curr_device)
        if self.rank == rank1:
            # broadcast
            self.result_tensor.fill_(result)

        dist.broadcast(self.result_tensor, src=rank1)
        dist.barrier()
        result = self.result_tensor.item()

        return result

    def _run_rank(self, rank1, rank2, b, w, h):
        torch.cuda.synchronize(device=self.curr_device)
        fn = torch.ones if self.rank == rank1 else torch.zeros
        test_loads = [
            fn(b, w, h, device=self.curr_device, dtype=torch.float32) for _ in range(5)
        ]
        main_loads = [
            fn(b, w, h, device=self.curr_device, dtype=torch.float32)
            for _ in range(BENCH_CNT)
        ]

        elapsed = 0

        if self.rank == rank1:
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize(device=self.curr_device)

            for i, t_gpu in enumerate(test_loads):
                dist.send(t_gpu, dst=rank2)

            torch.cuda.synchronize(device=self.curr_device)

            e1.record(torch.cuda.current_stream(device=self.curr_device))
            for i, t_gpu in enumerate(main_loads):
                dist.send(t_gpu, dst=rank2)
            e2.record(torch.cuda.current_stream(device=self.curr_device))

            torch.cuda.synchronize(device=self.curr_device)
            elapsed = e1.elapsed_time(e2)

        else:
            torch.cuda.synchronize(device=self.curr_device)

            for i in range(5):
                dist.recv(test_loads[i], src=rank1)

            torch.cuda.synchronize(device=self.curr_device)

            for i in range(BENCH_CNT):
                dist.recv(main_loads[i], src=rank1)

            torch.cuda.synchronize(device=self.curr_device)

        return elapsed / BENCH_CNT


class MPIDeviceBench(DeviceBench):
    def __init__(self, devices, offset=DEFAULT_OFFSET, comm=None, curr_device=None):
        super().__init__(devices, offset)
        self.comm = comm
        self.rank = comm.Get_rank()

        self.curr_device = curr_device

    def run_bench(self, print_result=False, file_path=None):
        # override print_result
        print_result = print_result and self.rank == 0
        if self.rank != 0 and file_path is not None:
            if self._load_file(file_path):
                return

        file_path = file_path if self.rank == 0 else None
        return super().run_bench(print_result, file_path)

    def _run_single(self, rank1, rank2, b, w, h):
        if self.rank == rank1 or self.rank == rank2:
            result = self._run_rank(rank1, rank2, b, w, h)
        else:
            result = 0

        self.result_tensor = torch.zeros(1, device="cpu")
        if self.rank == rank1:
            # broadcast
            self.result_tensor.fill_(result)

        comm.Bcast(self.result_tensor, root=rank1)
        result = self.result_tensor.item()
        comm.barrier()

        return result

    def _run_rank(self, rank1, rank2, b, w, h):
        torch.cuda.synchronize(device=self.curr_device)
        fn = torch.ones if self.rank == rank1 else torch.zeros

        dev = self.curr_device if rank1 == self.rank else "cpu"

        test_loads = [fn(b, w, h, device=dev, dtype=torch.float32) for _ in range(5)]
        main_loads = [
            fn(b, w, h, device=dev, dtype=torch.float32) for _ in range(BENCH_CNT)
        ]

        elapsed = 0

        if self.rank == rank1:
            torch.cuda.synchronize(device=self.curr_device)

            for i, t_gpu in enumerate(test_loads):
                t_cpu = t_gpu.cpu()
                comm.Send([t_cpu, MPI.FLOAT], dest=rank2, tag=i + 10)

            torch.cuda.synchronize(device=self.curr_device)

            start_time = time.time()
            for i, t_gpu in enumerate(main_loads):
                comm.Send([t_cpu, MPI.FLOAT], dest=rank2, tag=i + 20)
            torch.cuda.synchronize(device=self.curr_device)
            end_time = time.time()

            elapsed = (end_time - start_time) * 1000

        else:
            tl2 = [
                fn(b, w, h, device=self.curr_device, dtype=torch.float32)
                for _ in range(5)
            ]
            ml2 = [
                fn(b, w, h, device=self.curr_device, dtype=torch.float32)
                for _ in range(BENCH_CNT)
            ]

            torch.cuda.synchronize(device=self.curr_device)

            for i in range(5):
                comm.Recv([test_loads[i], MPI.FLOAT], source=rank1, tag=i + 10)
                tl2[i].copy_(test_loads[i])

            torch.cuda.synchronize(device=self.curr_device)

            for i in range(BENCH_CNT):
                comm.Recv([main_loads[i], MPI.FLOAT], source=rank1, tag=i + 20)
                ml2[i].copy_(main_loads[i])

            torch.cuda.synchronize(device=self.curr_device)

        return elapsed / BENCH_CNT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--nccl", action="store_true")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--devs", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    # WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", -1))
    # WORLD_RANK = int(os.environ.get("SLURM_PROCID", -1))

    if args.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        device = torch.device("cuda", rank)  # should be local rank
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", world_size=args.world_size, rank=rank
        )
        bench = MPIDeviceBench(
            [i for i in range(args.world_size)], comm=comm, curr_device=device
        )
    elif args.nccl:  # via torchrun
        torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
        rank = torch.distributed.get_rank()
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        bench = NCCLDeviceBench(
            [i for i in range(args.world_size)], rank=rank, curr_device=device
        )
    else:
        if len(args.devs) != args.world_size:
            args.devs = [i for i in range(args.world_size)]
        bench = DeviceBench([f"cuda:{i}" for i in args.devs])

    bench.run_bench(True)
