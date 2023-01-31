import os

import time
import threading
import torch.distributed as dist
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

    world_size = pipe_split + 1

    mp.spawn(run_local, args=(main_fn, args, pipe_split), nprocs=world_size, join=True)


def run_local(rank, main_fn, args, pipe_split):
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=512)

    world_size = pipe_split + 1

    # TODO: for DDP
    # backend = "nccl" if args.cuda else "gloo"
    # torch.distributed.init_process_group(
    #     backend=backend, rank=rank, world_size=world_size
    # )
    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
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


class Worker:
    def __init__(self, rank):
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

        self.result_lock = threading.Lock()
        self.rv = threading.Condition(self.result_lock)

        self.load = []
        self.rank = rank
        self.next = None
        self.value = None
        self.thread = threading.Thread(
            target=self.run, name=f"workerT_{rank}", daemon=True
        )
        self.thread.start()
        print(f"init {self.rank}")

    def set_next_worker(self, next_worker):
        print(f"set next from {self.rank}")
        self.next = next_worker
        # print("got: ", self.next.rpc_sync().pong(self.rank))

    def pong(self, x):
        msg = f"pong from {x} to {self.rank}"
        print(msg)
        return msg

    def run(self):
        print(f"start {self.rank}")
        while True:
            load = None
            with self.cv:
                self.cv.wait_for(lambda: len(self.load) > 0)
                load = self.load.pop(0)
                if load is not None:
                    load_type, value = load
                    if load_type == "send":
                        self.send(value)
                    elif load_type == "stop":
                        print(f"stop {self.rank}")
                        break

    def stop(self):
        self.append("stop", None)

    def send(self, value):
        if self.next is not None:
            print(f"send {value} from {self.rank}")
            self.next.rpc_sync().append("send", value + 1)
        else:
            with self.rv:
                print(f"save {value} from {self.rank}")
                self.value = value + 1
                self.rv.notify()

    def recv(self):
        with self.rv:
            self.rv.wait_for(lambda: self.value is not None)
            v = self.value
            self.value = None
            return v

    def append(self, load_type, value):
        with self.cv:
            print(f"append {value} of {load_type} from {self.rank}")
            self.load.append((load_type, value))
            self.cv.notify()


def run(no):
    workers = []
    for idx in range(no):
        worker = rpc.remote(f"worker_{idx+1}", Worker, args=(idx,))
        workers.append(worker)

    for prev, next_worker in zip(workers, workers[1:]):
        prev.rpc_sync().set_next_worker(next_worker)

    future = workers[-1].rpc_async().recv()
    time.sleep(2)
    workers[0].rpc_async().send(-10)
    future.wait()
    print("result:", future.value())

    future = workers[-1].rpc_async().recv()
    time.sleep(2)
    workers[0].rpc_async().append("send", 0)

    future.wait()
    print("result:", future.value())

    future = workers[-1].rpc_async().recv()
    time.sleep(2)
    workers[0].rpc_async().append("send", 5)

    future.wait()
    print("result:", future.value())

    # for worker in workers:
    #     worker.rpc_async().stop()


if __name__ == "__main__":
    split_no = 4
    run_master(run, (split_no,), pipe_split=split_no)
