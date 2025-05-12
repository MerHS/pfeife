import os
import subprocess
import argparse
import shlex
import signal
import re

from pfeife.mp import initialize_pfeife
from pfeife.mp.net_states import check_main_rank, parse_dp_mesh
from pfeife.mp.worker import ProcessWorker
from pfeife.option import PipeOption


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launcher",
        type=str,
        default="torchrun",
        choices=["python", "torchrun", "slurm"],
    )
    parser.add_argument("--world_size", type=int, default=1, help="Number of processes")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of the node")
    parser.add_argument("--all_main", action="store_true", help="All ranks are main")
    parser.add_argument(
        "--dp_mesh",
        type=str,
        default="None",
        help="Structure of FSDP Mesh. Format: '2,-4' = 2 (pos: PP size) * 4 (neg: DP size)",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Address of the master node",
    )
    parser.add_argument(
        "--master_port", default="24446", help="Port of the master node"
    )
    parser.add_argument("--run_proc", action="store_true")
    parser.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="Command and its arguments"
    )

    return parser.parse_args()


def get_slurm_env(args):
    WORLD_SIZE = os.environ.get("SLURM_NTASKS", 1)
    WORLD_RANK = os.environ.get("SLURM_PROCID", 0)
    LOCAL_RANK = os.environ.get("SLURM_LOCALID", 0)
    NNODES = os.environ.get("SLURM_JOB_NUM_NODES", 1)
    NODE_RANK = os.environ.get("SLURM_NODEID", 0)

    env = os.environ.copy()

    # Modify the environment by adding or changing variables
    env["WORLD_SIZE"] = str(WORLD_SIZE)
    env["RANK"] = str(WORLD_RANK)
    env["NODE_RANK"] = str(NODE_RANK)
    env["LOCAL_RANK"] = str(LOCAL_RANK)
    env["NNODES"] = str(NNODES)
    env["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", args.master_addr)
    env["MASTER_PORT"] = os.environ.get("MASTER_PORT", args.master_port)
    env["DP_MESH"] = str(args.dp_mesh)

    return env


def get_torchrun_env(args):
    env = os.environ.copy()

    env["WORLD_SIZE"] = str(args.world_size)
    env["NODE_RANK"] = str(args.node_rank)
    env["NNODES"] = str(args.nnodes)
    env["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", args.master_addr)
    env["MASTER_PORT"] = os.environ.get("MASTER_PORT", args.master_port)
    env["DP_MESH"] = str(args.dp_mesh)

    return env


def get_python_env(args):
    envs = []
    for rank in range(args.world_size):
        env = os.environ.copy()

        env["WORLD_SIZE"] = str(args.world_size)
        env["NNODES"] = str(args.nnodes)
        env["MASTER_ADDR"] = str(os.environ.get("MASTER_ADDR", args.master_addr))
        env["MASTER_PORT"] = str(os.environ.get("MASTER_PORT", args.master_port))
        env["RANK"] = str(rank)
        env["NODE_RANK"] = str(rank // args.nnodes)
        env["LOCAL_RANK"] = str(rank % args.world_size)

        envs.append(env)

    return envs


def init_main(args):
    file_path = os.path.abspath(__file__)
    all_main = "--all_main " if args.all_main else ""
    remain_cmd = all_main + " ".join(args.cmd)
    if args.launcher == "torchrun":
        env = get_torchrun_env(args)
        # TODO: set nproc_per_node
        cmd = f"torchrun --nproc_per_node {args.world_size} --master_addr {args.master_addr} --master_port {args.master_port} {file_path} --run_proc {remain_cmd}"
    elif args.launcher == "slurm":
        env = get_slurm_env(args)
        cmd = f"srun python {file_path} --run_proc {remain_cmd}"
    elif args.launcher == "python":
        envs = get_python_env(args)
        cmd = f"python {file_path} --run_proc {remain_cmd}"
    else:
        raise RuntimeError(f"Invalid launcher: {args.launcher}")

    if args.launcher == "python":
        procs = []
        try:
            for env in envs:
                proc = subprocess.Popen(cmd, env=env, shell=True)
                procs.append(proc)
            for proc in procs:
                proc.wait()
        except KeyboardInterrupt:
            for proc in procs:
                proc.send_signal(signal.SIGKILL)
            for proc in procs:
                proc.wait()
            exit(-1)
    else:
        try:
            proc = subprocess.Popen(cmd, env=env, shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGKILL)
            proc.wait()

        exit(proc.returncode)


def run_master(args):
    remain_cmd = " ".join(args.cmd)
    print(f"run master: {remain_cmd}")

    try:
        proc = subprocess.Popen(f"python {remain_cmd}", shell=True)
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGKILL)

    exit(proc.returncode)


def run_worker(args):
    print(f"run worker: {args.cmd}")
    try:
        # cmd_args = shlex.split(" ".join(args.cmd))
        # parser = argparse.ArgumentParser()
        # PipeOption.add_arguments(parser)
        # option_args = parser.parse_known_args(cmd_args)[0]
        # option = PipeOption.from_args(option_args)

        initialize_pfeife()
        worker = ProcessWorker()
        worker.wait_for_init()
        worker.run_loop()
        worker.destroy_workers()
    except KeyboardInterrupt:
        exit(-1)


if __name__ == "__main__":
    args = parse_args()
    if args.run_proc:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # initialize_pfeife()
        dp_mesh = parse_dp_mesh(os.environ["DP_MESH"], world_size)
        is_main = check_main_rank(rank, dp_mesh)

        if args.all_main or is_main:
            run_master(args)
        else:
            run_worker(args)
    else:
        init_main(args)
