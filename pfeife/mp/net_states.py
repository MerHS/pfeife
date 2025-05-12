import os
from enum import Enum

import torch
import torch.distributed as dist
import torch.distributed.device_mesh as dist_mesh

_PFEIFE_NETWORK_STATE = None


def parse_dp_mesh(dp_mesh, world_size):
    if dp_mesh is None or dp_mesh == "None":
        return (world_size,)
    return tuple(map(int, dp_mesh.split(",")))


# TODO: should be justified
def check_main_rank(rank, dp_mesh):
    # The axis with negative size is DP-axis. The other is PP-axis
    # (-a, b) -> think an a x b index matrix. the main rank is the final column
    # (a, -b) -> the main rank is the final row
    # else -> the main rank is the final rank
    if len(dp_mesh) == 1:
        return rank == dp_mesh[0] - 1

    a, b = abs(dp_mesh[0]), abs(dp_mesh[1])

    if dp_mesh[0] < 0:
        return rank % b == b - 1
    elif dp_mesh[1] < 0:
        return rank // b == a - 1
    else:
        return rank == a * b - 1


def target_main_rank(rank, dp_mesh):
    if len(dp_mesh) == 1:
        return dp_mesh[0] - 1

    a, b = abs(dp_mesh[0]), abs(dp_mesh[1])

    if dp_mesh[0] < 0:
        return rank // b * b + b - 1
    elif dp_mesh[1] < 0:
        return (a - 1) * b + rank % b
    else:
        return a * b - 1


def get_global_rank(pp_rank, main_rank, dp_mesh):
    if len(dp_mesh) == 1:
        return pp_rank

    a, b = abs(dp_mesh[0]), abs(dp_mesh[1])

    if dp_mesh[0] < 0:
        return main_rank - (b - 1 - pp_rank)
    elif dp_mesh[1] < 0:
        return pp_rank * b + main_rank % b
    else:
        return pp_rank


def get_pp_rank(rank, dp_mesh):
    if len(dp_mesh) == 1:
        return rank

    a, b = abs(dp_mesh[0]), abs(dp_mesh[1])

    if dp_mesh[0] < 0:
        return rank % b
    elif dp_mesh[1] < 0:
        return rank // b
    else:
        return rank


def get_pp_group_ranks(rank, dp_mesh):
    if len(dp_mesh) == 1:
        return list(range(dp_mesh[0]))

    a, b = abs(dp_mesh[0]), abs(dp_mesh[1])

    if dp_mesh[0] < 0:
        start = rank // b * b
        return [r for r in range(start, start + b)]
    elif dp_mesh[1] < 0:
        start = rank % b
        return [r for r in range(start, a * b, b)]
    else:
        return list(range(a * b))


def get_dp_group_ranks(rank, dp_mesh):
    if len(dp_mesh) == 1:
        return list([rank])

    a, b = abs(dp_mesh[0]), abs(dp_mesh[1])

    if dp_mesh[0] < 0:
        start = rank % b
        return [r for r in range(start, a * b, b)]
    elif dp_mesh[1] < 0:
        start = rank // b * b
        return [r for r in range(start, start + b)]
    else:
        return list([rank])


class PfeifeState:
    def __init__(self):
        env = os.environ
        self.world_size = int(env.get("WORLD_SIZE", 1))
        self.rank = int(env.get("RANK", 0))
        self.local_rank = int(env.get("LOCAL_RANK", 0))
        self.nnodes = int(env.get("NNODES", 1))
        self.node_rank = int(env.get("NODE_RANK", 0))

        self.dp_mesh = parse_dp_mesh(env.get("DP_MESH", "None"), self.world_size)
        self.is_main_rank = check_main_rank(self.rank, self.dp_mesh)
        self.is_main_process = self.rank == self.world_size - 1

        self.pp_rank = get_pp_rank(self.rank, self.dp_mesh)
        self.pp_main_rank = target_main_rank(self.rank, self.dp_mesh)
        self.pp_group_ranks = get_pp_group_ranks(self.rank, self.dp_mesh)
        self.dp_group_ranks = get_dp_group_ranks(self.rank, self.dp_mesh)
        self.pp_size = len(self.pp_group_ranks)
        self.dp_size = len(self.dp_group_ranks)

        # TODO: check if this is correct
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)

        torch.distributed.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
        )

        if len(self.dp_mesh) > 1:
            mesh_tuple = tuple([abs(d) for d in self.dp_mesh])
            mesh_dim_names = tuple(["pp" if d > 0 else "dp" for d in self.dp_mesh])
            self.device_mesh = dist_mesh.init_device_mesh(
                "cuda", mesh_tuple, mesh_dim_names=mesh_dim_names
            )
            self.pp_group = self.device_mesh.get_group("pp")
            self.dp_group = self.device_mesh.get_group("dp")
        else:
            self.device_mesh = None
            self.pp_group = None
            self.dp_group = None

        self.initialized = True

    def destroy(self):
        torch.distributed.destroy_process_group()

    def get_global_rank(self, pp_rank, pp_main_rank=None):
        if pp_main_rank is None:
            pp_main_rank = self.pp_main_rank
        return get_global_rank(pp_rank, pp_main_rank, self.dp_mesh)

    def get_dp_group_ranks(self, rank=None):
        if rank is None:
            rank = self.rank
        return get_dp_group_ranks(rank, self.dp_mesh)


def initialize_pfeife():
    """
    Warning: this function should be called by the pfeife launcher.
    """
    global _PFEIFE_NETWORK_STATE
    if _PFEIFE_NETWORK_STATE is None:
        _PFEIFE_NETWORK_STATE = PfeifeState()
    return _PFEIFE_NETWORK_STATE


def check_initialized():
    if _PFEIFE_NETWORK_STATE is None:
        raise RuntimeError("pfeife is not initialized. call initialize_pfeife() first.")


def get_state():
    check_initialized()
    return _PFEIFE_NETWORK_STATE


def get_device():
    return get_state().device
