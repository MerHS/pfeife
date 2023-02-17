import torch.distributed as dist

if __name__ == "__main__":
    print("connecting...")
    dist.init_process_group(
        "nccl", init_method="tcp://147.46.248.150:29500", rank=0, world_size=2
    )
    print("connected")
