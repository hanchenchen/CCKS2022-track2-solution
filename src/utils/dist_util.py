import functools
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend="nccl")


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = dist.get_rank()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
