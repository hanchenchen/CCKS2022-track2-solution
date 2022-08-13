import functools
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist():
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
