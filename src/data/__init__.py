import importlib
import math
import random
from functools import partial
from os import path as osp

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

from src.utils import get_root_logger, scandir

__all__ = ["create_dataset", "create_dataloader"]

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(data_folder)
    if v.endswith("_dataset.py")
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f"src.data.{file_name}") for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt["type"]

    # dynamic instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f"Dataset {dataset_type} is not found.")

    dataset = dataset_cls(dataset_opt)

    logger = get_root_logger()
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} ' "is created."
    )
    return dataset


def create_dataloader(dataset, dataset_opt, sampler, seed):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        sampler (torch.utils.data.sampler): Data sampler.
        seed (int): Seed.
    """
    rank = dist.get_rank()
    dataloader_args = dict(
        dataset=dataset,
        batch_size=dataset_opt["batch_size_per_gpu"],
        shuffle=False,
        sampler=sampler,
        num_workers=dataset_opt["num_worker_per_gpu"],
        pin_memory=dataset_opt["pin_memory"],
    )
    dataloader_args["worker_init_fn"] = partial(
        worker_init_fn,
        num_workers=dataset_opt["num_worker_per_gpu"],
        rank=rank,
        seed=seed,
    )
    return torch.utils.data.DataLoader(**dataloader_args)


def create_dataset_dataloader(dataset_opt, shuffle, seed):
    dataset = create_dataset(dataset_opt)
    if shuffle:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    dataloader = create_dataloader(dataset, dataset_opt, sampler, seed)

    batch_size_per_gpu = dataset_opt["batch_size_per_gpu"]
    world_size = dist.get_world_size()
    num_iter_per_epoch = math.ceil(len(dataset) / (batch_size_per_gpu * world_size))
    logger = get_root_logger()
    logger.info(
        f"Dataset statistics:"
        f"\n\tNumber of train samples: {len(dataset)}"
        f"\n\tBatch size per gpu: {batch_size_per_gpu}"
        f"\n\tWorld size (gpu number): {world_size}"
        f"\n\tRequire iter number per epoch: {num_iter_per_epoch}"
    )
    return dataset, dataloader, num_iter_per_epoch


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
