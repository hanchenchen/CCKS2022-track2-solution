import argparse
import datetime
import logging
import math
import time
from os import path as osp

import torch
import torch.distributed as dist

from src.data import create_dataset_dataloader
from src.models import create_model
from src.utils import (
    MessageLogger,
    dict2str,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_dist,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    parse,
    set_random_seed,
)


def parse_options():
    """
    parse options
    set distributed setting
    set ramdom seed
    set cudnn deterministic
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option YAML file."
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt)

    # distributed setting
    init_dist()
    opt["rank"] = dist.get_rank()
    opt["world_size"] = dist.get_world_size()

    # random seed
    seed = opt.get("manual_seed")
    assert seed is not None, "Seed must be set."
    set_random_seed(seed + opt["rank"])

    # cudnn deterministic
    if opt["cudnn_deterministic"]:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    return opt


def init_loggers(opt, prefix, log_level, use_tb_logger):
    log_file = osp.join(
        opt["path"]["log"], f"{prefix}_{opt['name']}_{get_time_str()}.log"
    )
    logger = get_root_logger(log_level=log_level, log_file=log_file, initialized=False)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize tensorboard logger and wandb logger
    tb_logger = None
    if use_tb_logger:
        tb_logger = init_tb_logger(log_dir=opt["path"]["tb_logger"])
        if opt["logger"]["wandb"]:
            init_wandb_logger(opt)
    return logger, tb_logger


def main():
    opt = parse_options()
    seed = opt["manual_seed"]

    # mkdir for experiments and logger
    make_exp_dirs(opt)

    # initialize loggers
    logger, tb_logger = init_loggers(
        opt, prefix="train", log_level=logging.INFO, use_tb_logger=True
    )

    # create train, validation, test datasets and dataloaders
    for phase in opt["datasets"]:
        opt["datasets"][phase]["phase"] = phase
    train_set, train_loader, num_iter_per_epoch = create_dataset_dataloader(
        opt["datasets"]["train"], shuffle=True, seed=seed
    )
    val_set, val_loader, _ = create_dataset_dataloader(
        opt["datasets"]["val"], shuffle=False, seed=seed
    )
    test_set, test_loader, _ = create_dataset_dataloader(
        opt["datasets"]["test"], shuffle=False, seed=seed
    )

    # log training statistics
    total_iters = int(opt["train"]["total_iter"])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
    logger.info(f"\n\tTotal epochs: {total_epochs}\n\tTotal iters: {total_iters}.")

    # create model
    model = create_model(opt, train_set, val_set, test_set)
    current_iter = model.current_iter

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # logger.info("Save model")
    # model.save_network()
    logger.info("Validate")
    model.dist_validation(val_loader, tb_logger)

    # training
    logger.info(f"Start training")
    start_time = time.time()

    for current_epoch in range(1, total_epochs + 1):
        model.set_current_epoch(current_epoch)
        train_loader.sampler.set_epoch(current_epoch)
        data_time = time.time()
        iter_time = time.time()
        for train_data in train_loader:
            data_time = time.time() - data_time

            current_iter += 1
            model.set_current_iter(current_iter)
            if current_iter > total_iters:
                break
            # training
            model.feed_data(train_data, train=True)
            model.optimize_parameters()
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": current_epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update({"time": iter_time, "data_time": data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            # save model, validation and test
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Save model")
                model.save_network()
                logger.info("Validate")
                model.dist_validation(val_loader, tb_logger)
            data_time = time.time()
            iter_time = time.time()

    if tb_logger is not None:
        tb_logger.close()
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")


if __name__ == "__main__":
    main()
