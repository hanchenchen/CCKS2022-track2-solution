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
    parser.add_argument("--model_path", type=str, help="Path to pre-trained model.")
    parser.add_argument(
        "--strict_load",
        action="store_true",
        default=False,
        help="Load pre-trained model strictly.",
    )
    parser.add_argument("--test_iter", type=int, help="Which iter to test.")
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
    return opt, args


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
    opt, args = parse_options()
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
    model = create_model(opt)
    start_epoch = 0
    current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    logger.info("Save model")
    model.save(0, 0)
    logger.info("Validate")
    model.validation(val_loader, current_iter, tb_logger)
    # logger.info("Test")
    # model.test(test_set, test_loader)
    # model.save_result(0, 0, "test")

    # training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        for train_data in train_loader:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
            )
            # training
            model.feed_data(train_data, train=True)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update({"time": iter_time, "data_time": data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            # save model, validation and test
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Save model")
                model.save(epoch, current_iter)
                logger.info("Validate")
                model.validation(val_loader, current_iter, tb_logger)
                # logger.info("Test")
                # model.test(test_set, test_loader)
                # model.save_result(epoch, current_iter, "test")
            data_time = time.time()
            iter_time = time.time()
        # end of iter
    # end of epoch

    logger.info("Save model")
    model.save(-1, -1)  # -1 stands for the latest
    logger.info("Validate")
    model.validation(val_loader, current_iter, tb_logger)
    # logger.info("Test")
    # model.test(test_set, test_loader)
    # model.save_result(-1, -1, "test")
    if tb_logger is not None:
        tb_logger.close()
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")


if __name__ == "__main__":
    main()
