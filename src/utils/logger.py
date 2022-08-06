import datetime
import logging
import sys
import time

import torch.distributed as dist

from .dist_util import master_only


class MessageLogger:
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default: None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt["name"]
        self.interval = opt["logger"]["print_freq"]
        self.start_iter = start_iter
        self.max_iters = opt["train"]["total_iter"]
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop("epoch")
        current_iter = log_vars.pop("iter")
        lrs = log_vars.pop("lrs")

        message = (
            f"[{self.exp_name[:5]}..][epoch:{epoch:3d}, "
            f"iter:{current_iter:8,d}, lr:("
        )
        for v in lrs:
            message += f"{v:.3e},"
        message += ")] "

        # time and estimated time
        if "time" in log_vars.keys():
            iter_time = log_vars.pop("time")
            data_time = log_vars.pop("data_time")

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f"[eta: {eta_str}, "
            message += f"time (data): {iter_time:.3f} ({data_time:.3f})] "

        # other items, especially losses
        for k, v in log_vars.items():
            message += f"{k}: {v:.4e} "
            # tensorboard logger
            if k.startswith("l_"):
                self.tb_logger.add_scalar(f"losses/{k}", v, current_iter)
            else:
                self.tb_logger.add_scalar(k, v, current_iter)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter

    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb

    wandb_id = wandb.util.generate_id()
    wandb.init(
        id=wandb_id,
        entity=opt["logger"]["wandb"]["entity"],
        project=opt["logger"]["wandb"]["project"],
        name=opt["name"],
        config=opt,
        sync_tensorboard=True,
    )

    logger = logging.getLogger()
    logger.info(f"Use wandb logger with id={wandb_id}.")


def get_root_logger(
    logger_name="src", log_level=logging.INFO, log_file=None, initialized=True
):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'src'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
        initialized (bool): Whether the logger has been initialized.

    Returns:
        logging.Logger: The root logger.
    """
    # if the logger has been initialized, just return it
    if initialized:
        return logging.getLogger(logger_name)

    logger = logging.getLogger(logger_name)
    format_str = "%(asctime)s %(levelname)s: %(message)s"
    # StreamHandler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(handler)
    logger.propagate = False  # https://zhuanlan.zhihu.com/p/25920526

    rank = dist.get_rank()
    if rank != 0:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(log_level)
        # FileHandler
        handler = logging.FileHandler(log_file, "w")
        handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(handler)

    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    msg = (
        "\nVersion Information: "
        f"\n\tPyTorch: {torch.__version__}"
        f"\n\tTorchVision: {torchvision.__version__}"
    )
    return msg
