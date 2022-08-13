from collections import OrderedDict

import torch


class BaseModel:
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda")
        self.current_iter = 0
        self.current_epoch = 0

    def set_current_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    def set_current_iter(self, current_iter):
        self.current_iter = current_iter

    def get_current_log(self):
        return self.log_dict

    def get_current_learning_rate(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            keys = []
            losses = []
            for name, value in loss_dict.items():
                keys.append(name)
                losses.append(value)
            losses = torch.stack(losses, 0)
            torch.distributed.reduce(losses, dst=0)
            if self.opt["rank"] == 0:
                losses /= self.opt["world_size"]
            loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
