from collections import OrderedDict
from os import path as osp

import yaml


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def update_deepspeed_config(opt):
    ds_opt = opt["deepspeed"]
    ds_opt["train_micro_batch_size_per_gpu"] = opt["datasets"]["train"][
        "batch_size_per_gpu"
    ]
    ds_opt["gradient_clipping"] = opt["train"]["grad_clip_norm"]
    ds_opt["steps_per_print"] = 1e9
    opt["deepspeed"] = ds_opt
    return opt


def parse(opt_path):
    """Parse option file.

    Args:
        opt_path (str): Option file path.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode="r") as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt = update_deepspeed_config(opt)

    opt["path"]["root"] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir)
    )
    experiments_root = osp.join(opt["path"]["root"], "experiments", opt["name"])
    opt["path"]["experiments_root"] = experiments_root
    opt["path"]["models"] = osp.join(experiments_root, "models")
    opt["path"]["tb_logger"] = osp.join(experiments_root, "tb_logger")
    opt["path"]["log"] = experiments_root

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = "\n"
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_level * 2) + k + ":["
            msg += dict2str(v, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + k + ": " + str(v) + "\n"
    return msg
