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

    opt["path"]["root"] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir)
    )
    experiments_root = osp.join(opt["path"]["root"], "experiments", opt["name"])
    opt["path"]["experiments_root"] = experiments_root
    opt["path"]["models"] = osp.join(experiments_root, "models")
    opt["path"]["tb_logger"] = osp.join(experiments_root, "tb_logger")
    opt["path"]["log"] = experiments_root

    if "debug" in opt["name"]:
        opt["train"]["total_iter"] = 20
        opt["logger"]["print_freq"] = 1
        opt["logger"]["save_checkpoint_freq"] = 10
        opt["logger"]["wandb"] = None
        # for k in opt["datasets"]:
        #     opt["datasets"][k]["num_worker_per_gpu"] = 0
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
