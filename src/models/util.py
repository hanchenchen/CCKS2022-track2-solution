from copy import deepcopy

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from src.utils import get_root_logger, master_only


def load_pretrained_model(net, load_path, strict, param_key, prefix_to_rm):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    net = get_bare_model(net)
    logger = get_root_logger()
    logger.info(f"Loading {net.__class__.__name__} model from {load_path}.")
    load_net = torch.load(load_path, map_location="cpu")
    if param_key is not None:
        load_net = load_net[param_key]
    # remove unnecessary 'module.'
    load_net = remove_prefix(load_net, prefix_to_rm)
    print_different_keys_loading(net, load_net, strict)
    net.load_state_dict(load_net, strict=strict)


def get_bare_model(net):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    return net


def remove_prefix(state_dict, prefix):
    for k, v in deepcopy(state_dict).items():
        if k.startswith(prefix):
            state_dict[k.replace(prefix, "")] = v
            state_dict.pop(k)
    return state_dict


def print_different_keys_loading(crt_net, load_net, strict):
    """Print keys with differnet name or different size when loading models.

    1. Print keys with differnet names.
    2. If strict=False, print the same key but with different tensor size.
        It also ignore these keys with different sizes (not load).

    Args:
        crt_net (torch model): Current network.
        load_net (dict): Loaded network.
        strict (bool): Whether strictly loaded. Default: True.
    """
    crt_net = get_bare_model(crt_net)
    crt_net = crt_net.state_dict()
    crt_net_keys = set(crt_net.keys())
    load_net_keys = set(load_net.keys())

    logger = get_root_logger()
    if crt_net_keys != load_net_keys:
        logger.warning("Current net - loaded net:")
        for v in sorted(list(crt_net_keys - load_net_keys)):
            logger.warning(f"  {v}")
        logger.warning("Loaded net - current net:")
        for v in sorted(list(load_net_keys - crt_net_keys)):
            logger.warning(f"  {v}")

    # check the size for the same keys
    if not strict:
        common_keys = crt_net_keys & load_net_keys
        for k in common_keys:
            if crt_net[k].size() != load_net[k].size():
                logger.warning(
                    f"Size different, ignore [{k}]: crt_net: "
                    f"{crt_net[k].shape}; load_net: {load_net[k].shape}"
                )
                load_net[k + ".ignore"] = load_net.pop(k)


@master_only
def print_network(net):
    """Print the str and parameter number of a network.

    Args:
        net (nn.Module)
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net_cls_str = f"{net.__class__.__name__} - " f"{net.module.__class__.__name__}"
    else:
        net_cls_str = f"{net.__class__.__name__}"

    net = get_bare_model(net)
    net_str = str(net)
    net_params = sum(map(lambda x: x.numel(), net.parameters()))

    logger = get_root_logger()
    logger.info(f"Network: {net_cls_str}, with parameters: {net_params:,d}")
    logger.info(net_str)


def fp32_to_fp16(tensor):
    # deepspeed does not auto cast tensor.
    if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
        tensor = tensor.to(dtype=torch.half)
    return tensor


def model_ema(self, net, net_ema, decay):
    net = get_bare_model(net)

    net_params = dict(net.named_parameters())
    net_ema_params = dict(net_ema.named_parameters())

    for k in net_ema_params.keys():
        net_ema_params[k].data.mul_(decay).add_(net_params[k].data, alpha=1 - decay)

    return net_ema
