import torch
import torch.distributed as dist

from src.data import create_dataset_dataloader
from src.models import create_model
from src.train import parse_options


def main():
    opt = parse_options()
    seed = opt["manual_seed"]

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

    # create model
    assert opt["path"]["pretrain_network"] is not None
    model = create_model(opt, train_set, val_set, test_set)
    model.test(val_set, val_loader)
    model.save_result("test_B", opt["path"]["pretrain_network"])


if __name__ == "__main__":
    main()
