import torch
import torch.distributed as dist

from src.data import create_dataset_dataloader
from src.models import create_model
from src.train import parse_options


def main():
    opt, args = parse_options()
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
    model = create_model(opt)
    model_path = args.model_path
    strict_load = args.strict_load
    test_iter = args.test_iter
    model.load_network(model.net, model_path, strict_load)
    model.test(test_set, test_loader)
    model.save_result(test_iter, test_iter, "test_B", model_path)


if __name__ == "__main__":
    main()
