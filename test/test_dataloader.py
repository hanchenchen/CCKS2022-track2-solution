import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import create_dataset
from src.utils import parse

opt = parse("options/63_grad_clip_norm_0.5.yml")
for phase in opt["datasets"]:
    opt["datasets"][phase]["phase"] = phase

dataset_opt = opt["datasets"]["train"]
train_set = create_dataset(dataset_opt)
train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
label = -1
for idx, data in tqdm(enumerate(train_loader)):
    # print(idx)
    # print(data["img1"].shape, data["img1"].max(), data["img1"].min())
    # print(data["label"])
    # break
    pass

dataset_opt = opt["datasets"]["val"]
val_set = create_dataset(dataset_opt)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
label = -1
for idx, data in tqdm(enumerate(val_loader)):
    # print(idx)
    # print(data["img1"].shape, data["img1"].max(), data["img1"].min())
    # print(data["img2"].shape, data["img2"].max(), data["img2"].min())
    # print(data["label"])
    # break
    pass

dataset_opt = opt["datasets"]["test"]
test_set = create_dataset(dataset_opt)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
for idx, data in tqdm(enumerate(test_loader)):
    # print(idx)
    # print(data["img1"].shape, data["img1"].max(), data["img1"].min())
    # print(data["img2"].shape, data["img2"].max(), data["img2"].min())
    # break
    pass
