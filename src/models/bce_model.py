import json
import os
from collections import OrderedDict
from copy import deepcopy

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from src.archs import define_network
from src.metrics import cal_metric
from src.models import lr_scheduler
from src.models.base_model import BaseModel
from src.utils import get_root_logger, master_only

from .util import fp32_to_fp16, print_network


class BCEModel(BaseModel):
    def __init__(self, opt, train_set, val_set, test_set):
        super(BCEModel, self).__init__(opt)

        # define network
        self.net = define_network(deepcopy(opt["network"]))

        self.net = self.net.to(self.device)
        self.init_training_settings()

        self.net, self.optimizer, _, _ = deepspeed.initialize(
            config_params=opt["deepspeed"],
            model=self.net,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
        )
        print_network(self.net)

        # load pretrained models
        load_path = opt["path"]["pretrain_network"]
        if load_path is not None:
            self.net.load_checkpoint(
                os.path.dirname(os.path.dirname(load_path)),
                os.path.basename(os.path.dirname(load_path)),
            )

        self.transform_train = train_set.transform
        self.transform_val = val_set.transform

    def init_training_settings(self):
        self.net.train()
        self.setup_optimizer()
        self.setup_scheduler()

    def setup_optimizer(self):
        train_opt = self.opt["train"]
        optim_type = train_opt["optim"].pop("type")
        if optim_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), **train_opt["optim"]
            )
        elif optim_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), **train_opt["optim"]
            )
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")

    def setup_scheduler(self):
        """Set up schedulers."""
        train_opt = self.opt["train"]
        scheduler_type = train_opt["scheduler"].pop("type")
        if scheduler_type in ["MultiStepLR", "MultiStepRestartLR"]:
            self.scheduler = lr_scheduler.MultiStepRestartLR(
                self.optimizer, **train_opt["scheduler"]
            )
        elif scheduler_type == "CosineAnnealingRestartLR":
            self.scheduler = lr_scheduler.CosineAnnealingRestartLR(
                self.optimizer, **train_opt["scheduler"]
            )
        else:
            raise NotImplementedError(
                f"Scheduler {scheduler_type} is not implemented yet."
            )

    def feed_data(self, data, train):
        if "img1" in data:
            self.img1 = data["img1"].to(self.device, non_blocking=True).float() / 255.0
            if train:
                self.img1 = self.transform_train(self.img1)
            else:
                self.img1 = self.transform_val(self.img1)
            if self.opt["deepspeed"]["fp16"]["enabled"]:
                self.img1 = fp32_to_fp16(self.img1)
        if "img2" in data:
            self.img2 = data["img2"].to(self.device, non_blocking=True).float() / 255.0
            if train:
                self.img2 = self.transform_train(self.img2)
            else:
                self.img2 = self.transform_val(self.img2)
            if self.opt["deepspeed"]["fp16"]["enabled"]:
                self.img2 = fp32_to_fp16(self.img2)
        if "txt1" in data:
            self.txt1 = {}
            for k in data["txt1"]:
                self.txt1[k] = (
                    data["txt1"][k].to(self.device, non_blocking=True).squeeze(1)
                )
                if self.opt["deepspeed"]["fp16"]["enabled"]:
                    self.txt1[k] = fp32_to_fp16(self.txt1[k])
        if "txt2" in data:
            self.txt2 = {}
            for k in data["txt2"]:
                self.txt2[k] = (
                    data["txt2"][k].to(self.device, non_blocking=True).squeeze(1)
                )
                if self.opt["deepspeed"]["fp16"]["enabled"]:
                    self.txt2[k] = fp32_to_fp16(self.txt2[k])
        if "label" in data:
            self.label = data["label"].to(self.device, non_blocking=True)
            if self.opt["deepspeed"]["fp16"]["enabled"]:
                self.label = fp32_to_fp16(self.label)

    def forward(self):
        feat11, feat12 = self.net(self.txt1, self.img1)
        feat21, feat22 = self.net(self.txt2, self.img2)

        logit1 = (feat11 * feat21).sum(dim=1)
        logit2 = (feat12 * feat22).sum(dim=1)

        logit = torch.stack([logit1, logit2], dim=1)

        feat1 = torch.cat([feat11, feat12], dim=1)
        feat2 = torch.cat([feat21, feat22], dim=1)

        return logit, feat1, feat2

    def optimize_parameters(self):
        self.optimizer.zero_grad()

        l_total = 0
        loss_dict = OrderedDict()

        logit, _, _ = self.forward()

        l_bce = F.cross_entropy(logit, self.label)
        l_total += l_bce
        loss_dict["l_bce"] = l_bce

        acc, P, R, f1 = cal_metric(logit, self.label.long())
        loss_dict["f1"] = f1 * torch.ones_like(l_bce)
        loss_dict["P"] = P * torch.ones_like(l_bce)
        loss_dict["R"] = R * torch.ones_like(l_bce)
        loss_dict["acc"] = acc * torch.ones_like(l_bce)

        self.net.backward(l_total)
        self.net.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad()
    def dist_validation(self, dataloader, tb_logger):
        logits = torch.zeros((10000, 2)).to(self.device)
        if self.opt["deepspeed"]["fp16"]["enabled"]:
            logits = fp32_to_fp16(logits)
        labels = torch.zeros((10000)).to(self.device).long()
        valid = torch.zeros((10000)).to(self.device).long()
        last = 0
        self.net.eval()
        for i, data in tqdm(enumerate(dataloader)):
            if i % dist.get_world_size() != dist.get_rank():
                continue
            self.feed_data(data, train=False)
            logit, _, _ = self.forward()
            b = logit.shape[0]
            logits[last : last + b] = logit
            labels[last : last + b] = self.label
            valid[last : last + b] = 1
            last = last + b
        self.net.train()

        dist.barrier()
        logits_list = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
        dist.all_gather(logits_list, logits)
        logits = torch.cat(logits_list)
        labels_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
        dist.all_gather(labels_list, labels)
        labels = torch.cat(labels_list)
        valid_list = [torch.zeros_like(valid) for _ in range(dist.get_world_size())]
        dist.all_gather(valid_list, valid)
        valid = torch.cat(valid_list)
        logits = logits[valid == 1]
        labels = labels[valid == 1]
        print(1, logits.shape, labels.shape)

        acc, P, R, f1 = cal_metric(logits, labels)
        logger = get_root_logger()
        logger.info(f"f1: {f1:.4f}, P: {P:.4f}, R: {R:.4f}, acc: {acc:.4f}")
        if dist.get_rank() == 0:
            tb_logger.add_scalar(f"metrics/f1", f1, self.current_iter)
            tb_logger.add_scalar(f"metrics/P", P, self.current_iter)
            tb_logger.add_scalar(f"metrics/R", R, self.current_iter)
            tb_logger.add_scalar(f"metrics/acc", acc, self.current_iter)

    @torch.no_grad()
    def test(self, dataset, dataloader):
        feats1 = []
        feats2 = []
        ids1 = []
        ids2 = []
        self.net.eval()
        for data in tqdm(dataloader):
            self.feed_data(data, train=False)
            _, feat1, feat2 = self.forward()
            feats1.append(feat1)
            feats2.append(feat2)
            ids1 += data["id1"]
            ids2 += data["id2"]
        self.net.train()
        feats1 = torch.cat(feats1).tolist()
        feats2 = torch.cat(feats2).tolist()

        self.test_result = []
        for i in range(len(feats1)):
            d = {}
            d["src_item_id"] = ids1[i]
            d["src_item_emb"] = str(feats1[i])
            d["tgt_item_id"] = ids2[i]
            d["tgt_item_emb"] = str(feats2[i])
            d["threshold"] = float(0)
            self.test_result.append(d)
        print(2, len(self.test_result))

    def save_network(self):
        self.net.save_checkpoint(self.opt["path"]["models"])

    @master_only
    def save_result(self, label, model_path):
        result_path = model_path.replace(".pt", f"_{label}_result.jsonl")
        f = open(result_path, "w")
        for line in self.test_result:
            f.write(json.dumps(line))
            f.write("\n")
        f.close()
