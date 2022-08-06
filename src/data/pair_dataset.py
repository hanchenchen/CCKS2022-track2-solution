import os

import numpy as np
import torch
from torchvision.io import read_image
from transformers import BertTokenizer

from .util import clean_str, read_info, read_pair, str2dict


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt["phase"]

        self.pair = read_pair(opt["pair_path"])
        self.info = read_info(opt["info_path"])
        for item_id in self.info:
            name = self.info[item_id]["item_image_name"]
            name = os.path.splitext(name)[0] + ".jpg"
            path = os.path.join(opt["image_dir"], name)
            self.info[item_id]["path"] = path
        self.tokenizer = BertTokenizer.from_pretrained(opt["tokenizer"])
        self.max_len = self.opt["max_len"]
        self.key_ps = ["颜色分类", "货号", "型号", "品牌", "尺寸", "口味", "品名", "批准文号", "系列", "尺码"]

    def __len__(self):
        return len(self.pair)

    def get_img(self, info):
        path = info["path"]
        img = read_image(path)
        return img

    def get_pvs(self, info):
        for k in ["title", "item_pvs", "sku_pvs"] + self.key_ps:
            if k not in info:
                info[k] = ""
            info[k] = clean_str(info[k])
        info["pvs"] = info["item_pvs"] + ";" + info["sku_pvs"]
        info = {**info, **str2dict(info["pvs"])}
        return info

    def get_txt(self, info):
        txt = [info[k] for k in ["title"] + self.key_ps]
        # print(1, txt)
        txt = [x for x in txt if x != ""]
        # print(2, txt)
        txt = ";".join(txt)
        # print(3, txt)
        txt1 = self.tokenizer(
            txt,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        # txt2 = self.tokenizer(
        #     [info[k] for k in self.key_ps],
        #     return_tensors="pt",
        #     max_length=10,
        #     padding="max_length",
        #     truncation=True,
        # )
        # for k in ["input_ids", "token_type_ids", "attention_mask"]:
        #     txt1[k] = torch.cat((txt1[k], txt2[k].view(1, -1)), dim=1)
        return txt1

    def __getitem__(self, idx):
        ret = {}

        id1 = self.pair[idx]["src_item_id"]
        ret.update({"id1": id1})
        info1 = self.get_pvs(self.info[id1])

        id2 = self.pair[idx]["tgt_item_id"]
        ret.update({"id2": id2})
        info2 = self.get_pvs(self.info[id2])

        # if self.phase == "train":
        #     for k in self.key_ps:
        #         if np.random.uniform() < 0.5:
        #             info1[k], info2[k] = info2[k], info1[k]

        ret.update({"txt1": self.get_txt(info1)})
        ret.update({"txt2": self.get_txt(info2)})
        ret.update({"img1": self.get_img(info1)})
        ret.update({"img2": self.get_img(info2)})

        if self.phase != "test":
            label = float(self.pair[idx]["item_label"])
            ret.update({"label": label})

        return ret
