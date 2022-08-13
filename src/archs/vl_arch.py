import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, SwinModel
from transformers.models.swin.modeling_swin import SwinLayer

from .vl_arch_util import get_attn_mask


class VLArch(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if "fp16" in opt and opt["fp16"]:
            # Hack SwinLayer.get_attn_mask() to support FP16.
            SwinLayer.get_attn_mask = get_attn_mask

        self.txt_encoder1 = BertModel.from_pretrained(opt["bert"])
        self.txt_encoder1.pooler = nn.Identity()
        self.img_encoder1 = SwinModel.from_pretrained(opt["vit"])
        # There is LayerNorm between Transformer output and AvgPool1d in Swin.
        dim1 = (
            self.txt_encoder1.config.hidden_size + self.img_encoder1.config.hidden_size
        )
        self.fc1 = nn.Linear(dim1, 512)

        self.txt_encoder2 = BertModel.from_pretrained(opt["bert"])
        self.txt_encoder2.pooler = nn.Identity()
        self.img_encoder2 = SwinModel.from_pretrained(opt["vit"])
        dim2 = (
            self.txt_encoder2.config.hidden_size + self.img_encoder2.config.hidden_size
        )
        self.fc2 = nn.Linear(dim2, 512)

    def forward(self, txt, img):
        txt_feat1 = self.txt_encoder1(**txt).pooler_output[:, 0]
        txt_feat2 = self.txt_encoder2(**txt).pooler_output[:, 0]
        img_feat1 = self.img_encoder1(pixel_values=img).pooler_output
        img_feat2 = self.img_encoder2(pixel_values=img).pooler_output

        feat1 = torch.cat([txt_feat1, img_feat1], dim=1)
        feat2 = torch.cat([txt_feat2, img_feat2], dim=1)

        feat1 = self.fc1(feat1)
        feat2 = self.fc2(feat2)

        return feat1, feat2
