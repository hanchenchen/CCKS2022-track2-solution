import torch
from transformers.models.swin.modeling_swin import window_partition


def get_attn_mask(self, height, width):
    if self.shift_size > 0:
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, height, width, 1)).half()
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                img_mask[:, height_slice, width_slice, :] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
    else:
        attn_mask = None
    return attn_mask
