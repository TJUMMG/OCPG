import torch
import torch.nn as nn
import torchvision
try:
    from frozen_batchnorm2d import FrozenBatchNorm2d
except:
    from .frozen_batchnorm2d import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter
import math
import numpy as np
import torch.nn.functional as F


class SimModel(nn.Module):
    def __init__(self, backbone, dilation=False, background_thres=0.5):
        super().__init__()
        self.background_thres = background_thres
        backbone = getattr(torchvision.models, backbone)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
        )
        return_layers = {
            "layer1": "feat1",
            "layer2": "feat2",
            "layer3": "feat3",
            "layer4": "feat4",
        }
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        checkpoint_path = "checkpoints/densecl_r101_imagenet_200ep.pth"
        self.backbone.load_state_dict(
            torch.load(checkpoint_path)["state_dict"], strict=False
        )

    def forward_point(self, x, point_list, valid):
        fea = self.backbone(x)["feat4"]
        keys = fea  ### [B, C3, H3, W3]
        b, c, h, w = keys.shape
        out_masks = []
        for n_p, point_loc in enumerate(point_list):
            if valid[n_p]:
                scale_factors = [1.0]
                queries_list = []
                for scale_factor in scale_factors:
                    point_cur = [
                        int(point_loc[0] * w * scale_factor),
                        int(point_loc[1] * h * scale_factor),
                    ]
                    cur_queries = keys[:, :, point_cur[1], point_cur[0]]
                    queries_list.append(cur_queries)

                queries = torch.stack(queries_list, dim=1)

                keys = keys / keys.norm(dim=1, keepdim=True)
                queries = queries / queries.norm(dim=-1, keepdim=True)
                attn = torch.matmul(queries, keys.view(b, c, -1))
                attn = (attn - attn.min(-1, keepdim=True)[0]) / attn.max(
                    -1, keepdim=True
                )[0]
                soft_masks = attn.reshape(b, attn.shape[1], h, w)
                out_masks.append(soft_masks)
            else:
                out_masks.append(torch.zeros((b, 1, h, w)).to(x.device))
        out_masks = torch.cat(out_masks, dim=1)
        return out_masks

    def forward_bbox(self, x, bbox_list, valid):
        h_ori, w_ori = x.shape[-2:]
        fea = self.backbone(x)["feat4"]
        keys = fea  ### [B, C3, H3, W3]
        b, c, h, w = keys.shape
        out_masks = []
        for n_b, bbox in enumerate(bbox_list):
            if valid[n_b]:

                scale_factors = [1.0]
                queries_list = []
                bbox_masks = []
                for scale_factor in scale_factors:
                    box_cur = [
                        int(bbox[0] / w_ori * w * scale_factor),
                        int(bbox[1] / h_ori * h * scale_factor),
                        int(bbox[2] / w_ori * w * scale_factor),
                        int(bbox[3] / h_ori * h * scale_factor),
                    ]
                    bbox_mask = torch.zeros((h, w)).bool().to(x.device)
                    bbox_mask[box_cur[1] : box_cur[3], box_cur[0] : box_cur[2]] = True
                    range_x = list(range(box_cur[0], box_cur[2] + 1))
                    range_y = list(range(box_cur[1], box_cur[3] + 1))
                    i = 1
                    while(len(range_x) * len(range_y) > 256):
                        range_x = list(range(box_cur[0], box_cur[2] + 1, i+1))
                        range_y = list(range(box_cur[1], box_cur[3] + 1, i+1))
                        i += 1
                    x_candi = torch.tensor(range_x)
                    y_candi = torch.tensor(range_y)
                    gridx, gridy = torch.meshgrid(x_candi, y_candi)
                    locs = torch.stack([gridx, gridy], dim=-1).flatten(0, 1)  # [N, 2]
                    for loc in locs:
                        cur_queries = keys[:, :, loc[1], loc[0]]
                        queries_list.append(cur_queries)
                        bbox_masks.append(bbox_mask)
                queries = torch.stack(queries_list, dim=1)  # [b, n, d]
                bbox_masks = torch.stack(bbox_masks, dim=0)[None]  # [1, n, h, w]
                bbox_masks_flatten = bbox_masks.flatten(-2)

                keys = keys / keys.norm(dim=1, keepdim=True)
                queries = queries / queries.norm(dim=-1, keepdim=True)
                attn = torch.matmul(queries, keys.view(b, c, -1))
                attn = (attn - attn.min(-1, keepdim=True)[0]) / attn.max(
                    -1, keepdim=True
                )[0]

                attn_reshape = attn.reshape(b, attn.shape[1], h, w)

                attn_scale = attn_reshape
                attn_x = attn_scale.max(dim=-2)[0]
                attn_y = attn_scale.max(dim=-1)[0]

                score_x = (attn_x * bbox_masks.max(dim=-2)[0]).sum(dim=-1) / ((attn_x + bbox_masks.max(dim=-2)[0] - attn_x * bbox_masks.max(dim=-2)[0]).sum(dim=-1) + 1e-5)
                score_y = (attn_y * bbox_masks.max(dim=-1)[0]).sum(dim=-1) / ((attn_y + bbox_masks.max(dim=-1)[0] - attn_y * bbox_masks.max(dim=-1)[0]).sum(dim=-1) + 1e-5)
                score = (score_x + score_y) / 2

                _, max_loc = torch.topk(score, 1, 1)
                attn_selected = torch.gather(
                    attn, 1, max_loc.unsqueeze(-1).repeat(1, 1, attn.shape[-1])
                )
                
                soft_masks = attn_selected.reshape(b, attn_selected.shape[1], h, w)
                out_masks.append(soft_masks)
            else:
                out_masks.append(torch.zeros((b, 1, h, w)).to(x.device))
        out_masks = torch.cat(out_masks, dim=1)
        return out_masks

    def forward(self, x, query_list, valid, mode="point"):
        if mode == "point":
            out_masks = self.forward_point(x, query_list, valid)
        elif mode == "bbox":
            out_masks = self.forward_bbox(x, query_list, valid)
        return out_masks
