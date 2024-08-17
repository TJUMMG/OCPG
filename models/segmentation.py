"""
Segmentaion Part 
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from skimage import color

from einops import rearrange, repeat

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

# import fvcore.nn.weight_init as weight_init

# from .position_encoding import PositionEmbeddingSine1D

BN_MOMENTUM = 0.1

def get_norm(norm, out_channels): # only support GN or LN
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, visual, text,
                text_key_padding_mask: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None,
                visual_pos: Optional[Tensor] = None):
        visual = rearrange(visual, 't h w b c -> (t h w) b c')
        visual2 = self.multihead_attn(query=self.with_pos_embed(visual, visual_pos),
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text, attn_mask=None,
                                   key_padding_mask=text_key_padding_mask)[0]
        visual = visual * visual2
        return visual


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def masked_ce_loss(inputs, weights, masks, box_regions=None, num_boxes=5, alpha=0.7, beta=0.3, thres=0.5):
    # inputs: [m, 1, h, w]
    # masks: [m, 1, h, w]

    weight = weights
    weight[weight>alpha] = alpha
    weight[weight<beta] = beta
    weight = torch.abs(weight - thres)
    weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-5)

    inputs = inputs.sigmoid()
    # foreground = (weights >= alpha).float()
    # background = (weights <= beta).float()
    # uncertain = torch.logical_and(weights > beta, weights < alpha).float()
    # uncertain_weight = torch.abs(weights - thres) + 0.5
    
    if box_regions is not None:
        weight[box_regions==0] = 1
    #     background[torch.where(box_regions == 0)] = 1
    #     foreground[torch.where(box_regions == 0)] = 0
    #     uncertain[torch.where(box_regions == 0)] = 0
    # weight = foreground * 1 + background * 1 + uncertain_weight * uncertain  # [m, h, w]
    loss = F.binary_cross_entropy_with_logits(inputs*weight, masks*weight)
    if torch.isnan(loss):
        print("input nan", torch.isnan(inputs).sum()>0)
        print("masks nan", torch.isnan(masks).sum()>0)
        print("weights nan", torch.isnan(weight).sum()>0)
    return loss, weight


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x**2.0).sum(dim=1) + (target**2.0).sum(dim=1) + eps
    loss = 1.0 - (2 * intersection / union)
    return loss

def generate_mask_from_heatmap(heatmap, thres=0.5):
    background = (
        torch.ones((1, heatmap.shape[-2], heatmap.shape[-1])).to(heatmap.device) * thres
    )
    masks_with_bg = torch.cat([background, heatmap])
    masks = torch.zeros_like(masks_with_bg)
    masks[masks_with_bg.max(dim=0)[1]] = 1
    return masks


def generate_box_region_mask(boxes, mask_size, sizes):
    #  boxes: [m, 4]    sizes: [m, 2]
    boxes_region = torch.zeros((len(boxes), mask_size[0], mask_size[1])).to(
        boxes.device
    )
    for i, (box, size) in enumerate(zip(boxes, sizes)):
        h, w = size
        boxes_scale = (
            box * torch.as_tensor([w, h, w, h]).to(boxes.device)
        ).int()
        if (boxes_scale[3]-boxes_scale[1]) > 0 and (boxes_scale[2]-boxes_scale[0]) > 0:
            boxes_region[
                i, boxes_scale[1] : boxes_scale[3], boxes_scale[0] : boxes_scale[2]
            ] = 1
    return boxes_region

def generate_point_region_mask(points, mask_size, sizes):
    #  boxes: [m, 4]    sizes: [m, 2]
    points_region = torch.zeros((len(points), mask_size[0], mask_size[1])).to(
        points.device
    )
    for i, (point, size) in enumerate(zip(points, sizes)):
        h, w = size
        point_scale = (
            point * torch.as_tensor([w, h]).to(point.device)
        ).int()
        points_region[i, point_scale[1], point_scale[0]] = 1
    return points_region

def proj_loss(inputs, box_regions, masks, num_boxes, with_mean_term=False):
    # inputs: [m, 1, h, w]
    # masks: [m, 1, h, w]
    inputs = inputs.sigmoid()
    mask_losses_y = dice_coefficient(
        inputs.max(dim=2, keepdim=True)[0], box_regions.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        inputs.max(dim=3, keepdim=True)[0], box_regions.max(dim=3, keepdim=True)[0]
    )
    loss_ins_max = (mask_losses_y + mask_losses_x).mean()

    mask_losses_y = dice_coefficient(
        inputs.mean(dim=2, keepdim=True),
        masks.float().mean(dim=2, keepdim=True),
    )
    mask_losses_x = dice_coefficient(
        inputs.mean(dim=3, keepdim=True),
        masks.float().mean(dim=3, keepdim=True),
    )
    loss_ins = (mask_losses_y + mask_losses_x).mean()
    if with_mean_term:
        return loss_ins_max + 0.1 * loss_ins
    else:
        return loss_ins_max

def length_regularization(mask_score):
    gradient_H = torch.abs(mask_score[:, :, 1:, :] - mask_score[:, :, :-1, :])
    gradient_W = torch.abs(mask_score[:, :, :, 1:] - mask_score[:, :, :, :-1])
    curve_length = torch.sum(gradient_H, dim=(1,2,3)) + torch.sum(gradient_W, dim=(1,2,3))
    return curve_length


def region_levelset(mask_score, lst_target):
    '''
    mask_score: predcited mask scores        tensor:(N,2,W,H) 
    lst_target:  input target for levelset   tensor:(N,C,W,H) 
    '''
    mask_score_f = mask_score[:, 0, :, :].unsqueeze(1)
    mask_score_b = mask_score[:, 1, :, :].unsqueeze(1)
    interior_ = torch.sum(mask_score_f * lst_target, (2, 3)) / torch.sum(mask_score_f, (2, 3)).clamp(min=0.00001)
    exterior_ = torch.sum(mask_score_b * lst_target, (2, 3)) / torch.sum(mask_score_b, (2, 3)).clamp(min=0.00001)
    interior_region_level = torch.pow(lst_target - interior_.unsqueeze(-1).unsqueeze(-1), 2)
    exterior_region_level = torch.pow(lst_target - exterior_.unsqueeze(-1).unsqueeze(-1), 2)
    region_level_loss = interior_region_level*mask_score_f + exterior_region_level*mask_score_b
    level_set_loss = torch.sum(region_level_loss, (1, 2, 3))/lst_target.shape[1]
    return level_set_loss


def levelset_loss(mask_logits, targets, box_mask_target):
    mask_logits = mask_logits.sigmoid()
    back_scores = 1.0 - mask_logits
    mask_scores_concat = torch.cat((mask_logits, back_scores), dim=1)

    pixel_num = box_mask_target.sum((1, 2, 3))
    pixel_num = torch.clamp(pixel_num, min=1)

    mask_scores_phi = mask_scores_concat * box_mask_target
    img_target_wbox = targets * box_mask_target

    region_levelset_loss = region_levelset(mask_scores_phi, img_target_wbox) / pixel_num
    length_regu = 0.00001 * length_regularization(mask_scores_phi) / pixel_num
    loss_levelst = region_levelset_loss + length_regu
    return loss_levelst.mean()


def levelset_loss_video(mask_logits, targets, box_mask_target):
    # mask_logits: [b, t, h, w]
    # targets: [b, t, c, h, w]
    # box_mask_target: [b, t, h, w]
    assert mask_logits.shape[1] == targets.shape[1] == box_mask_target.shape[1]
    mask_logits = mask_logits.sigmoid()
    t = mask_logits.shape[1]
    lengths = []
    level_losses = []
    for f_idx_i in range(t):
        mask_logits_i = mask_logits[:, f_idx_i].unsqueeze(1)   # [b, 1, h, w]
        targets_i = targets[:, f_idx_i]   # [b, c, h, w]
        box_mask_target_i = box_mask_target[:, f_idx_i].unsqueeze(1)   # [b, 1, h, w]
        targets_i = targets_i * box_mask_target_i
        mask_logits_i = mask_logits_i * box_mask_target_i
        c1 = torch.sum(mask_logits_i * targets_i, (2, 3)) / torch.sum(mask_logits_i, (2, 3)).clamp(min=0.00001)
        c2 = torch.sum((1 - mask_logits_i) * targets_i, (2, 3)) / torch.sum((1 - mask_logits_i), (2, 3)).clamp(min=0.00001)
        pixel_num = box_mask_target.sum((1, 2, 3))
        pixel_num = torch.clamp(pixel_num, min=1)
        length_regu = 0.00001 * length_regularization(mask_logits_i) / pixel_num
        lengths.append(length_regu.mean())
        for f_idx_j in range(t):
            mask_logits_j = mask_logits[:, f_idx_j].unsqueeze(1)   # [b, 1, h, w]
            targets_j = targets[:, f_idx_j]   # [b, c, h, w]
            box_mask_target_j = box_mask_target[:, f_idx_j].unsqueeze(1)   # [b, 1, h, w]
            targets_j = targets_j * box_mask_target_j
            mask_logits_j = mask_logits_j * box_mask_target_j

            interior_region_level = torch.pow(targets_j - c1.unsqueeze(-1).unsqueeze(-1), 2)
            exterior_region_level = torch.pow(targets_j - c2.unsqueeze(-1).unsqueeze(-1), 2)
            region_level_loss = interior_region_level*mask_logits_j + exterior_region_level*(1-mask_logits_j)
            if f_idx_i == f_idx_j:
                alpha = 1
            else:
                alpha = 0.1
            level_losses.append(alpha * region_level_loss.mean())
    return sum(lengths) / t + sum(level_losses) / (t * t)



def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x

def unfold_w_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    
    return unfolded_x

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def compute_pairwise_term_neighbor(mask_logits, mask_logits_neighbor, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob_neigh = F.logsigmoid(mask_logits_neighbor)
    log_bg_prob_neigh = F.logsigmoid(-mask_logits_neighbor)

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)
    
    log_fg_prob_unfold = unfold_w_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    # print('log_fg_prob shape:', log_fg_prob.shape, 'log_fg_prob unfold:', log_fg_prob_unfold.shape)
    log_bg_prob_unfold = unfold_w_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob_neigh[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob_neigh[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def get_images_color_similarity(images, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)
    return similarity

def get_neighbor_images_color_similarity(images, images_neighbor, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_w_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images_neighbor[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    return similarity

def get_neighbor_images_patch_color_similarity(images, images_neighbor, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_w_center(
        images, kernel_size=kernel_size, dilation= 1 #dilation
    )
    unfolded_images_neighbor = unfold_w_center(
        images_neighbor, kernel_size=kernel_size, dilation= 1 #dilation
    )
    unfolded_images = unfolded_images.flatten(1,2)
    unfolded_images_neighbor = unfolded_images_neighbor.flatten(1,2)
    similarity = get_neighbor_images_color_similarity(unfolded_images, unfolded_images_neighbor, 3, 3) 
    
    return similarity

def transform_images(images):
    # images: [b, t, 3, h, w]
    mean = torch.tensor([0.485, 0.456, 0.406])[None, None, :, None, None].to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, None, :, None, None].to(images.device)
    images = (images * std + mean) * 255
    images = rearrange(images, 'b t c h w -> b t h w c')
    images = torch.as_tensor(color.rgb2lab(images.byte().cpu().numpy()), device=images.device, dtype=torch.float32)
    images = rearrange(images, 'b t h w c -> b t c h w')
    return images

if __name__ == '__main__':
    
    img_rgb = torch.randn((2, 3, 3, 256, 256))
    img_lab = transform_images(img_rgb)
    img_lab_stack = rearrange(img_lab, 'b t c h w -> (b t) c h w')
    lab_sim = [get_images_color_similarity(im.unsqueeze(0), 3, 2) for im in img_lab_stack]
    lab_sim = torch.cat(lab_sim, dim=0)

    src_mask = torch.randn((2, 3, 1, 256, 256))
    tgt_mask = torch.ones((2, 3, 1, 256, 256))
    src_mask_stack = rearrange(src_mask, 'b t c h w -> (b t) c h w')
    pairwise_losses = compute_pairwise_term(src_mask_stack, 3, 2)

    for k, frame in enumerate(img_lab):
        cur_src_mask = src_mask[k]
        cur_tgt_mask = tgt_mask[k]
        # frame: [t, 3, h, w]
        images_lab_sim_nei = [get_neighbor_images_patch_color_similarity(frame[ii].unsqueeze(0), frame[ii+1].unsqueeze(0), 3, 3) for ii in range(0, len(frame), 3)] # change k form 3 to 5, ori is 3, ori dilation is 3
        images_lab_sim_nei1 = [get_neighbor_images_patch_color_similarity(frame[ii].unsqueeze(0), frame[ii+2].unsqueeze(0), 3, 3) for ii in range(0, len(frame), 3)]
        images_lab_sim_nei2 = [get_neighbor_images_patch_color_similarity(frame[ii+1].unsqueeze(0), frame[ii+2].unsqueeze(0), 3, 3) for ii in range(0, len(frame), 3)]
        images_lab_sim_nei = torch.cat(images_lab_sim_nei, dim=0)   # [n, 9, h, w]
        images_lab_sim_nei1 = torch.cat(images_lab_sim_nei1, dim=0)   # [n, 9, h, w]
        images_lab_sim_nei2 = torch.cat(images_lab_sim_nei2, dim=0)   # [n, 9, h, w]
        pairwise_losses_neighbor = compute_pairwise_term_neighbor(
                cur_src_mask[:1], cur_src_mask[1:2], 3, 3
            )
        pairwise_losses_neighbor1 = compute_pairwise_term_neighbor(
            cur_src_mask[:1], cur_src_mask[2:3], 3, 3
        ) 
        pairwise_losses_neighbor2 = compute_pairwise_term_neighbor(
            cur_src_mask[1:2], cur_src_mask[2:3], 3, 3
        )
        cur_tgt_mask_sum = cur_tgt_mask.sum(dim=0, keepdim=True)    # [1, 1, h, w]
        cur_tgt_mask_sum = (cur_tgt_mask_sum > 1.0).float()
        weights_neighbor = (images_lab_sim_nei >= 0.05).float() * cur_tgt_mask_sum
        weights_neighbor1 = (images_lab_sim_nei1 >= 0.05).float() * cur_tgt_mask_sum
        weights_neighbor2 = (images_lab_sim_nei2 >= 0.05).float() * cur_tgt_mask_sum
        loss_pairwise_neighbor = (pairwise_losses_neighbor * weights_neighbor).sum() / weights_neighbor.sum().clamp(min=1.0)
        loss_pairwise_neighbor1 = (pairwise_losses_neighbor1 * weights_neighbor1).sum() / weights_neighbor1.sum().clamp(min=1.0)
        loss_pairwise_neighbor2 = (pairwise_losses_neighbor2 * weights_neighbor2).sum() / weights_neighbor2.sum().clamp(min=1.0)
        