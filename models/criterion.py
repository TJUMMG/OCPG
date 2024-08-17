import torch
import torch.nn.functional as F
from torch import nn


from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss, proj_loss, masked_ce_loss, generate_box_region_mask, levelset_loss, levelset_loss_video, transform_images, get_images_color_similarity, get_neighbor_images_patch_color_similarity, compute_pairwise_term, compute_pairwise_term_neighbor, generate_point_region_mask)
from einops import rearrange, repeat

class SetCriterion(nn.Module):
    """ This class computes the loss for SgMg.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, args, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 1
        self.mask_out_stride_low = self.mask_out_stride * 2
        self.iter = 0
        self._warmup_iters =100000

    # t*q labels 0/1 indicates whether is a blank frame.
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] 
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')
        # judge the valid frames
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices): 
            valid_ind = valid.nonzero().flatten() 
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        if self.num_classes == 1:
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            pass
        return losses, None, None, None

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        src_boxes = src_boxes.flatten(0, 1)

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses, None, None, None

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        src_masks_low = outputs["pred_masks_low"]
        # future use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                                size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks_low)

        start = int(self.mask_out_stride // 2)
        start_low = int(self.mask_out_stride_low // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks_low = target_masks[:, :, start_low::self.mask_out_stride_low, start_low::self.mask_out_stride_low]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]

        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        self.iter += 1
        src_lst = outputs["ls_features"]    # [b, t, c, h, w]
        target_heatmap, valid = nested_tensor_from_tensor_list([t["weights"] for t in targets], 
                                                            size_divisibility=32, split=False).decompose()
        target_weakmask, valid = nested_tensor_from_tensor_list([t["weak_masks"] for t in targets], 
                                                            size_divisibility=32, split=False).decompose()
        t = target_weakmask.shape[1]
        

        target_heatmap_low = target_heatmap[:, :, start_low::self.mask_out_stride_low, start_low::self.mask_out_stride_low]
        target_heatmap = target_heatmap[:, :, start::self.mask_out_stride, start::self.mask_out_stride]

        target_weakmask_low = target_weakmask[:, :, start_low::self.mask_out_stride_low, start_low::self.mask_out_stride_low]
        target_weakmask = target_weakmask[:, :, start::self.mask_out_stride, start::self.mask_out_stride]

        sizes = torch.stack([t['size'] for t in targets])
        sizes = repeat(sizes, 'b n -> (b t) n', t=t)
        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)
        target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        tgt_boxes_region = generate_box_region_mask(
            target_boxes,
            (im_h, im_w),
            sizes
            )
        tgt_boxes_region = rearrange(tgt_boxes_region, '(b t) h w -> b t h w', t=t)
        
        tgt_boxes_region_low = tgt_boxes_region[:, :, start_low::self.mask_out_stride_low, start_low::self.mask_out_stride_low]
        tgt_boxes_region = tgt_boxes_region[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        warmup_factor = min(float(self.iter) / float(self._warmup_iters), 1.0)

        target_weakmask = target_weakmask * tgt_boxes_region
        target_weakmask_low = target_weakmask_low * tgt_boxes_region_low

        loss_mask, weight = masked_ce_loss(src_masks, target_heatmap, target_weakmask, tgt_boxes_region, num_boxes)
        loss_mask_low, _ = masked_ce_loss(src_masks_low, target_heatmap_low, target_weakmask_low, tgt_boxes_region_low, num_boxes)

        src_masks_scale = F.interpolate(src_masks, src_lst.shape[-2:], mode='bilinear', align_corners=True)
        # ls_loss = levelset_loss_video(src_masks_scale, src_lst, tgt_boxes_region_low)
        # ls_loss_low = levelset_loss_video(src_masks_low, src_lst, tgt_boxes_region_low)
        
        ###
        tgt_boxes_region_scale = F.interpolate(tgt_boxes_region, src_lst.shape[-2:], mode='nearest')
        src_masks_stack = repeat(src_masks_scale, 'b t h w -> (b t) n h w', n=1)
        src_masks_stack_low = repeat(src_masks_low, 'b t h w -> (b t) n h w', n=1)
        tgt_boxes_region_stack = repeat(tgt_boxes_region_scale, 'b t h w -> (b t) n h w', n=1)
        src_lst_stack = rearrange(src_lst, 'b t c h w -> (b t) c h w')[:, :-1]
        ls_loss = levelset_loss(src_masks_stack, src_lst_stack, tgt_boxes_region_stack)
        ls_loss_low = levelset_loss(src_masks_stack_low, src_lst_stack, tgt_boxes_region_stack)
        ###


        losses = {
            "loss_proj": proj_loss(src_masks, tgt_boxes_region, target_weakmask, num_boxes, with_mean_term=True),    
            "loss_mask": (1-warmup_factor) * loss_mask,
            "loss_lst": warmup_factor * ls_loss,
            "loss_proj_low": proj_loss(src_masks_low, tgt_boxes_region_low, target_weakmask_low, num_boxes, with_mean_term=True),
            "loss_mask_low": (1-warmup_factor) * loss_mask_low,
            "loss_lst_low": warmup_factor * ls_loss_low
        }
        
        return losses, src_masks.sigmoid(), target_masks, target_weakmask

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # use hungarian matching results.
        indices = outputs["main_matcher_index"]
        aux_indices = outputs["aux_matcher_index"]

        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T] tensor([1, 1, 0, 0, 0]
        num_boxes = target_valid.sum().item()
        device = outputs['pred_masks_low'].device
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():  # True
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()  # 5

        losses = {}
        src_m, tgt_m = None, None
        for loss in self.losses:
            loss_dic, src_map, tgt_map, weak_map = self.get_loss(loss, outputs, targets, indices, num_boxes)
            losses.update(loss_dic)
            if src_map is not None:
                src_m, tgt_m, weak_m = src_map, tgt_map, weak_map

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer. change outputs->aux_outputs
        if 'aux_outputs' in outputs:
            assert len(aux_indices) == len(outputs['aux_outputs']), "Aux index len not match."
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = aux_indices[i]
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)[0]
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, src_m, tgt_m, weak_m


