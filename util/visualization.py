import cv2
import numpy as np
import torch


def generate_ce_weight(heatmap, size, box=None, alpha=0.7, beta=0.3, thres=0.5):
    weight = heatmap
    weight[weight>alpha] = alpha
    weight[weight<beta] = beta
    weight = np.abs(weight - thres)
    weight = (weight - weight.min()) / (weight.max() - weight.min())

    # foreground = (heatmap >= alpha).astype(float)
    # background = (heatmap <= beta).astype(float)
    # uncertain = np.logical_and(heatmap > beta, heatmap < alpha).astype(float)
    # uncertain_weight = np.abs(heatmap - thres)  + 0.5
    if box is not None:
        box_regions = np.zeros_like(heatmap)
        h, w = size
        box = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2])
        boxes_scale = (box * np.array([w, h, w, h])).astype(int)
        box_regions[
            boxes_scale[1] : boxes_scale[3], boxes_scale[0] : boxes_scale[2]
        ] = 1
        weight[box_regions==0] = 1
        # background[np.where(box_regions == 0)] = 1
        # foreground[np.where(box_regions == 0)] = 0
        # uncertain[np.where(box_regions == 0)] = 0
    # weight = foreground * 1 + background * 1 + uncertain_weight * uncertain
    # weight = box_regions
    return weight


def generate_mask_from_heatmap(heatmap, thres=0.5):
    background = np.ones((1, heatmap.shape[-2], heatmap.shape[-1])) * thres
    masks_with_bg = np.concatenate([background, heatmap])
    masks = np.zeros_like(masks_with_bg)
    max_idx = np.argmax(masks_with_bg, axis=0)
    for i in range(masks.shape[0]):
        masks[i, max_idx == i] = 1
    return masks[1:]


def viz_heatmap(heatmap, rgb_img=None):
    # heatmap: [h, w] \in [0,1]  rgb_img: [h, w, 3] np.array
    out_img = rgb_img.copy()
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, 11)
    out_img = cv2.addWeighted(out_img, 1, heatmap, 0.6, 1)
    return out_img


def viz_bbox(bbox, size, rgb_img):
    # bbox: [4] \in [0,1]  rgb_img: [h, w, 3] np.array
    out_img = rgb_img.copy()
    h, w = size
    x_c, y_c, bw, bh = bbox
    bbox_xyxy = np.array(
        [(x_c - 0.5 * bw), (y_c - 0.5 * bh), (x_c + 0.5 * bw), (y_c + 0.5 * bh)]
    )
    bbox_scale = (bbox_xyxy * np.array([w, h, w, h])).astype(int)
    out_img = cv2.rectangle(
        out_img,
        (bbox_scale[0], bbox_scale[1]),
        (bbox_scale[2], bbox_scale[3]),
        (255, 0, 0),
        3,
    )
    return out_img


def viz_point(point, size, rgb_img):
    # point: [2] \in [0,1]  rgb_img: [h, w, 3] np.array
    out_img = rgb_img.copy()
    h, w = size
    point_scale = (point * np.array([w, h])).astype(int)
    out_img = cv2.circle(out_img, (point_scale[0], point_scale[1]), 3, (255, 0, 0), -1)
    return out_img


def viz_mask(mask, rgb_img=None):
    # mask: [h, w] \in {0,1}  rgb_img: [h, w, 3] np.array
    out_img = rgb_img.copy()
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask_color[:, :, 0] = mask * 255
    out_img = cv2.addWeighted(out_img, 1, mask_color.astype(np.uint8), 0.6, 1)
    return out_img


def img_recover(img):
    # img: tensor [3, h, w]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(img.device)
    img_recovered = (img * std + mean) * 255
    img_recovered = img_recovered.byte().permute(1, 2, 0).cpu().numpy()
    return img_recovered


def visualize(samples, targets):
    viz_dict = {}
    for i, (frames, target) in enumerate(zip(samples.tensors, targets)):
        h, w = target["size"].numpy()
        if "valid_indices" in target.keys():
            valid_frame = frames.index_select(0, target["valid_indices"])  # [1, 3, h, w]
            frames = valid_frame[:, :, :h, :w]
        else:
            frames = frames[:, :, :h, :w]
        for frame_id, frame in enumerate(frames):
            rgb_frame = img_recover(frame)  # [h, w, 3] np.array
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            masks, boxes, weak_masks, weights = (
                target["masks"].cpu().numpy(),
                target["boxes"].cpu().numpy(),
                target["weak_masks"].cpu().numpy(),
                target["weights"].cpu().numpy(),
            )
            # weak_mask_box = generate_mask_from_heatmap(heat_bbox)
            # weak_mask_point = generate_mask_from_heatmap(heat_point)
            for j, (mask, box, weight, weak_m) in enumerate(
                zip(
                    masks,
                    boxes,
                    weights,
                    weak_masks
                )
            ):
                weight_p = generate_ce_weight(weight, (h, w), box)
                img_masked = viz_mask(mask, rgb_frame)
                img_bbox = viz_bbox(box, (h, w), rgb_frame)
                img_masked_weak = viz_mask(weak_m, rgb_frame)
                img_heat = viz_heatmap(weight, rgb_frame)
                img_heat_p = viz_heatmap(weight_p, rgb_frame)
                final_viz = np.concatenate(
                    [
                        img_masked,
                        img_bbox,
                        img_heat,
                        img_masked_weak,
                        img_heat_p
                    ],
                    axis=1,
                )
                viz_dict["batch{}_frame{}_instance{}".format(i, frame_id, j)] = final_viz

    return viz_dict
