import os

import cv2
import torch
import torch.nn.functional as F
import h5py
import time

from data import img_transform, load_img_davis, load_video_a2d
from sim_model import SimModel
import numpy as np
from torchvision import transforms
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax  # y1, y2, x1, x2


def visualize_generated_anno(video_path, anno_path, vid, frame_id):
    save_path = "result/viz/a2d/{}_{}".format(vid, frame_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    annos = h5py.File(os.path.join(anno_path, vid, "{:05d}.h5".format(frame_id + 1)))
    cap = cv2.VideoCapture(os.path.join(video_path, vid + ".mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    flag, frame = cap.read()
    assert flag
    heatmap_bbox = np.array(annos["heatBBox"])
    heatmap_point = np.array(annos["heatPoint"])
    instances = list(annos["instance"])
    instance_masks = np.array(annos["reMask"])
    if len(instances) == 1:
        instance_masks = instance_masks[np.newaxis, ...]
    gt = instance_masks.swapaxes(1, 2)
    for i in range(len(instance_masks)):
        heat_color_point = (heatmap_point[i] * 255).astype(np.uint8)
        heat_color_point = cv2.applyColorMap(heat_color_point, colormap=11)
        heat_color_bbox = (heatmap_bbox[i] * 255).astype(np.uint8)
        heat_color_bbox = cv2.applyColorMap(heat_color_bbox, colormap=11)
        gt_color = (gt[i] * 255).astype(np.uint8)
        gt_color = np.stack(
            [np.zeros_like(gt_color), np.zeros_like(gt_color), gt_color], axis=-1
        )
        out_heat_point = cv2.addWeighted(frame, 1, heat_color_point, 0.6, 1)
        out_heat_bbox = cv2.addWeighted(frame, 1, heat_color_bbox, 0.6, 1)
        out_gt = cv2.addWeighted(frame, 1, gt_color, 0.6, 1)
        cv2.imwrite(os.path.join(save_path, "gt_{}.jpg".format(i)), out_gt)
        cv2.imwrite(
            os.path.join(save_path, "heat_point_{}.jpg".format(i)), out_heat_point
        )
        cv2.imwrite(
            os.path.join(save_path, "heat_bbox_{}.jpg".format(i)), out_heat_bbox
        )


@torch.no_grad()
def generate_mask(video_path, anno_path, save_path, model, cuda=True):
    # save_path = anno_path.replace(
    #     "a2d_annotation_with_instances", "a2d_annotation_with_instances_weakly_test"
    # )
    video_ids = os.listdir(anno_path)

    all_tp = []
    all_tb = []

    for vid in tqdm(video_ids, total=len(video_ids)):
        save_path_vid = os.path.join(save_path, vid)
        if not os.path.exists(save_path_vid):
            os.makedirs(save_path_vid)
        frame_ids = os.listdir(os.path.join(anno_path, vid))
        cap = cv2.VideoCapture(os.path.join(video_path, vid + ".mp4"))
        idx = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            if "{:05d}.h5".format(idx + 1) in frame_ids:
                ret, frame = cap.retrieve()
                h, w, _ = frame.shape
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                annos = h5py.File(
                    os.path.join(anno_path, vid, "{:05d}.h5".format(idx + 1))
                )
                # out_anno = deepcopy(annos)
                instances = list(annos["instance"])
                instance_masks = np.array(annos["reMask"])
                if len(instances) == 1:
                    instance_masks = instance_masks[np.newaxis, ...]
                instance_masks = instance_masks.swapaxes(1, 2) * 255

                centers = []
                bboxes = []
                centers_norm = []
                instance_valid = []
                for instance_mask in instance_masks:
                    if (instance_mask > 0).any():
                        dist = cv2.distanceTransform(
                            instance_mask, cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL
                        )
                        _, _, _, center = cv2.minMaxLoc(dist)
                        center_norm = (center[0] / w, center[1] / h)
                        y1, y2, x1, x2 = bounding_box(instance_mask)
                        bbox = np.array([x1, y1, x2, y2])
                        bbox[0::2] = np.clip(bbox[0::2], 0, w)
                        bbox[1::2] = np.clip(bbox[1::2], 0, h)
                        bboxes.append(bbox)
                        centers.append(center)
                        centers_norm.append(center_norm)
                        instance_valid.append(1)
                    else:
                        bboxes.append(np.array([0, 0, 0, 0]))
                        centers.append([0, 0])
                        centers_norm.append([0, 0])
                        instance_valid.append(0)
                if cuda:
                    frame = frame.cuda()
                t0 = time.time()
                masks_point = model(frame[None], centers_norm, instance_valid, "point")
                t1 = time.time()
                masks_bbox = model(frame[None], bboxes, instance_valid, "bbox")
                all_tb.append(time.time()-t1)
                all_tp.append(t1-t0)

                masks_point = F.interpolate(
                    masks_point, (h, w), mode="bilinear", align_corners=True
                )
                masks_bbox = F.interpolate(
                    masks_bbox, (h, w), mode="bilinear", align_corners=True
                )
                masks_point = masks_point[0].cpu().numpy()
                masks_bbox = masks_bbox[0].cpu().numpy()
                assert masks_point.shape == masks_bbox.shape == instance_masks.shape
                out_annos = h5py.File(
                    os.path.join(save_path_vid, "{:05d}.h5".format(idx + 1)), "w"
                )
                for k in annos.keys():
                    out_annos.create_dataset(k, data=annos[k])

                out_annos.create_dataset("heatBBox", data=masks_bbox)
                out_annos.create_dataset("heatPoint", data=masks_point)
                out_annos.create_dataset("centerPoint", data=centers)

                annos.close()
                out_annos.close()
            idx += 1
        cap.release()
    all_t = [sum(all_tp), sum(all_tb), len(all_tp)/sum(all_tp), len(all_tp)/sum(all_tb)]
    print(all_t)


if __name__ == "__main__":
    video_path = "/media/HardDisk_B/Users/wx/wwk_files/datasets/referring_video_segmentation/a2d-sentences/Release/clips320H"
    anno_path = "/media/HardDisk_B/Users/wx/wwk_files/datasets/referring_video_segmentation/a2d-sentences/text_annotations/a2d_annotation_with_instances"
    save_path = "./anno_weak/a2d/a2d_annotation_with_instances_weakly"
    dilation = False
    cuda = True

    model = SimModel("resnet101", dilation)
    if cuda:
        model.cuda()
    generate_mask(video_path, anno_path, save_path, model)
    # visualize_generated_anno(video_path, save_path, "bicxykHGpic", 34)
