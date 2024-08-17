import os

import cv2
import torch
import torch.nn.functional as F
import h5py

from data import img_transform, load_img_davis, load_video_a2d
from sim_model import SimModel
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import json
from PIL import Image

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

def transform_anno_to_each_frame(meta_path, exp_meta_path):
    anno_dict = json.load(open(meta_path))
    exp_dict = json.load(open(exp_meta_path))
    # annos = anno_dict['videos']
    annos = exp_dict['videos']
    annos_out = {}
    for vid in annos.keys():
        if vid not in annos_out.keys():
            annos_out[vid] = {}
        obj_ids = []
        for exp_info_id in annos[vid]['expressions'].keys():
            obj_id = annos[vid]['expressions'][exp_info_id]['obj_id']
            if obj_id not in obj_ids:
                obj_ids.append(obj_id)
        for frame_id in annos[vid]['frames']:
            if frame_id not in annos_out[vid].keys():
                annos_out[vid][frame_id] = []
            annos_out[vid][frame_id] = obj_ids
    return annos_out

@ torch.no_grad()
def generate_mask(anno_dict, video_path, anno_path, save_path, model, cuda=True):
    for vid in tqdm(anno_dict.keys()):
        video_save_path = os.path.join(save_path, vid)
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        for frame_id in anno_dict[vid].keys():
            if not os.path.exists(os.path.join(video_save_path, "{}.h5".format(frame_id))):
                obj_ids = anno_dict[vid][frame_id]
                frame = Image.open(os.path.join(video_path, vid, frame_id+'.jpg')).convert('RGB')
                mask = Image.open(os.path.join(anno_path, vid, frame_id+'.png')).convert('P')
                frame = transform(frame)
                mask = np.array(mask)
                h, w = mask.shape

                centers = []
                bboxes = []
                centers_norm = []
                instance_valid = []
                obj_ids = [int(id) for id in obj_ids]
                for obj_id in obj_ids:
                    mask_cur = ((mask==obj_id) * 255).astype(np.uint8)
                    if (mask_cur > 0).any():
                        dist = cv2.distanceTransform(
                            mask_cur, cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL
                        )
                        _, _, _, center = cv2.minMaxLoc(dist)
                        center_norm = (center[0] / w, center[1] / h)
                        y1, y2, x1, x2 = bounding_box(mask_cur)
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

                masks_point = model(frame[None], centers_norm, instance_valid, "point")
                masks_bbox = model(frame[None], bboxes, instance_valid, "bbox")
                masks_point = masks_point[0].cpu().numpy()
                masks_bbox = masks_bbox[0].cpu().numpy()
                out_annos = h5py.File(
                        os.path.join(video_save_path, "{}.h5".format(frame_id)), "w"
                    )
                out_annos.create_dataset("obj_ids", data=obj_ids)
                out_annos.create_dataset("heatBBox", data=masks_bbox)
                out_annos.create_dataset("heatPoint", data=masks_point)
                out_annos.create_dataset("centerPoint", data=centers)
                out_annos.close()
    


if __name__ == "__main__":
    video_path = "/media/HardDisk_B/Users/wx/wwk_files/datasets/referring_video_segmentation/Refer-YouTube-VOS/train/JPEGImages/"
    anno_path = "/media/HardDisk_B/Users/wx/wwk_files/datasets/referring_video_segmentation/Refer-YouTube-VOS/train/Annotations/"
    save_path = "./anno_weak/ref-youtube-vos/train/AnnotationsWeakly/"
    meta_path = "/media/HardDisk_B/Users/wx/wwk_files/datasets/referring_video_segmentation/Refer-YouTube-VOS/train/meta.json"
    exp_meta_path = "/media/HardDisk_B/Users/wx/wwk_files/datasets/referring_video_segmentation/Refer-YouTube-VOS/meta_expressions/train/meta_expressions.json"
    dilation = False
    cuda = True

    model = SimModel("resnet101", dilation)
    if cuda:
        model.cuda()

    annos_by_frame = transform_anno_to_each_frame(meta_path, exp_meta_path)
    generate_mask(annos_by_frame, video_path, anno_path, save_path, model, cuda)
    