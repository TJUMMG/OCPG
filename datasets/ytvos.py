"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
import datasets.transforms_video as T
import torch.nn.functional as F
import os
from PIL import Image
import json
import numpy as np
import random
import h5py
from util.box_ops import center_of_mass

from datasets.categories import ytvos_category_dict as category_dict


def weight2mask(heatmaps: torch.Tensor, instance_index: int, thres=0.5):
    # heatmaps: [n, h, w]
    n, h, w = heatmaps.shape
    background = torch.full([1, h, w], thres).to(heatmaps)
    stacked_heatmap = torch.cat([heatmaps, background], dim=0)
    ins_mask = stacked_heatmap.argmax(dim=0)    # [h, w]
    final_mask = (ins_mask == instance_index).float()
    # final_mask = (heatmaps[instance_index] > thres).float()
    mask = final_mask.unsqueeze(0)

    width_proj = mask.max(1)[0]
    height_proj = mask.max(2)[0]
    box_width, box_height = width_proj.sum(1), height_proj.sum(1)
    center_ws, _ = center_of_mass(width_proj[:, None, :])
    _, center_hs = center_of_mass(height_proj[:, :, None])
    boxes = torch.stack([center_ws-0.5*box_width, center_hs-0.5*box_height, center_ws+0.5*box_width, center_hs+0.5*box_height], 1)
    return final_mask, boxes[0]


class YTVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """
    def __init__(self, args, img_folder: Path, ann_file: Path, transforms, return_masks: bool,
                 num_frames: int, max_skip: int, supervision):
        self.args = args
        self.supervision = supervision
        self.img_folder = img_folder     
        self.ann_file = ann_file
        if 'train' in str(self.img_folder):
            self.mode = "train"
        elif 'valid' in str(self.img_folder):
            self.mode = 'valid'
        else:
            raise NotImplementedError

        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames     
        self.max_skip = max_skip
        # create video meta data
        self.prepare_metas()       
        self.is_noun = lambda pos: pos[:2] == "NN"
        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        # for each video
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            # for each expression
            for exp_id, exp_dict in vid_data['expressions'].items():
                exp = exp_dict['exp']
                oid = int(exp_dict['obj_id'])
                # for each frame
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp
                    meta['obj_id'] = oid
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 

    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict
            video, exp, obj_id, category, frames, frame_id = \
                        meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
            exp = " ".join(exp.lower().split())
            category_id = category_dict[category]
            vid_len = len(frames)

            num_frames = self.num_frames
            sample_indx = [frame_id]
            if self.num_frames != 1:
                # local sample [before and after].
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # global sampling [in rest frames]
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >= global_n:
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()
            # random reverse
            if self.mode == "train" and np.random.rand() < 0.3:
                sample_indx = sample_indx[::-1]

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            weak_masks, weights = [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')
                f = h5py.File(os.path.join(str(self.img_folder), 'AnnotationsWeakly', video, frame_name + '.h5'))
                if self.supervision == 'box':
                    heatmaps = np.array(f["heatPoint"])
                elif self.supervision == 'point':
                    heatmaps = np.array(f["heatPoint"])
                heatmaps = torch.from_numpy(heatmaps)
                try:
                    instance_idx = list(f['obj_ids']).index(obj_id)
                    weak_mask, weak_box = weight2mask(heatmaps, instance_idx)
                    heatmap = heatmaps[instance_idx]
                except:
                    weak_mask = torch.zeros((heatmaps.shape[-2], heatmaps.shape[-1]))
                    heatmap = torch.zeros((heatmaps.shape[-2], heatmaps.shape[-1]))
                    # print(video, frame_name, list(f['obj_ids']), obj_id)
                
                weak_masks.append(weak_mask)
                weights.append(heatmap)

                # create the target
                label = torch.tensor(category_id)
                mask = np.array(mask)
                mask = (mask==obj_id).astype(np.float32)
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    if self.supervision == 'point':
                        box = weak_box
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    valid.append(0)
                mask = torch.from_numpy(mask).float()

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0) 
            boxes = torch.stack(boxes, dim=0) 
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)
            

            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'labels': labels,                        # [T,] class id of categories
                'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]  whether include object
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]),
                'size': torch.as_tensor([int(h), int(w)])
            }
            weak_masks = torch.stack(weak_masks, dim=0)
            weights = torch.stack(weights, dim=0)
            weak_masks = F.interpolate(weak_masks[None], (h, w), mode='bilinear', align_corners=True)[0]
            weights = F.interpolate(weights[None], (h, w), mode='bilinear', align_corners=True)[0]
            target['weights'] = weights
            target['weak_masks'] = weak_masks

            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)
            
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target


def make_coco_transforms(current_epoch, image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    # CLIP at first to save time
    if image_set == 'train':
        return T.Compose([
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            T.RandomHorizontalFlip(),
            # T.PhotometricDistort(),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(args, img_folder, ann_file, transforms=make_coco_transforms(args.current_epoch, image_set, max_size=args.max_size), return_masks=args.masks,
                           num_frames=args.num_frames, max_skip=args.max_skip, supervision=args.supervision)
    return dataset

