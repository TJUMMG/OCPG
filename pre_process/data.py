import cv2
import os
from PIL import Image
from torchvision import transforms
import numpy as np


def img_transform(img, annos):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    h, w, _ = img.shape
    centers = []
    centers_norm = []
    label_list = list(np.unique(annos))
    label_list.remove(0)
    for label in label_list:
        anno = (annos == label).astype(np.uint8) * 255
        dist = cv2.distanceTransform(anno, cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
        _, _, _, center = cv2.minMaxLoc(dist)
        center_norm = (center[0] / w, center[1] / h)
        centers.append(center)
        centers_norm.append(center_norm)
    img = Image.fromarray(img)
    img = transform(img)
    return img, centers, centers_norm


def load_img_davis(img_path, anno_path):
    imgs = os.listdir(img_path)
    out_pairs = {}
    for img in imgs:
        out_pairs[img.split(".")[0]] = {}
        frame = cv2.imread(os.path.join(img_path, img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_pairs[img.split(".")[0]]["frame"] = frame
        anno = cv2.imread(
            os.path.join(
                anno_path,
                img.split(".")[0] + ".png",
            )
        )
        out_pairs[img.split(".")[0]]["label"] = anno
    return out_pairs


def load_video_a2d(video_path, anno_path):
    annos = os.listdir(anno_path)
    out_pairs = {}

    for anno in annos:
        out_pairs[str(int(anno.split(".")[0]))] = {}
        ann_img = cv2.imread(os.path.join(anno_path, anno))
        out_pairs[str(int(anno.split(".")[0]))]["label"] = ann_img

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if str(idx + 1) in out_pairs.keys():
            ret, frame = cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_pairs[str(idx + 1)]["frame"] = frame
        idx += 1
    cap.release()
    return out_pairs
