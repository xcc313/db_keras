import cv2
import imgaug.augmenters as iaa
import math
import random
import numpy as np
import os.path as osp
import pyclipper
from shapely.geometry import Polygon
from core.tools.dataset_traversal import SimpleReader

def generate(img_set_dir, label_file_path, batch_size=16, image_size=640, is_training=True):

    if is_training:
        process = SimpleReader({'img_set_dir': img_set_dir, 'image_shape': (3, image_size, image_size)})
    else:
        process = SimpleReader({'img_set_dir': img_set_dir, 'image_shape': (3, 736, 1280)})

    with open(label_file_path, "rb") as fin:
        label_infor_list = fin.readlines()
    img_num = len(label_infor_list)
    img_id_list = list(range(img_num))
    random.shuffle(img_id_list)

    b = 0
    current_idx = 0

    while True:
        if current_idx >= img_num:
            if is_training:
                np.random.shuffle(img_id_list)
            current_idx = 0

        label_infor = label_infor_list[img_id_list[current_idx]]
        outs = process(label_infor)
        image, gt, mask, thresh_map, thresh_mask = outs
        if b == 0:
            # Init batch arrays
            batch_images = np.zeros([batch_size, image_size, image_size, 3], dtype=np.float32)
            batch_gts = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_masks = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_thresh_maps = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_thresh_masks = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_loss = np.zeros([batch_size, ], dtype=np.float32)

        batch_images[b] = image
        batch_gts[b] = gt
        batch_masks[b] = mask
        batch_thresh_maps[b] = thresh_map
        batch_thresh_masks[b] = thresh_mask
        b += 1
        current_idx += 1
        if b == batch_size:
            inputs = [batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks]
            outputs = batch_loss
            yield inputs, outputs
            b = 0
