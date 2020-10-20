import cv2
import imgaug.augmenters as iaa
import math
import random
import numpy as np
import os.path as osp
import pyclipper
from shapely.geometry import Polygon
from core.tools.db_process import DBProcessTrain, DBProcessTest


def show_polys(image, anns, window_name):
    for ann in anns:
        poly = np.array(ann['poly']).astype(np.int32)
        cv2.drawContours(image, np.expand_dims(poly, axis=0), -1, (0, 255, 0), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)


def draw_thresh_map(polygon, canvas, mask, shrink_ratio=0.4):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = np.min(distance_map, axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymin,
            xmin_valid - xmin:xmax_valid - xmin],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


def compute_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result


def generate(img_set_dir, label_file_path, batch_size=16, image_size=640, is_training=True):

    if is_training:
        process = DBProcessTrain({'img_set_dir': img_set_dir, 'image_shape': (3, image_size, image_size)})
    else:
        process = DBProcessTest({'img_set_dir': img_set_dir, 'image_shape': (3, 736, 1280)})

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
