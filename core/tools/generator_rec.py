import cv2
import random
import numpy as np
from .img_tools import process_image
from core.tools.character import CharacterOps

def generate_rec(params, is_training=True):

    char_ops_params = {
        "character_type": params.character_type,
        "character_dict_path": params.rec_char_dict_path,
        "use_space_char": params.use_space_char,
        "max_text_length": params.max_text_length,
        "loss_type":'ctc'
    }

    with open(params.label_file_path, "rb") as fin:
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
        substr = label_infor.decode('utf-8').strip("\n").split("\t")
        img_path = params.img_set_dir + "/" + substr[0]
        img = cv2.imread(img_path)
        if img is None:
            print("{} does not exist!".format(img_path))
            continue
        if img.shape[-1] == 1 or len(list(img.shape)) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        label = substr[1]

        outs = process_image(
            img=img,
            image_shape=params.image_shape,
            label=label,
            char_ops=CharacterOps(char_ops_params),
            loss_type=params.loss_type,
            max_text_length=params.max_text_length,
            distort=params.use_distort)

        image, gt, mask, thresh_map, thresh_mask = outs
        if b == 0:
            # Init batch arrays
            batch_images = np.zeros([batch_size, image_size, image_size, 3], dtype=np.float32)
            batch_gts = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_loss = np.zeros([batch_size, ], dtype=np.float32)

        batch_images[b] = image
        batch_gts[b] = gt
        batch_masks[b] = mask

        b += 1
        current_idx += 1
        if b == batch_size:
            inputs = [batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks]
            outputs = batch_loss
            yield inputs, outputs
            b = 0
