import cv2
import random
import numpy as np
from copy import deepcopy
from .img_tools import process_image

def generate_rec(params, globals , is_training=True):
    batch_size = params.batch_size
    image_shape = deepcopy(params.image_shape)
    image_shape.insert(0, -1)

    with open(params.label_path, "rb") as fin:
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
        img_path = params.img_dir + "/" + substr[0]
        img = cv2.imread(img_path)
        if img is None:
            print("{} does not exist!".format(img_path))
            continue
        if img.shape[-1] == 1 or len(list(img.shape)) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        label = substr[1]

        # the number of time distributed values, or another words the length of the time axis in the output,
        # or equivalently the width of the image after convolutions. Needed to input in the CTC loss
        # each maxpooling halves the width dimension so in our model scaling is 1/4 with 2 maxpoolings
        t_dist_dim = int(params.image_shape[1] / 4)

        outs = process_image(
            img=img,
            image_shape=params.image_shape,
            label=label,
            char_ops=globals.char_ops,
            loss_type=globals.loss_type,
            max_text_length=globals.max_text_length,
            distort=params.use_distort)

        image, gt = outs
        if b == 0:
            # Init batch arrays
            batch_images = np.zeros(batch_size, dtype=np.float32)
            batch_gts = np.zeros(batch_size, dtype=np.float32)
            batch_label_length = np.zeros(batch_size, dtype=np.float32)
            batch_input_length = np.zeros(batch_size, dtype=np.float32)
            batch_loss = np.zeros([batch_size, ], dtype=np.float32)

        batch_images[b] = image
        batch_gts[b] = gt
        batch_label_length[b] = len(gt)
        batch_input_length[b] = t_dist_dim

        b += 1
        current_idx += 1
        if b == batch_size:
            inputs = [batch_images, batch_gts]
            outputs = batch_loss
            yield [inputs, label, ], outputs
            b = 0
