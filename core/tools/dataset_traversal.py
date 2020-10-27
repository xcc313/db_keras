#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import math
import random
import numpy as np
import cv2
import imghdr

from .img_tools import process_image, process_image_srn, get_img_data


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists

class SimpleReader(object):
    def __init__(self, params):
        if params['mode'] != 'train':
            self.num_workers = 1
        else:
            self.num_workers = params['num_workers']
        if params['mode'] != 'test':
            self.img_set_dir = params['img_set_dir']
            self.label_file_path = params['label_file_path']
        self.use_gpu = params['use_gpu']
        self.char_ops = params['char_ops']
        self.image_shape = params['image_shape']
        self.loss_type = params['loss_type']
        self.max_text_length = params['max_text_length']
        self.mode = params['mode']
        self.infer_img = params['infer_img']
        self.use_tps = False
        if "num_heads" in params:
            self.num_heads = params['num_heads']
        if "tps" in params:
            self.use_tps = True
        self.use_distort = False
        if "distort" in params:
            self.use_distort = params['distort'] and params['use_gpu']
            if not params['use_gpu']:
                print(
                    "Distort operation can only support in GPU.Distort will be set to False."
                )
        if params['mode'] == 'train':
            self.batch_size = params['train_batch_size_per_card']
            self.drop_last = True
        else:
            self.batch_size = params['test_batch_size_per_card']
            self.drop_last = False
            self.use_distort = False

    def __call__(self, process_id):
        if self.mode != 'train':
            process_id = 0

        def get_device_num():
            if self.use_gpu:
                gpus = os.environ.get("CUDA_VISIBLE_DEVICES", '1')
                gpu_num = len(gpus.split(','))
                return gpu_num
            else:
                cpu_num = os.environ.get("CPU_NUM", 1)
                return int(cpu_num)

        def sample_iter_reader():
            if self.mode != 'train' and self.infer_img is not None:
                image_file_list = get_image_file_list(self.infer_img)
                for single_img in image_file_list:
                    img = cv2.imread(single_img)
                    if img.shape[-1] == 1 or len(list(img.shape)) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    if self.loss_type == 'srn':
                        norm_img = process_image_srn(
                            img=img,
                            image_shape=self.image_shape,
                            char_ops=self.char_ops,
                            num_heads=self.num_heads,
                            max_text_length=self.max_text_length)
                    else:
                        norm_img = process_image(
                            img=img,
                            image_shape=self.image_shape,
                            char_ops=self.char_ops,
                            tps=self.use_tps,
                            infer_mode=True)
                    yield norm_img
            else:
                with open(self.label_file_path, "rb") as fin:
                    label_infor_list = fin.readlines()
                img_num = len(label_infor_list)
                img_id_list = list(range(img_num))
                random.shuffle(img_id_list)
                if sys.platform == "win32" and self.num_workers != 1:
                    print("multiprocess is not fully compatible with Windows."
                          "num_workers will be 1.")
                    self.num_workers = 1
                if self.batch_size * get_device_num(
                ) * self.num_workers > img_num:
                    raise Exception(
                        "The number of the whole data ({}) is smaller than the batch_size * devices_num * num_workers ({})".
                        format(img_num, self.batch_size * get_device_num() *
                               self.num_workers))
                for img_id in range(process_id, img_num, self.num_workers):
                    label_infor = label_infor_list[img_id_list[img_id]]
                    substr = label_infor.decode('utf-8').strip("\n").split("\t")
                    img_path = self.img_set_dir + "/" + substr[0]
                    img = cv2.imread(img_path)
                    if img is None:
                        print("{} does not exist!".format(img_path))
                        continue
                    if img.shape[-1] == 1 or len(list(img.shape)) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    label = substr[1]
                    if self.loss_type == "srn":
                        outs = process_image_srn(
                            img=img,
                            image_shape=self.image_shape,
                            num_heads=self.num_heads,
                            max_text_length=self.max_text_length,
                            label=label,
                            char_ops=self.char_ops,
                            loss_type=self.loss_type)

                    else:
                        outs = process_image(
                            img=img,
                            image_shape=self.image_shape,
                            label=label,
                            char_ops=self.char_ops,
                            loss_type=self.loss_type,
                            max_text_length=self.max_text_length,
                            distort=self.use_distort)
                    if outs is None:
                        continue
                    yield outs

        def batch_iter_reader():
            batch_outs = []
            for outs in sample_iter_reader():
                batch_outs.append(outs)
                if len(batch_outs) == self.batch_size:
                    yield batch_outs
                    batch_outs = []
            if not self.drop_last:
                if len(batch_outs) != 0:
                    yield batch_outs

        if self.infer_img is None:
            return batch_iter_reader
        return sample_iter_reader
