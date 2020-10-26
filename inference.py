from absl import app, flags
import cv2
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
import numpy as np
import datetime
import pickle
from core.tools import build_cfg, generate
from core.tools.db_process import DBProcessTest
from core.tools.db_postprocess import DBPostProcess
from core.model.r50vd_db import DetModel

flags.DEFINE_string('config', './configs/det_r50_vd_db.yml', 'path to config file')
FLAGS = flags.FLAGS


def main(_argv):
    print(FLAGS.config)
    cfg = build_cfg(FLAGS.config)
    cfg.update({'mode': 'train'})

    model_algorithm = cfg['det']['algorithm']
    if model_algorithm == 'DB':
        _, inference_model = DetModel(cfg)()
        inference_model.load_weights(r"E:\dm\model_weights\db_inference_model.h5", by_name=True, skip_mismatch=True)
        inference_model.summary()
    else:
        raise NotImplementedError('%s not support yet !' % model_algorithm)

    try:

        process = DBProcessTest(cfg)
        postprocess = DBPostProcess(cfg)
        for image_path in glob.glob(osp.join(r'E:\dm\repo\CHINESE-OCR\ctpn\data\demo', '*.jpg')):
            print(image_path)
            im = cv2.imread(image_path)
            src_image = im.copy()
            im, ratio_list = process(im)
            p = inference_model.predict(im)[0]

            boxes = postprocess(p, [ratio_list])[0]

            for box in boxes:
                box = np.array(box).astype(np.int32).reshape(-1, 2)
                cv2.polylines(src_image, [box], True, color=(255, 255, 0), thickness=2)

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', src_image)
            cv2.waitKey(0)
            image_fname = osp.split(image_path)[-1]
            cv2.imwrite('test/' + image_fname, src_image)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run(main)
