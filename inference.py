from absl import app, flags
import cv2
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
import numpy as np
import datetime
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
        inference_model.load_weights(r"C:\Users\DL\Downloads\db_05_2.4870_4.3123.h5", by_name=True, skip_mismatch=True)
        inference_model.summary()
    else:
        raise NotImplementedError('%s not support yet !' % model_algorithm)

    try:

        process = DBProcessTest(cfg)
        postprocess = DBPostProcess(cfg)
        for image_path in glob.glob(osp.join(r'E:\dm\tmp\DB_predict', '*.jpg')):
            im = cv2.imread(image_path)
            src_image = im.copy()
            im, ratio_list = process(im)
            p = inference_model.predict(im)[0]
            boxes = postprocess(p, ratio_list)

            for box in boxes:
                cv2.drawContours(src_image, [np.array(box)], -1, (0, 255, 0), 2)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', src_image)
            cv2.waitKey(0)
            image_fname = osp.split(image_path)[-1]
            cv2.imwrite('test/' + image_fname, src_image)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run(main)
