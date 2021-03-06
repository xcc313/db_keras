from absl import app, flags
import cv2
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
import tensorflow as tf
import numpy as np
from core.tools import build_cfg, generate_rec, DotDict
from core.model.r34vd_crnn import RecModel

flags.DEFINE_string('config', './configs/rec_r34_vd_ctc.yml', 'path to config file')
FLAGS = flags.FLAGS

def decode_ctc(args):
    """returns a list of decoded ctc losses"""

    y_pred, input_length = args

    ctc_decoded = tf.keras.backend.ctc_decode(
        y_pred, input_length, greedy=True)

    return ctc_decoded

def main(_argv):
    print(FLAGS.config)
    cfg = build_cfg(FLAGS.config)
    cfg.update({'mode': 'train'})
    cfg = DotDict.to_dot_dict(cfg)

    _, inference_model, _ = RecModel(cfg)()
    inference_model.load_weights(r"E:\dm\model_weights\db_rec_26.h5", by_name=True, skip_mismatch=True)

    try:
        val_generator = generate_rec(cfg['test'], cfg, is_training=False)
        inputs, _ = next(val_generator)
        # make a batch of data
        batch_imgs, batch_labels, input_length, label_lens = inputs
        y_preds = inference_model.predict(batch_imgs)

        pred_tensor, _ = decode_ctc([y_preds, np.squeeze(input_length)])

        for idx, val in enumerate(pred_tensor[0]):

            image = batch_imgs[idx]
            # map back to strings
            predictions = cfg.char_ops.decode(val)
            print(predictions)
            cv2.imshow('image', image)
            cv2.waitKey(0)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run(main)
