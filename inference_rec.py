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

flags.DEFINE_string('config', './configs/rec_r34_vd_ctc_captcha.yml', 'path to config file')
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
    cfg['test'].update({'batch_size': 2})
    cfg = DotDict.to_dot_dict(cfg)

    _, inference_model, _ = RecModel(cfg)()
    inference_model.load_weights('./checkpoints/2025-02-06/db_09_1.5797_1.9254.h5', by_name=True, skip_mismatch=True)
    try:
        val_generator = generate_rec(cfg['test'], cfg, is_training=False)
        inputs, _ = next(val_generator)
        # make a batch of data
        batch_imgs, batch_labels, input_length, label_lens = inputs
        y_preds = inference_model.predict(batch_imgs)

        pred_tensor, _ = decode_ctc([y_preds, np.squeeze(input_length)])

        preds_idx = y_preds.argmax(axis=2)
        preds_prob = y_preds.max(axis=2)
        for idx, val in enumerate(preds_idx):

            image = batch_imgs[idx]
            label_len = label_lens[idx]
            label = batch_labels[idx]
            label = label[:int(label_len)]
            # map back to strings
            predictions = cfg.char_ops.decode(val)
            real = cfg.char_ops.decode(label)
            print("{}={}~{}".format(real == predictions,real, predictions))
        for idx, val in enumerate(pred_tensor[0]):

            image = batch_imgs[idx]
            label_len = label_lens[idx]
            label = batch_labels[idx]
            label = label[:int(label_len)]
            # map back to strings
            predictions = cfg.char_ops.decode(val)
            real = cfg.char_ops.decode(label)
            print("{}={}~{}".format(real == predictions,real, predictions))
            cv2.imshow('image', image)
            cv2.waitKey(0)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run(main)
