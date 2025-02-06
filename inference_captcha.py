from absl import app, flags
import cv2
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
from core.tools import build_cfg, DotDict
from core.tools.img_tools import resize_norm_img
from core.model.r34vd_crnn import RecModel


flags.DEFINE_string('config', './configs/rec_r34_vd_ctc_captcha.yml', 'path to config file')
FLAGS = flags.FLAGS

def decode_ctc(args):
    """returns a list of decoded ctc losses"""

    y_pred, input_length = args

    ctc_decoded = tf.keras.backend.ctc_decode(
        y_pred, input_length, greedy=True)

    return ctc_decoded


def convert_pd(inference_model):
    full_model = tf.function(lambda Input: inference_model(Input))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(inference_model.inputs[0].shape, inference_model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="captcha_r34_ctc-160x32-C4_model.pb",
                      as_text=False)

def main(_argv):
    print(FLAGS.config)
    cfg = build_cfg(FLAGS.config)
    cfg.update({'mode': 'train'})
    cfg = DotDict.to_dot_dict(cfg)

    _, inference_model, _ = RecModel(cfg)()
    inference_model.load_weights(r"E:\dm\repo\DB_keras\checkpoints\2022-04-29/db_02_9.4414_5.7740.h5", by_name=True, skip_mismatch=True)

    try:
        convert_pd(inference_model)
        # inference_model.save("./captcha_model_0.1294_0.6092.h5")
        image_shape = (32, 160, 3)
        batch_imgs = []
        image_dir = "E:/dm/tmp/90x42_predict"
        for f in os.listdir(image_dir):
            img = cv2.imread(image_dir + "/" + f)
            outs = resize_norm_img(img, image_shape)
            batch_imgs.append(outs)

        # make a batch of data
        batch_imgs = np.array(batch_imgs)
        y_preds = inference_model.predict(batch_imgs)

        pred_tensor, _ = decode_ctc([y_preds, [image_shape[1]/4] * batch_imgs.shape[0]])

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
