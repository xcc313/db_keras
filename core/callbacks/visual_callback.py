import tensorflow as tf
import numpy as np

def decode_ctc(args):
    """returns a list of decoded ctc losses"""

    y_pred, input_length = args

    ctc_decoded = tf.keras.backend.ctc_decode(
        y_pred, input_length, greedy=True)

    return ctc_decoded


class PredVisualize(tf.keras.callbacks.Callback):

    def __init__(self, model, val_datagen, char_ops, printing=False):
        """CTC decode the results and visualize output"""
        self.model = model
        self.datagen = iter(val_datagen)
        self.char_ops = char_ops

    def on_epoch_end(self, batch, logs=None):

        inputs, _ = next(self.datagen)
        #make a batch of data
        batch_imgs, batch_labels, input_length, label_lens = inputs

        #predict from batch
        y_preds = self.model.predict(batch_imgs)

        #call the ctc decode
        pred_tensor, _ = decode_ctc([y_preds, np.squeeze(input_length)])
        pred_labels = tf.keras.backend.get_value(pred_tensor[0])

        #map back to strings
        predictions = [ self.char_ops.decode(word) for word in pred_labels.tolist()]
        truths = [self.char_ops.decode(word) for word in batch_labels.tolist()]

        print('predictions {}   truths {}'.format(predictions, truths))
