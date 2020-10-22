# -*- coding: utf-8 -*-
import tensorflow as tf
import shutil


class BestKeepCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self,
                 save_path,
                 eval_model,
                 only_save_weight=True):
        super(BestKeepCheckpoint, self).__init__()
        self.save_path = save_path
        self.eval_model = eval_model
        self.only_save_weight = only_save_weight

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 3 == 0:
            save_path = self.save_path.format(epoch=epoch)
            print("save model " + save_path)
            self.eval_model.save(save_path)
            shutil.copy(save_path, "/content/drive/My\ Drive/ICDAR_2019_LSVT/")
