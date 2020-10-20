import tensorflow as tf
from keras import layers, models

class DBHead(object):
    def __init__(self, params):
        self.__name__ = 'DBHead'
        self.mode = params['mode']
        self.k = params['det']['k']

    def binarize(self, fuse):
        p = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
        p = layers.BatchNormalization()(p)
        p = layers.ReLU()(p)
        p = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
        p = layers.BatchNormalization()(p)
        p = layers.ReLU()(p)
        p = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                   activation='sigmoid')(p)
        return p


    def thresh(self, fuse):
        t = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
        t = layers.BatchNormalization()(t)
        t = layers.ReLU()(t)
        t = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(t)
        t = layers.BatchNormalization()(t)
        t = layers.ReLU()(t)
        t = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                   activation='sigmoid')(t)
        return t

    def __call__(self, input):

        C2, C3, C4, C5 = input

        in2 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
        in3 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
        in4 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
        in5 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)

        # 1 / 32 * 8 = 1 / 4
        P5 = layers.UpSampling2D(size=(8, 8))(
            layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))
        # 1 / 16 * 4 = 1 / 4
        out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(in5)])
        P4 = layers.UpSampling2D(size=(4, 4))(
            layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4))
        # 1 / 8 * 2 = 1 / 4
        out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(out4)])
        P3 = layers.UpSampling2D(size=(2, 2))(
            layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3))
        # 1 / 4
        P2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
            layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(out3)]))
        # (b, /4, /4, 256)
        fuse = layers.Concatenate()([P2, P3, P4, P5])

        # probability map
        p = self.binarize(fuse)
        if self.mode != "train":
            return p, None, None

        # threshold map
        t = self.thresh(fuse)
        # approximate binary map
        b_hat = layers.Lambda(lambda x: 1 / (1 + tf.exp(-self.k * (x[0] - x[1]))))([p, t])
        # y = layers.Concatenate(axis=3)([])
        return p, t, b_hat
