import tensorflow as tf

class ResNet(object):
    def __init__(self, params):
        None

    def __call__(self, input):
        #VC  3 (3,3) replace (7,7)
        x = self.conv_bn_layer(input= input, num_filters=32, kernel_size=(3, 3), strides=(2, 2), act="relu",
                               name="conv1_1")
        x = self.conv_bn_layer(input=x, num_filters=32, kernel_size=(3, 3), strides=(1, 1), act="relu",
                               name="conv1_2")
        x = self.conv_bn_layer(input=x, num_filters=64, kernel_size=(3, 3), strides=(1, 1), act="relu",
                               name="conv1_3")
        conv = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        depth = [3, 4, 6, 3]
        num_filters = [64, 128, 256, 512]
        outs = []

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)
                conv = self.basic_block(
                    input=conv,
                    num_filters=num_filters[block],
                    strides=(2, 2) if i == 0 and block != 0 else (1, 1),
                    name=conv_name,
                    if_first=block == i == 0
                    )
            outs.append(conv)

        return outs


    def shortcut(self, input, ch_out, strides, name, if_first=False):
        ch_in = input.shape[3]
        if ch_in != ch_out or strides != (1, 1):
            if if_first:
                return self.conv_bn_layer(input, ch_out, (1, 1), strides, name=name)
            else:
                return self.conv_bn_layer_vd(input, ch_out, (1, 1), name=name)
        elif if_first:
            return self.conv_bn_layer(input, ch_out, (1, 1), strides, name=name)
        else:
            return input


    def basic_block(self, input, num_filters, strides, name ,if_first):
        conv0 = self.conv_bn_layer(input=input, num_filters=num_filters, kernel_size=(3, 3), strides=strides, act="relu",
                                   name=name + "_branch2a")
        conv1 = self.conv_bn_layer(input=conv0, num_filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                                   name=name + "_branch2b")
        short = self.shortcut(input, num_filters, strides, name, if_first=if_first)

        add = tf.keras.layers.Add()([short, conv1])

        x = tf.keras.layers.Activation("relu", name=name + "_act")(add)

        return x

    def bottleneck_block(self, input, num_filters, strides, name, if_first):
        conv0 = self.conv_bn_layer(input=input, num_filters=num_filters, kernel_size=(1, 1), strides=(1, 1), act="relu",
                                   name=name + "_branch2a")
        conv1 = self.conv_bn_layer(input=conv0, num_filters=num_filters, kernel_size=(3, 3), strides=strides, act="relu",
                                   name=name + "_branch2b")
        conv2 = self.conv_bn_layer(input=conv1, num_filters=num_filters * 4, kernel_size=(1, 1), strides=(1, 1), act=None,
                                   name=name + "_branch2c")

        short = self.shortcut(input, num_filters * 4, strides, name=name + "_branch1", if_first=if_first)

        add = tf.keras.layers.Add()([short, conv2])

        x = tf.keras.layers.Activation("relu", name=name + "_act")(add)
        return x


    def conv_bn_layer(self, input, num_filters, kernel_size, strides, act=None, name=None):
        x = tf.keras.layers.Conv2D(num_filters, kernel_size, strides=strides, use_bias=False, name=name, padding="same")(input)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + "_bn")(x)
        if act:
            x = tf.keras.layers.Activation(act, name=name + "_act")(x)
        return x


    def conv_bn_layer_vd(self, input, num_filters, kernel_size, act=None, name=None):
        pool = tf.keras.layers.AvgPool2D(strides=(2, 2))(input)
        padding = "valid" #if (num_filters - 1) // 2 == 0 else (1, 1)
        conv = tf.keras.layers.Conv2D(num_filters, kernel_size, strides=(1, 1), name=name, padding=padding)(pool)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + "_bn")(conv)
        if act:
            x = tf.keras.layers.Activation(act, name=name + "_act")(x)
        return x


