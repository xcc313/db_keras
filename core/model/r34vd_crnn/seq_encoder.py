from keras import layers


class SequenceEncoder(object):
    def __init__(self, params):
        super(SequenceEncoder, self).__init__()
        self.char_num = params.char_ops.get_char_num()
        self.rnn_hidden_size = params['hidden_size']
    def __call__(self, inputs):
        """CNN  block and flattening to time distributed layer from
        the original image of shape ( height, width, 3) will be mapped to (width/4, 512)"""
        #transpose because the "time axis" is width
        inputs = layers.Permute(dims=(2, 1, 3))(inputs)
        # flatten the height and channels to one dimension, after this the dimension
        # is batch, width, len(height)*len(channels)
        encoder_features = layers.TimeDistributed(layers.Flatten())(inputs)

        x = layers.Bidirectional(layers.LSTM(self.rnn_hidden_size, return_sequences=True, use_bias=True,
                                             recurrent_activation='sigmoid'))(encoder_features)
        x = layers.TimeDistributed(layers.Dense(self.rnn_hidden_size))(x)
        x = layers.Bidirectional(layers.LSTM(self.rnn_hidden_size, return_sequences=True, use_bias=True,
                                             recurrent_activation='sigmoid'))(x)
        # the +1 stands for a blank character needed in CTC
        fc= layers.TimeDistributed(layers.Dense(self.char_num + 1), name="DenseSoftmax")(x)
        return fc
